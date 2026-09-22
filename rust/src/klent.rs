//! Single-threaded KLENT collection, compact storage and Python tensor encoding.
use crate::{game::{board_index, Rng, MARK_CLASS_SQUARES, NO_MARK_CLASS, TRACK_MARK_CLASSES}, Game, Scratch};
use std::panic::{catch_unwind, AssertUnwindSafe};

pub fn bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
}
fn float(x: u16) -> f32 { f32::from_bits((x as u32) << 16) }

#[repr(C)]
#[derive(Clone)]
pub struct Record<T, D> {
    board: [u8; 20],
    scores: [u8; 2],
    mark_classes: [i8; MARK_CLASS_SQUARES],
    actions: [u8; 2],
    policy: [[u16; 80]; 2],
    returns: T,
    discounted_scores: D,
}
type Position = Record<[u16; 2], [[u16; 81]; 2]>;
type History = Record<[f32; 2], ()>;
const _: () = assert!(std::mem::size_of::<Position>() == 672 + MARK_CLASS_SQUARES);
const _: () = assert!(std::mem::size_of::<History>() == 352 + MARK_CLASS_SQUARES);

fn finish(history: &mut Vec<History>, reward: [f32; 2], terminal_scores: [u8; 2],
          lambda: f32, score_lambda: f32, buffer: &mut Vec<Position>) {
    let mut ret = reward;
    let mut score_dist = [[0.0f32; 81]; 2];
    for p in 0..2 {
        assert!(terminal_scores[p] <= 80);
        score_dist[p][terminal_scores[p] as usize] = 1.0;
    }
    for t in (0..history.len()).rev() {
        if t+1 < history.len() {
            for p in 0..2 { ret[p] = (1.0-lambda)*history[t+1].returns[p] + lambda*ret[p]; }
        }
        // The following iteration needs V_t, so write into a separate destination.
        let h = &history[t];
        for p in 0..2 {
            assert!(h.scores[p] <= 80);
            for value in &mut score_dist[p] { *value *= score_lambda; }
            score_dist[p][h.scores[p] as usize] += 1.0-score_lambda;
        }
        buffer.push(Position { board: h.board, scores: h.scores, mark_classes: h.mark_classes,
            actions: h.actions, policy: h.policy, returns: ret.map(bf16),
            discounted_scores: score_dist.map(|dist| dist.map(bf16)) });
    }
    let start = buffer.len()-history.len();
    buffer[start..].reverse();
    history.clear();
}

fn pack(cells: &[u8; 80]) -> [u8; 20] {
    std::array::from_fn(|i| (0..4).fold(0, |v, j| v | cells[i*4+j] << (2*j)))
}
fn unpack(board: &[u8; 20]) -> [u8; 80] {
    std::array::from_fn(|i| (board[i/4] >> (2*(i%4))) & 3)
}

/// BF16 one-hot planes: unplayable, empty, removed, own, opponent.
fn encode(cells: &[u8; 80], scores: [u8; 2], player: usize, board: &mut [u16], score: &mut [u16]) {
    board.fill(0);
    for cell in 0..160 { if (cell/16 + cell%16)%2 != 0 { board[cell] = bf16(1.0); } }
    for (sq, &mark) in cells.iter().enumerate() {
        let plane = match mark { 0 => 1, 3 => 2, 1 => 3+player, 2 => 4-player, _ => unreachable!() };
        board[plane*160 + board_index(sq)] = bf16(1.0);
    }
    score[0] = bf16(scores[player] as f32);
    score[1] = bf16(scores[1-player] as f32);
}

fn policy(game: &Game, player: usize, logits: &[f32], q: &[f32], alpha: f32, beta: f32) -> ([f32; 80], f32) {
    let legal = game.legal_moves(player);
    let mut p = [0.0; 80];
    let mut max = f32::NEG_INFINITY;
    for a in 0..80 {
        if legal[a/64] >> (a%64) & 1 != 0 {
            // log-softmax differs from logits by a constant, cancelled by softmax.
            p[a] = (q[a] + beta*logits[a])/(alpha+beta);
            assert!(p[a].is_finite(), "nonfinite model output");
            max = max.max(p[a]);
        }
    }
    let mut sum = 0.0;
    for a in 0..80 {
        p[a] = if legal[a/64] >> (a%64) & 1 != 0 { (p[a]-max).exp() } else { 0.0 };
        sum += p[a];
    }
    assert!(sum > 0.0);
    let mut value = 0.0;
    for a in 0..80 { p[a] /= sum; if p[a] > 0.0 { value += p[a]*q[a]; } }
    (p, value)
}
fn sample(p: &[f32; 80], rng: &mut Rng) -> usize {
    let threshold = rng.random() as f32;
    let mut sum = 0.0;
    let mut last = 0;
    for (i, &v) in p.iter().enumerate() {
        if v > 0.0 { last = i; sum += v; if threshold < sum { return i; } }
    }
    last
}

pub struct Arena {
    games: Vec<Game>, histories: Vec<Vec<History>>, buffer: Vec<Position>,
    capacity: usize, max_ply: usize, alpha: f32, beta: f32, lambda: f32, score_lambda: f32,
    rng: Rng, shuffle_rng: Rng, scratch: Scratch, evaluation: bool, pub results: [usize; 3],
}
impl Arena {
    pub fn new(n: usize, capacity: usize, max_ply: usize, alpha: f32, beta: f32, lambda: f32,
               score_lambda: f32, seed: u64, evaluation: bool) -> Self {
        assert!(n > 0 && max_ply == 80);
        assert!(evaluation || capacity >= n.checked_mul(max_ply).unwrap());
        assert!(alpha.is_finite() && beta.is_finite() && alpha >= 0.0 && beta >= 0.0 && alpha+beta > 0.0);
        assert!(lambda.is_finite() && (0.0..=1.0).contains(&lambda));
        assert!(score_lambda.is_finite() && (0.0..=1.0).contains(&score_lambda));
        Self { games: vec![Game::new(); n], histories: (0..n).map(|_| Vec::with_capacity(if evaluation {0} else {max_ply})).collect(),
            buffer: Vec::with_capacity(if evaluation {0} else {capacity}), capacity, max_ply, alpha, beta, lambda, score_lambda,
            rng: Rng::new(seed), shuffle_rng: Rng::new(seed ^ 0x9e37_79b9_7f4a_7c15),
            scratch: Scratch::new(), evaluation, results: [0; 3] }
    }
    pub fn inputs(&self, boards: &mut [u16], scores: &mut [u16]) {
        boards.fill(0); scores.fill(0);
        for (i, g) in self.games.iter().enumerate() {
            if g.finished { continue; }
            for p in 0..2 {
                let row = 2*i+p;
                encode(&g.cells, g.scores.map(|s| u8::try_from(s).unwrap()), p,
                    &mut boards[row*800..(row+1)*800], &mut scores[row*2..row*2+2]);
            }
        }
    }
    pub fn step(&mut self, logits: &[f32], q: &[f32]) -> bool {
        for i in 0..self.games.len() {
            if self.games[i].finished { continue; }
            let mut policies = [[0u16; 80]; 2];
            let mut values = [0.0; 2];
            let mut actions = [0u8; 2];
            for p in 0..2 {
                let row = (2*i+p)*80;
                let (dist, value) = if self.evaluation {
                    policy(&self.games[i], p, &logits[row..row+80], &[0.0; 80], 0.0, 1.0)
                } else { policy(&self.games[i], p, &logits[row..row+80], &q[row..row+80], self.alpha, self.beta) };
                policies[p] = dist.map(bf16); values[p] = value;
                actions[p] = sample(&dist, &mut self.rng) as u8;
            }
            if !self.evaluation {
                assert!(self.histories[i].len() < self.max_ply);
                self.histories[i].push(History { board: pack(&self.games[i].cells),
                    scores: self.games[i].scores.map(|s| u8::try_from(s).unwrap()),
                    mark_classes: self.games[i].mark_classes, actions, policy: policies, returns: values,
                    discounted_scores: () });
            }
            self.games[i].action_step(actions[0] as usize, actions[1] as usize, &mut self.scratch);
            if self.games[i].finished {
                let reward = self.games[i].terminal_values();
                if self.evaluation {
                    let new_player = if i < self.games.len()/2 {1} else {0};
                    self.results[if reward[new_player]>0.0 {0} else if reward[new_player]==0.0 {1} else {2}] += 1;
                }
                else {
                    let history = &mut self.histories[i];
                    assert!(self.buffer.len()+history.len() <= self.capacity);
                    let terminal_scores = self.games[i].scores.map(|s| u8::try_from(s).unwrap());
                    finish(history, reward, terminal_scores, self.lambda, self.score_lambda, &mut self.buffer);
                    self.games[i] = Game::new();
                }
            }
        }
        if self.evaluation { self.results.iter().sum::<usize>() == self.games.len() }
        else { self.capacity-self.buffer.len() < self.games.len()*self.max_ply }
    }
    pub fn reset(&mut self) -> usize {
        let dropped = self.histories.iter().map(Vec::len).sum();
        for h in &mut self.histories { h.clear(); }
        self.games.fill(Game::new()); dropped
    }
    pub fn shuffle(&mut self) {
        for i in (1..self.buffer.len()).rev() {
            let j = self.shuffle_rng.randint((i+1) as u64) as usize;
            self.buffer.swap(i,j);
        }
    }
}

// Project-private ABI. Python owns handles and validates contiguous tensor shapes.
// All fallible calls catch Rust panics rather than unwinding across the C boundary.
#[no_mangle]
pub extern "C" fn klent_new(n: usize, m: usize, max_ply: usize, alpha: f32, beta: f32, lambda: f32,
                            score_lambda: f32, seed: u64, evaluation: bool) -> *mut Arena {
    catch_unwind(|| Box::into_raw(Box::new(Arena::new(n,m,max_ply,alpha,beta,lambda,score_lambda,seed,evaluation)))).unwrap_or(std::ptr::null_mut())
}
#[no_mangle]
pub unsafe extern "C" fn klent_free(h: *mut Arena) { if !h.is_null() { drop(Box::from_raw(h)); } }
#[no_mangle]
pub extern "C" fn klent_mark_classes_enabled() -> bool { TRACK_MARK_CLASSES }
#[no_mangle]
pub unsafe extern "C" fn klent_inputs(h: *mut Arena, b: *mut u16, s: *mut u16) -> i32 {
    catch_unwind(AssertUnwindSafe(|| { let a=&*h; a.inputs(std::slice::from_raw_parts_mut(b,a.games.len()*1600), std::slice::from_raw_parts_mut(s,a.games.len()*4)); 0 })).unwrap_or(-1)
}
#[no_mangle]
pub unsafe extern "C" fn klent_step(h: *mut Arena, logits: *const f32, q: *const f32) -> i32 {
    catch_unwind(AssertUnwindSafe(|| { let a=&mut *h; let len=a.games.len()*160; a.step(std::slice::from_raw_parts(logits,len),std::slice::from_raw_parts(q,len)) as i32 })).unwrap_or(-1)
}
#[no_mangle]
pub unsafe extern "C" fn klent_stats(h: *mut Arena, out: *mut usize) {
    let a=&*h; std::slice::from_raw_parts_mut(out,4).copy_from_slice(&[a.buffer.len(),a.results[0],a.results[1],a.results[2]]);
}
#[no_mangle]
pub unsafe extern "C" fn klent_reset(h: *mut Arena) -> usize { (&mut *h).reset() }
#[no_mangle]
pub unsafe extern "C" fn klent_shuffle(h: *mut Arena) { (&mut *h).shuffle() }
#[no_mangle]
pub unsafe extern "C" fn klent_clear(h: *mut Arena) { (&mut *h).buffer.clear(); }
#[no_mangle]
pub unsafe extern "C" fn klent_batch(h: *mut Arena, start: usize, count: usize, b: *mut u16, s: *mut u16, pi: *mut u16, actions: *mut i64, returns: *mut f32, classes: *mut i8, discounted: *mut u16) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let a=&*h;
        let boards=std::slice::from_raw_parts_mut(b,count*1600);
        let scores=std::slice::from_raw_parts_mut(s,count*4);
        let policies=std::slice::from_raw_parts_mut(pi,count*320);
        let moves=std::slice::from_raw_parts_mut(actions,count*2);
        let targets=std::slice::from_raw_parts_mut(returns,count*2);
        let mark_classes=std::slice::from_raw_parts_mut(classes,count*160);
        let discounted_scores=std::slice::from_raw_parts_mut(discounted,count*324);
        boards.fill(0); scores.fill(0); policies.fill(0); moves.fill(0); targets.fill(0.0);
        mark_classes.fill(NO_MARK_CLASS);
        discounted_scores.fill(0);
        let end=(start+count).min(a.buffer.len());
        for (i,pos) in a.buffer[start..end].iter().enumerate() {
            for p in 0..2 {
                let row=2*i+p;
                encode(&unpack(&pos.board),pos.scores,p,&mut boards[row*800..(row+1)*800],&mut scores[row*2..row*2+2]);
                for sq in 0..80 { policies[row*160+board_index(sq)]=pos.policy[p][sq]; }
                moves[row]=board_index(pos.actions[p] as usize) as i64;
                targets[row]=float(pos.returns[p]);
                for q in 0..2 {
                    let start=row*162+q*81;
                    discounted_scores[start..start+81].copy_from_slice(&pos.discounted_scores[(p+q)%2]);
                }
                if TRACK_MARK_CLASSES {
                    for sq in 0..80 {
                        let class=pos.mark_classes[sq];
                        mark_classes[row*80+sq] = if class < 0 { class }
                            else if p == 0 { class }
                            else if class < 3 { class+3 } else { class-3 };
                    }
                }
            }
        }
        (end-start) as i32
    })).unwrap_or(-1)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn exact_returns_use_next_state_and_preserve_order() {
        for lambda in [0.0, 0.5, 1.0] {
            let mut history: Vec<History> = (0..3).map(|i| History {
                board: [i;20], scores: [0;2], mark_classes: [NO_MARK_CLASS;MARK_CLASS_SQUARES], actions: [0;2], policy: [[0;80];2],
                returns: [i as f32 * 0.25, -(i as f32)*0.25], discounted_scores: (),
            }).collect();
            let mut buffer = Vec::new();
            finish(&mut history, [1.0,-1.0], [2, 3], lambda, 0.94, &mut buffer);
            let g1=(1.0-lambda)*0.5+lambda;
            let g0=(1.0-lambda)*0.25+lambda*g1;
            for (i,g) in [g0,g1,1.0].into_iter().enumerate() {
                assert_eq!(buffer[i].returns,[bf16(g),bf16(-g)]);
                assert_eq!(buffer[i].board,[i as u8;20]);
            }
            assert!(history.is_empty());
        }
    }
    #[test]
    fn discounted_scores_include_current_and_terminal_positions() {
        for score_lambda in [0.0, 0.5, 1.0] {
            let mut history: Vec<History> = [[0, 3], [1, 2]].into_iter().map(|scores| History {
                board: [0;20], scores, mark_classes: [NO_MARK_CLASS;MARK_CLASS_SQUARES],
                actions: [0;2], policy: [[0;80];2], returns: [0.0;2], discounted_scores: (),
            }).collect();
            let mut buffer = Vec::new();
            finish(&mut history, [0.0;2], [2, 1], 0.94, score_lambda, &mut buffer);
            for p in 0..2 {
                let current = [0usize, 3][p];
                let next = [1usize, 2][p];
                let terminal = [2usize, 1][p];
                let d0 = &buffer[0].discounted_scores[p];
                let d1 = &buffer[1].discounted_scores[p];
                let expected_current = 1.0-score_lambda;
                let expected_next = score_lambda*(1.0-score_lambda);
                let expected_terminal = score_lambda*score_lambda;
                assert_eq!(d0[current],bf16(expected_current));
                assert_eq!(d0[next],bf16(expected_next));
                assert_eq!(d0[terminal],bf16(expected_terminal));
                assert_eq!(d1[next],bf16(expected_current));
                assert_eq!(d1[terminal],bf16(score_lambda));
                assert!((d0.iter().map(|&x| float(x)).sum::<f32>()-1.0).abs()<0.01);
            }
        }
    }
    #[test] fn packing_and_layout() {
        let cells=std::array::from_fn(|i| (i%4) as u8);
        assert_eq!(unpack(&pack(&cells)),cells);
        assert_eq!(std::mem::size_of::<Position>(),672 + MARK_CLASS_SQUARES);
        let mut b=vec![0;1600]; let mut s=vec![0;4];
        encode(&cells,[3,7],0,&mut b[..800],&mut s[..2]);
        encode(&cells,[3,7],1,&mut b[800..],&mut s[2..]);
        assert_eq!(s,[bf16(3.0),bf16(7.0),bf16(7.0),bf16(3.0)]);
        assert_eq!(&b[480..640],&b[1440..1600]);
    }
    #[test] fn shuffle_reorders_whole_positions() {
        let mut arena=Arena::new(1,80,80,0.03,0.1,0.8,0.94,1,false);
        arena.buffer=(0..40).map(|i| Position {
            board: [i;20], scores: [i,i], mark_classes: [i as i8;MARK_CLASS_SQUARES], actions: [i,i],
            policy: [[i as u16;80];2], returns: [bf16(i as f32);2],
            discounted_scores: [[bf16(i as f32);81];2],
        }).collect();
        arena.shuffle();
        let order: Vec<_> = arena.buffer.iter().map(|p| p.board[0]).collect();
        assert_ne!(order,(0..40).collect::<Vec<_>>());
        let mut sorted=order.clone(); sorted.sort_unstable();
        assert_eq!(sorted,(0..40).collect::<Vec<_>>());
        for p in &arena.buffer {
            let i=p.board[0];
            assert_eq!(p.scores,[i,i]);
            assert_eq!(p.mark_classes,[i as i8;MARK_CLASS_SQUARES]);
            assert_eq!(p.actions,[i,i]);
            assert_eq!(p.policy[0][0],i as u16);
            assert_eq!(p.returns,[bf16(i as f32);2]);
            assert_eq!(p.discounted_scores[0][0],bf16(i as f32));
        }
    }
    #[test] fn policy_mask_and_formula() {
        let g=Game::new(); let logits=std::array::from_fn::<_,80,_>(|i| i as f32/80.0);
        let (p,v)=policy(&g,0,&logits,&[0.25;80],0.03,0.1);
        assert!((p.iter().sum::<f32>()-1.0).abs()<1e-6);
        assert!((v-0.25).abs()<1e-6);
        for i in 0..80 { if i%8>=4 { assert_eq!(p[i],0.0); } }
        assert!((p[1]/p[0]-(0.1*(logits[1]-logits[0])/0.13).exp()).abs()<1e-5);
    }
    #[test] fn collection_returns_capacity_and_evaluation() {
        for lambda in [0.0,0.5,1.0] {
            let mut a=Arena::new(4,640,80,0.03,0.1,lambda,0.94,1,false);
            for _ in 0..1000 { if a.step(&[0.0;640],&[0.25;640]) { break; } }
            assert!(a.buffer.len()>320 && a.buffer.len()<=640);
            for pos in &a.buffer { for p in 0..2 {
                let cells=unpack(&pos.board); assert_eq!(cells[pos.actions[p] as usize],0);
                assert!(float(pos.returns[p]).abs()<=1.0);
                if lambda==1.0 { assert!([-1.0,0.0,1.0].contains(&float(pos.returns[p]))); }
            } }
            let dropped=a.reset(); assert!(dropped<320); assert!(a.histories.iter().all(Vec::is_empty));
        }
        let mut a=Arena::new(4,0,80,0.03,0.1,0.8,0.94,1,true);
        for _ in 0..80 { if a.step(&[0.0;640],&[0.0;640]) { break; } }
        assert_eq!(a.results.iter().sum::<usize>(),4);
        let mut expected = [0;3];
        for (i,g) in a.games.iter().enumerate() {
            let reward = g.terminal_values()[if i<2 {1} else {0}];
            expected[if reward>0.0 {0} else if reward==0.0 {1} else {2}] += 1;
        }
        assert_eq!(a.results,expected);
        let mut b=vec![1;6400]; let mut s=vec![1;16]; a.inputs(&mut b,&mut s);
        assert!(b.iter().chain(&s).all(|&x| x==0));
    }
}
