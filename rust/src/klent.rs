//! Single-threaded KLENT collection and Python tensor encoding.
use crate::{game::{board_index, Rng}, Game, Scratch};
use std::panic::{catch_unwind, AssertUnwindSafe};

const MAX_PLY: usize = crate::game::SQUARES;

#[derive(Clone)]
pub struct Position {
    board: [u8; 80],
    actions: [u8; 2],
    policy: [[f32; 80]; 2],
    returns: [f32; 2],
}

type History = Position;

fn finish(history: &mut Vec<History>, reward: [f32; 2], lambda: f32,
          buffer: &mut Vec<Position>) {
    let mut ret = reward;
    for t in (0..history.len()).rev() {
        if t + 1 < history.len() {
            for p in 0..2 {
                ret[p] = (1.0-lambda)*history[t+1].returns[p] + lambda*ret[p];
            }
        }
        let mut position = history[t].clone();
        position.returns = ret;
        buffer.push(position);
    }
    let start = buffer.len() - history.len();
    buffer[start..].reverse();
    history.clear();
}

/// FP32 one-hot planes: unplayable, empty, removed, own, opponent.
pub(crate) fn encode(cells: &[u8; 80], player: usize, board: &mut [f32]) {
    board.fill(0.0);
    for cell in 0..160 { if (cell/16 + cell%16)%2 != 0 { board[cell] = 1.0; } }
    for (sq, &mark) in cells.iter().enumerate() {
        let plane = match mark { 0 => 1, 3 => 2, 1 => 3+player, 2 => 4-player, _ => unreachable!() };
        board[plane*160 + board_index(sq)] = 1.0;
    }
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
fn mix_uniform(p: &[f32; 80], legal: [u64; 2], fraction: f32) -> [f32; 80] {
    if fraction == 0.0 { return *p; }
    let count = legal.iter().map(|bits| bits.count_ones()).sum::<u32>();
    assert!(count > 0);
    let floor = fraction / count as f32;
    std::array::from_fn(|a| {
        if legal[a/64] >> (a%64) & 1 != 0 { (1.0-fraction)*p[a] + floor } else { 0.0 }
    })
}

// Each quarter contains two equal side assignments. The current model owns
// the opposite side in every game; historical models only choose moves.
fn opponent_slot(game: usize, count: usize) -> (usize, usize) {
    if count < 8 { return (0, game % 2); }
    let quarter = game / (count / 4);
    let side = (game % (count / 4) >= count / 8) as usize;
    (quarter, side)
}

pub struct Arena {
    games: Vec<Game>, histories: Vec<Vec<History>>, buffer: Vec<Position>,
    capacity: usize, alpha: f32, beta: f32, lambda: f32, exploration_fraction: f32,
    rng: Rng, shuffle_rng: Rng, scratch: Scratch, evaluation: bool, pub results: [usize; 3],
}
impl Arena {
    pub fn new(n: usize, capacity: usize, alpha: f32, beta: f32, lambda: f32,
               exploration_fraction: f32, seed: u64, evaluation: bool) -> Self {
        assert!(n > 0 && (n < 8 || n % 8 == 0));
        assert!(evaluation || capacity >= n.checked_mul(MAX_PLY).unwrap());
        assert!(alpha.is_finite() && beta.is_finite() && alpha >= 0.0 && beta >= 0.0 && alpha+beta > 0.0);
        assert!(lambda.is_finite() && (0.0..=1.0).contains(&lambda));
        assert!(exploration_fraction.is_finite() && (0.0..=1.0).contains(&exploration_fraction));
        Self { games: vec![Game::new(); n], histories: (0..n).map(|_| Vec::with_capacity(if evaluation {0} else {MAX_PLY})).collect(),
            buffer: Vec::with_capacity(if evaluation {0} else {capacity}), capacity, alpha, beta, lambda, exploration_fraction,
            rng: Rng::new(seed), shuffle_rng: Rng::new(seed ^ 0x9e37_79b9_7f4a_7c15),
            scratch: Scratch::new(), evaluation, results: [0; 3] }
    }
    pub fn inputs(&self, boards: &mut [f32]) {
        boards.fill(0.0);
        for (i, g) in self.games.iter().enumerate() {
            if g.finished { continue; }
            for p in 0..2 {
                let row = 2*i+p;
                encode(&g.cells, p, &mut boards[row*800..(row+1)*800]);
            }
        }
    }
    pub fn step(&mut self, logits: &[f32], q: &[f32], opponent_logits: &[f32], opponent_q: &[f32]) -> bool {
        for i in 0..self.games.len() {
            if self.games[i].finished { continue; }
            let mut policies = [[0.0f32; 80]; 2];
            let mut values = [0.0; 2];
            let mut actions = [0u8; 2];
            let (quarter, opponent_side) = opponent_slot(i, self.games.len());
            for p in 0..2 {
                let row = (2*i+p)*80;
                let (dist, value) = if self.evaluation {
                    policy(&self.games[i], p, &logits[row..row+80], &[0.0; 80], 0.0, 1.0)
                } else { policy(&self.games[i], p, &logits[row..row+80], &q[row..row+80], self.alpha, self.beta) };
                policies[p] = dist; values[p] = value;
                let playing_dist = if !self.evaluation && p == opponent_side && quarter >= 2 {
                    policy(&self.games[i], p, &opponent_logits[row..row+80],
                           &opponent_q[row..row+80], self.alpha, self.beta).0
                } else { dist };
                let explore = !self.evaluation && quarter == 1;
                let sampling_dist = if explore {
                    mix_uniform(&playing_dist, self.games[i].legal_moves(p), self.exploration_fraction)
                } else { playing_dist };
                actions[p] = sample(&sampling_dist, &mut self.rng) as u8;
            }
            if !self.evaluation {
                assert!(self.histories[i].len() < MAX_PLY);
                self.histories[i].push(History { board: self.games[i].cells,
                    actions, policy: policies, returns: values });
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
                    finish(history, reward, self.lambda, &mut self.buffer);
                    self.games[i] = Game::new();
                }
            }
        }
        if self.evaluation { self.results.iter().sum::<usize>() == self.games.len() }
        else { self.capacity-self.buffer.len() < self.games.len()*MAX_PLY }
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
pub extern "C" fn klent_new(n: usize, m: usize, alpha: f32, beta: f32, lambda: f32,
                            exploration_fraction: f32, seed: u64, evaluation: bool) -> *mut Arena {
    catch_unwind(|| Box::into_raw(Box::new(Arena::new(n,m,alpha,beta,lambda,exploration_fraction,seed,evaluation)))).unwrap_or(std::ptr::null_mut())
}
#[no_mangle]
pub unsafe extern "C" fn klent_free(h: *mut Arena) { if !h.is_null() { drop(Box::from_raw(h)); } }
#[no_mangle]
pub unsafe extern "C" fn klent_inputs(h: *mut Arena, b: *mut f32) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let arena = &*h;
        let boards = std::slice::from_raw_parts_mut(b, arena.games.len() * 1600);
        arena.inputs(boards);
        0
    })).unwrap_or(-1)
}
#[no_mangle]
pub unsafe extern "C" fn klent_step(h: *mut Arena, logits: *const f32, q: *const f32,
                                     opponent_logits: *const f32, opponent_q: *const f32) -> i32 {
    catch_unwind(AssertUnwindSafe(|| { let a=&mut *h; let len=a.games.len()*160;
        a.step(std::slice::from_raw_parts(logits,len),std::slice::from_raw_parts(q,len),
               std::slice::from_raw_parts(opponent_logits,len),std::slice::from_raw_parts(opponent_q,len)) as i32
    })).unwrap_or(-1)
}
#[no_mangle]
/// Writes buffered positions, then evaluation wins, draws, and losses.
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
pub unsafe extern "C" fn klent_batch(h: *mut Arena, start: usize, count: usize,
    b: *mut f32, pi: *mut f32, actions: *mut i64, returns: *mut f32) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let a = &*h;
        let boards = std::slice::from_raw_parts_mut(b, count*1600);
        let policies = std::slice::from_raw_parts_mut(pi, count*320);
        let moves = std::slice::from_raw_parts_mut(actions, count*2);
        let targets = std::slice::from_raw_parts_mut(returns, count*2);
        boards.fill(0.0);
        policies.fill(0.0);
        moves.fill(0);
        targets.fill(0.0);
        let end = (start+count).min(a.buffer.len());
        for (i, pos) in a.buffer[start..end].iter().enumerate() {
            for p in 0..2 {
                let row = 2*i+p;
                encode(&pos.board, p, &mut boards[row*800..(row+1)*800]);
                for sq in 0..80 { policies[row*160+board_index(sq)] = pos.policy[p][sq]; }
                moves[row] = board_index(pos.actions[p] as usize) as i64;
                targets[row] = pos.returns[p];
            }
        }
        (end-start) as i32
    })).unwrap_or(-1)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_returns_use_next_state_and_preserve_order() {
        for lambda in [0.0, 0.5, 1.0] {
            let mut history: Vec<History> = (0..3).map(|i| History {
                board: [0;80], actions: [i;2], policy: [[0.0;80];2],
                returns: [i as f32 * 0.25, -(i as f32)*0.25],
            }).collect();
            let mut buffer = Vec::new();
            finish(&mut history, [1.0,-1.0], lambda, &mut buffer);
            let g1=(1.0-lambda)*0.5+lambda;
            let g0=(1.0-lambda)*0.25+lambda*g1;
            for (i,g) in [g0,g1,1.0].into_iter().enumerate() {
                assert_eq!(buffer[i].returns,[g,-g]);
                assert_eq!(buffer[i].actions,[i as u8;2]);
            }
            assert!(history.is_empty());
        }
    }
    #[test]
    fn board_layout() {
        let cells=std::array::from_fn(|i| (i%4) as u8);
        let mut b=vec![0.0;1600];
        encode(&cells,0,&mut b[..800]);
        encode(&cells,1,&mut b[800..]);
        assert_eq!(&b[480..640],&b[1440..1600]);
    }
    #[test]
    fn shuffle_reorders_whole_positions() {
        let mut arena=Arena::new(1,80,0.03,0.1,0.8,0.0,1,false);
        arena.buffer=(0..40).map(|i| Position {
            board: [i;80], actions: [i,i], policy: [[i as f32;80];2], returns: [i as f32;2],
        }).collect();
        arena.shuffle();
        let order: Vec<_> = arena.buffer.iter().map(|p| p.board[0]).collect();
        assert_ne!(order,(0..40).collect::<Vec<_>>());
        let mut sorted=order.clone(); sorted.sort_unstable();
        assert_eq!(sorted,(0..40).collect::<Vec<_>>());
        for p in &arena.buffer {
            let i=p.board[0];
            assert_eq!(p.actions,[i,i]);
            assert_eq!(p.policy[0][0],i as f32);
            assert_eq!(p.returns,[i as f32;2]);
        }
    }
    #[test]
    fn policy_mask_and_formula() {
        let g=Game::new(); let logits=std::array::from_fn::<_,80,_>(|i| i as f32/80.0);
        let (p,v)=policy(&g,0,&logits,&[0.25;80],0.03,0.1);
        assert!((p.iter().sum::<f32>()-1.0).abs()<1e-6);
        assert!((v-0.25).abs()<1e-6);
        for i in 0..80 { if i%8>=4 { assert_eq!(p[i],0.0); } }
        assert!((p[1]/p[0]-(0.1*(logits[1]-logits[0])/0.13).exp()).abs()<1e-5);
    }
    #[test]
    fn self_play_exploration_mixes_only_legal_moves() {
        let g=Game::new();
        let logits=std::array::from_fn::<_,80,_>(|i| i as f32);
        let (p,_) = policy(&g,0,&logits,&[0.0;80],0.03,0.1);
        let mixed = mix_uniform(&p,g.legal_moves(0),0.1);
        assert!((mixed.iter().sum::<f32>()-1.0).abs()<1e-6);
        for i in 0..80 {
            if i%8 < 4 { assert!((mixed[i]-(0.9*p[i]+0.1/40.0)).abs()<1e-7); }
            else { assert_eq!(mixed[i],0.0); }
        }
        assert_eq!(mix_uniform(&p,g.legal_moves(0),0.0),p);
    }
    #[test]
    fn collection_returns_capacity_and_evaluation() {
        for lambda in [0.0,0.5,1.0] {
            let mut a=Arena::new(4,640,0.03,0.1,lambda,0.1,1,false);
            for _ in 0..1000 { if a.step(&[0.0;640],&[0.25;640],&[0.0;640],&[0.25;640]) { break; } }
            assert!(a.buffer.len()>320 && a.buffer.len()<=640);
            for pos in &a.buffer { for p in 0..2 {
                assert_eq!(pos.board[pos.actions[p] as usize],0);
                assert!(pos.returns[p].abs()<=1.0);
                if lambda==1.0 { assert!([-1.0,0.0,1.0].contains(&pos.returns[p])); }
            } }
            let dropped=a.reset(); assert!(dropped<320); assert!(a.histories.iter().all(Vec::is_empty));
        }
        let mut a=Arena::new(4,0,0.03,0.1,0.8,0.1,1,true);
        for _ in 0..80 { if a.step(&[0.0;640],&[0.0;640],&[0.0;640],&[0.0;640]) { break; } }
        assert_eq!(a.results.iter().sum::<usize>(),4);
        let mut expected = [0;3];
        for (i,g) in a.games.iter().enumerate() {
            let reward = g.terminal_values()[if i<2 {1} else {0}];
            expected[if reward>0.0 {0} else if reward==0.0 {1} else {2}] += 1;
        }
        assert_eq!(a.results,expected);
        let mut b=vec![1.0;6400]; a.inputs(&mut b);
        assert!(b.iter().all(|&x| x==0.0));
    }

    #[test]
    fn historical_opponent_changes_moves_but_not_targets() {
        let mut a=Arena::new(8,640,0.0,1.0,0.8,0.0,1,false);
        let current=[0.0;1280];
        let mut old=[0.0;1280];
        let old_q=[2.0;1280];
        for row in 0..16 { old[row*80+1]=100.0; old[row*80+5]=100.0; }
        a.step(&current,&current,&old,&old_q);
        assert_eq!(a.histories[4][0].actions[0],1);
        assert_eq!(a.histories[5][0].actions[1],5);
        assert!((a.histories[4][0].policy[0][1]-1.0/40.0).abs()<1e-6);
        assert!((a.histories[5][0].policy[1][5]-1.0/40.0).abs()<1e-6);
        assert_eq!(a.histories[4][0].returns[0],0.0);
        assert_eq!(a.histories[5][0].returns[1],0.0);
        assert_eq!(opponent_slot(0,1024),(0,0));
        assert_eq!(opponent_slot(127,1024),(0,0));
        assert_eq!(opponent_slot(128,1024),(0,1));
        assert_eq!(opponent_slot(512,1024),(2,0));
        assert_eq!(opponent_slot(1023,1024),(3,1));
        for quarter in 0..4 {
            for side in 0..2 {
                assert_eq!((0..1024).filter(|&i| opponent_slot(i,1024)==(quarter,side)).count(),128);
            }
        }
    }
}
