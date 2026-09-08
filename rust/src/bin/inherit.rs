//! What a promoted root inherits: the visit/strategy totals its slot built up while
//! searching from the parent, measured against the simulations the new step will run.
//!
//! Runs short searches (a few steps), then reports, over all slots that promoted into an
//! existing node at the last step: the distribution of inherited `total` (PUCT) or summed
//! `strategy_sum` (EXP3), the fresh step's own budget S, and how many of the node's children
//! survived the sweep that fired the step. Children are counted from the promoted key; a
//! survivor is a joint child still present in the arena after the post-step sweep.
//!
//! The stub assigns a one-hot prior — 100% of the mass on one legal square per player —
//! with random values. `--powerlaw` restores the older peaked stub.
//!
//! Usage: inherit [--exp3] [--powerlaw] [--cycles N] [--steps N]

use alpha_lines_game::game::{Rng, SQUARES, PLAYABLE_SQUARE, ROW, WIDTH};
use alpha_lines_game::mcts::search::{Evaluate, Search};
use alpha_lines_game::mcts::variant::{squares, Exp3, Puct};
use alpha_lines_game::mcts::Config;
use alpha_lines_game::zobrist;

struct Stub {
    rng: Rng,
    onehot: bool,
}

fn in_half(k: usize, player: usize) -> bool {
    let (r, j) = (k / ROW, k % ROW);
    let c = 2 * j + (r & 1);
    if player == 0 { c < WIDTH / 2 } else { c >= WIDTH / 2 }
}

fn spike_square(cells: &[u8; SQUARES], player: usize) -> usize {
    let opening = cells.iter().all(|&c| c == PLAYABLE_SQUARE);
    let pref = if player == 0 { 0 } else { 4 };
    if cells[pref] == PLAYABLE_SQUARE && (!opening || in_half(pref, player)) {
        return pref;
    }
    for k in 0..SQUARES {
        if cells[k] == PLAYABLE_SQUARE && (!opening || in_half(k, player)) {
            return k;
        }
    }
    (0..SQUARES).find(|&k| cells[k] == PLAYABLE_SQUARE).unwrap_or(0)
}

impl Evaluate for Stub {
    fn evaluate(&mut self, positions: &[[u8; SQUARES]], priors: &mut [f32], values: &mut [f32]) {
        if self.onehot {
            let width = priors.len() / (2 * SQUARES);
            let nrows = priors.len() / SQUARES;
            for idx in 0..nrows {
                let player = if idx >= width { 1 } else { 0 };
                let spike = spike_square(&positions[idx % width], player);
                let row = &mut priors[idx * SQUARES..(idx + 1) * SQUARES];
                for v in row.iter_mut() {
                    *v = 0.0;
                }
                row[spike] = 1.0;
            }
        } else {
            for row in priors.chunks_mut(SQUARES) {
                let o = self.rng.randint(SQUARES as u64) as usize;
                for j in 0..SQUARES {
                    row[(o + j) % SQUARES] = 1.0 / (1.0 + j as f32).powf(6.0);
                }
            }
        }
        for v in values.iter_mut() {
            *v = self.rng.random() as f32 * 2.0 - 1.0;
        }
    }
}

fn arg(args: &[String], key: &str, default: usize) -> usize {
    args.iter().position(|a| a == key).map(|i| args[i + 1].parse().expect("numeric")).unwrap_or(default)
}

fn quantiles(mut v: Vec<f64>) -> (f64, f64, f64, f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| v[((v.len() as f64 * p).floor() as usize).min(v.len() - 1)];
    (v[0], q(0.25), q(0.5), q(0.75), v[v.len() - 1])
}

fn report_totals(label: &str, totals: &[f64], kids: &[usize], matched: usize, s_budget: u32) {
    println!("{label}: {matched} promoted roots (matched an existing node)");
    if totals.is_empty() {
        println!("  no promotions matched; nothing inherited");
        return;
    }
    let (mn, q1, med, q3, mx) = quantiles(totals.to_vec());
    println!("  inherited total over {matched}: min {mn:.0} q1 {q1:.0} med {med:.0} q3 {q3:.0} max {mx:.0}");
    println!("  fresh step budget S = {s_budget}; max/S = {:.1}x", mx / s_budget as f64);
    let (mut zero, mut one, mut many) = (0, 0, 0);
    let mut kid_sum = 0usize;
    for &k in kids {
        kid_sum += k;
        match k {
            0 => zero += 1,
            1 => one += 1,
            _ => many += 1,
        }
    }
    println!("  joint children surviving the sweep: mean {:.1}, with 0 kids {zero}, 1 kid {one}, 2+ kids {many}",
        kid_sum as f64 / kids.len().max(1) as f64);
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let exp3 = a.iter().any(|x| x == "--exp3");
    let onehot = !a.iter().any(|x| x == "--powerlaw");
    let cycles = arg(&a, "--cycles", 400);
    let want_steps = arg(&a, "--steps", 2);

    let cfg = Config::default();
    let mut model = Stub { rng: Rng::new(7), onehot };
    println!("inherit: {} steps max ({} cycles cap), one-hot={}, S={} T={} G={}",
        want_steps, cycles, onehot, cfg.s, cfg.t, cfg.g);

    macro_rules! go {
        ($V:ty, $total:expr) => {{
            let mut s = Search::<$V>::new(cfg, 0xA1FA);
            let mut steps = 0usize;
            for _ in 0..cycles {
                if !s.cycle(&mut model).is_empty() {
                    steps += 1;
                    if steps >= want_steps {
                        break;
                    }
                }
            }
            println!("  ran {} steps ({} cycles)", steps, s.diag.cycles);
            let mut totals = Vec::new();
            let mut kids = Vec::new();
            let mut matched = 0usize;
            for i in 0..s.slots.len() {
                let r = &s.slots[i].root;
                if r.game.move_count as usize != steps || r.pending {
                    continue;
                }
                if s.arena.get(r.key).is_none() {
                    continue; // fresh root, inherited nothing from the arena
                }
                matched += 1;
                totals.push($total(r));
                let l0: Vec<usize> = squares(r.game.legal_moves(0)).collect();
                let l1: Vec<usize> = squares(r.game.legal_moves(1)).collect();
                let mut k = 0;
                for &x in &l0 {
                    for &y in &l1 {
                        if s.arena.get(zobrist::step(r.key, &r.game.cells, x, y)).is_some() {
                            k += 1;
                        }
                    }
                }
                kids.push(k);
            }
            report_totals(if exp3 { "EXP3" } else { "PUCT" }, &totals, &kids, matched, cfg.s);
        }};
    }

    if exp3 {
        go!(Exp3, |r: &alpha_lines_game::mcts::slot::Root<alpha_lines_game::mcts::variant::Exp3Stats>| {
            let mut t = 0.0f64;
            for p in 0..2 {
                t += r.stats.strategy_sum[p].iter().map(|&x| x as f64).sum::<f64>();
            }
            t
        });
    } else {
        go!(Puct, |r: &alpha_lines_game::mcts::slot::Root<alpha_lines_game::mcts::variant::PuctStats>| {
            (r.stats.total[0] + r.stats.total[1]) as f64
        });
    }
}
