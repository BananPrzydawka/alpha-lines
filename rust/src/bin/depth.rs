//! Where the depth goes, over exactly one move.
//!
//! Runs `S - 1` cycles so no step ever fires: every statistic is from the first move's 100
//! simulations, before any promotion or sweep can muddy it. Sweeps the exploration constant
//! — Dirichlet epsilon for PUCT, the mixture floor gamma for EXP3 — because that is the only
//! knob that spreads a game across sibling children.
//!
//! Usage: depth [--exp3]

use alpha_lines_game::game::{Rng, SQUARES};
use alpha_lines_game::mcts::search::{Evaluate, Search};
use alpha_lines_game::mcts::variant::{Exp3, Puct, Variant};
use alpha_lines_game::mcts::Config;
use alpha_lines_game::zobrist;

struct Stub(Rng);
impl Evaluate for Stub {
    fn evaluate(&mut self, _p: &[[u8; SQUARES]], priors: &mut [f32], values: &mut [f32]) {
        for row in priors.chunks_mut(SQUARES) {
            let o = self.0.randint(SQUARES as u64) as usize;
            for j in 0..SQUARES {
                row[(o + j) % SQUARES] = 1.0 / (1.0 + j as f32).powf(6.0);
            }
        }
        for v in values.iter_mut() {
            *v = self.0.random() as f32 * 2.0 - 1.0;
        }
    }
}

/// Every bucket that holds anything, as a percentage.
fn spread(h: &[u64], total: u64) -> String {
    let last = h.iter().rposition(|&c| c > 0).unwrap_or(0);
    (0..=last)
        .map(|k| format!("{:.1}", 100.0 * h[k] as f64 / total.max(1) as f64))
        .collect::<Vec<_>>()
        .join(" ")
}

/// How wide the search actually got at a game's own root, averaged over a sample of games.
fn root_width<V: Variant>(s: &Search<V>, visited: impl Fn(&Search<V>, usize, usize, usize) -> bool) -> (f64, f64, f64) {
    let sample = 64.min(s.slots.len());
    let (mut a0, mut a1, mut kids) = (0.0, 0.0, 0.0);
    for i in 0..sample {
        let g = &s.slots[i].root.game;
        let key = s.slots[i].root.key;
        let p0: Vec<usize> = (0..SQUARES).filter(|&sq| visited(s, i, 0, sq)).collect();
        let p1: Vec<usize> = (0..SQUARES).filter(|&sq| visited(s, i, 1, sq)).collect();
        let mut k = 0;
        for &x in &p0 {
            for &y in &p1 {
                if s.arena.get(zobrist::step(key, &g.cells, x, y)).is_some() {
                    k += 1;
                }
            }
        }
        a0 += p0.len() as f64;
        a1 += p1.len() as f64;
        kids += k as f64;
    }
    (a0 / sample as f64, a1 / sample as f64, kids / sample as f64)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let exp3 = a.iter().any(|x| x == "--exp3");
    let base = Config::default();
    let knobs: &[f32] = if exp3 { &[0.0, 0.01, 0.05, 0.1, 0.25] } else { &[0.0, 0.05, 0.25, 0.5, 1.0] };
    let name = if exp3 { "gamma" } else { "epsilon" };

    println!("one move only: {} cycles, no step fires. {} slots, B={}.",
        base.s - 1, base.g, base.b);
    println!("depth spread is % of descents at depth 0,1,2,... with no truncation.\n");

    for &k in knobs {
        let mut cfg = base;
        if exp3 { cfg.exp3_gamma = k } else { cfg.epsilon = k }
        let mut model = Stub(Rng::new(7));

        macro_rules! go {
            ($V:ty, $vis:expr) => {{
                let mut s = Search::<$V>::new(cfg, 0xA1FA);
                for _ in 0..(cfg.s - 1) { s.cycle(&mut model); }
                let d = &s.diag;
                let (w0, w1, kids) = root_width(&s, $vis);
                println!("{name} = {k}");
                println!("  descents {}  mean depth {:.2}  nodes {}", d.descents, d.mean_depth(), s.arena.len());
                println!("  root width: p0 {w0:.1} moves tried, p1 {w1:.1}, {kids:.1} joint children exist");
                println!("  all      {}", spread(&d.depth_hist, d.descents));
                println!("  new node {}", spread(&d.depth_new, d.descents));
                println!("  pending  {}", spread(&d.depth_dup, d.descents));
                println!("  terminal {}\n", spread(&d.depth_terminal, d.descents));
            }};
        }
        if exp3 {
            go!(Exp3, |s: &Search<Exp3>, i: usize, p: usize, sq: usize| s.slots[i].root.stats.strategy_sum[p][sq] > 0.0);
        } else {
            go!(Puct, |s: &Search<Puct>, i: usize, p: usize, sq: usize| s.slots[i].root.stats.visit[p][sq] > 0);
        }
    }
}
