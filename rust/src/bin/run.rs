//! Drives a real search against a stand-in model and reports what it did.
//!
//! This is the rig the timings come from: a full loop with steps, sweeps and reseeding, not
//! a micro-benchmark of one call. The model is a stub, so the evaluate column is not a real
//! GPU cost — everything else is.
//!
//! The configuration is the spec's and is not adjustable: 8192 game slots, a 2048-wide
//! buffer filled by walking the slots from 0, a step once 2048 games are ready, 100
//! simulations per game per move.
//!
//! Usage: run [--cycles N] [--exp3]

use std::time::Instant;

use alpha_lines_game::game::{Rng, SQUARES};
use alpha_lines_game::mcts::search::{Evaluate, Search};
use alpha_lines_game::mcts::variant::{Exp3, Exp3Stats, Puct, PuctStats, Variant};
use alpha_lines_game::mcts::{Config, Node};

/// Stands in for an untrained network: near-uniform priors, noisy values.
struct Stub(Rng);
impl Evaluate for Stub {
    fn evaluate(&mut self, _pos: &[[u8; SQUARES]], priors: &mut [f32], values: &mut [f32]) {
        for p in priors.iter_mut() {
            *p = 1.0 + self.0.random() as f32 * 0.05;
        }
        for v in values.iter_mut() {
            *v = self.0.random() as f32 * 2.0 - 1.0;
        }
    }
}

fn arg(args: &[String], key: &str, default: usize) -> usize {
    args.iter().position(|a| a == key).map(|i| args[i + 1].parse().expect("numeric")).unwrap_or(default)
}

fn go<V: Variant>(cfg: Config, cycles: usize, node_bytes: usize) {
    let mut search = Search::<V>::new(cfg.clone(), 0xA1FA);
    let mut model = Stub(Rng::new(7));
    let (mut moves, mut done) = (0usize, 0usize);

    let t = Instant::now();
    for _ in 0..cycles {
        let recs = search.cycle(&mut model);
        moves += recs.len();
        done += recs.iter().filter(|r| r.finished).count();
    }
    let secs = t.elapsed().as_secs_f64();
    let d = &search.diag;

    let ns = |x: u128| x as f64 / 1e6;
    println!("\n{cycles} cycles in {secs:.1}s   {:.0} cycles/s", cycles as f64 / secs);
    println!("  collect  {:>8.0} ms  ({:.0}% )   {:.0} ns/descent",
        ns(d.t_collect), 100.0 * d.t_collect as f64 / (secs * 1e9), d.t_collect as f64 / d.descents.max(1) as f64);
    println!("  evaluate {:>8.0} ms  (stub, not a real model)", ns(d.t_evaluate));
    println!("  scatter  {:>8.0} ms   {:.0} ns/row", ns(d.t_scatter), d.t_scatter as f64 / (d.buffer_unique + d.root_evals).max(1) as f64);
    println!("  backup   {:>8.0} ms   {:.0} ns/path", ns(d.t_backup), d.t_backup as f64 / (d.buffer_unique + d.duplicate_hits).max(1) as f64);
    println!("  step     {:>8.0} ms over {} steps  ({:.0} ms each)", ns(d.t_step), d.steps, ns(d.t_step) / d.steps.max(1) as f64);

    println!("\nsearch");
    println!("  descents      {}  ({:.0}/s), mean depth {:.2}", d.descents, d.descents as f64 / secs, d.mean_depth());
    println!("  new nodes     {}  ({:.0}% of descents)", d.buffer_unique, 100.0 * d.buffer_unique as f64 / d.descents.max(1) as f64);
    println!("  shared        {}  ({:.0}%)", d.duplicate_hits, 100.0 * d.duplicate_hits as f64 / d.descents.max(1) as f64);
    println!("  terminal      {}  ({:.0}%)", d.terminal_hits, 100.0 * d.terminal_hits as f64 / d.descents.max(1) as f64);
    println!("  root evals    {}   exhausted {}   max-descents {}", d.root_evals, d.exhausted, d.max_descents_hit);
    println!("  buffer short  {:.1}% of rows   deepest slot {}", 100.0 * d.buffer_shortfall as f64 / (d.cycles * cfg.b as u64) as f64, d.deepest_slot);

    println!("\ngames");
    println!("  steps {}   moves {moves}   games finished {done}", d.steps);

    println!("\narena");
    println!("  live {}  ({:.0} MB of {:.0} MB budget)   free list {}",
        search.arena.len(), search.arena.len() as f64 * node_bytes as f64 / 1e6,
        cfg.node_capacity as f64 * node_bytes as f64 / 1e6, search.arena.free_depth());
    println!("  created {}   deleted {}   id shifts {}", d.buffer_unique, d.nodes_deleted, search.arena.shifted);
    let (mean, worst) = search.arena.probe_lengths();
    println!("  probe {mean:.2} mean, {worst} worst   id overflows {}", search.arena.overflows);
    println!("  id histogram {:?}  max {}", d.id_histogram, d.max_id_count);
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let cycles = arg(&a, "--cycles", 16000);
    let cfg = Config::default();
    println!(
        "G={} B={} T={} S={} capacity={}",
        cfg.g, cfg.b, cfg.t, cfg.s, cfg.node_capacity
    );
    if a.iter().any(|x| x == "--exp3") {
        go::<Exp3>(cfg, cycles, std::mem::size_of::<Node<Exp3Stats>>());
    } else {
        go::<Puct>(cfg, cycles, std::mem::size_of::<Node<PuctStats>>());
    }
}
