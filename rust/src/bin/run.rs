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
//! Usage: run [--cycles N] [--exp3] [--peaked] [--zero-values]

use std::time::Instant;

use alpha_lines_game::game::{Rng, SQUARES};
use alpha_lines_game::mcts::search::{Evaluate, Search};
use alpha_lines_game::mcts::variant::{Exp3, Exp3Stats, Puct, PuctStats, Variant};
use alpha_lines_game::mcts::{Config, Node};

/// Stands in for a network. Nothing is trained and nothing is updated.
///
/// `Flat` is an untrained net: near-uniform priors, so the search has no opinion and spreads
/// over the whole legal set. `Peaked` stands in for a trained one by raising uniform draws to
/// a power, which concentrates the mass on a few squares the way a confident policy would —
/// the point being to read what the search does when its effective branching factor is small.
struct Stub {
    rng: Rng,
    peaked: bool,
    /// Return zero for every value, so selection is driven by the policy alone. Random values
    /// give every edge a noisy `q` in [-1, 1], which is the same size as the exploration term
    /// and so competes with the prior for control of the argmax.
    zero_values: bool,
}
impl Evaluate for Stub {
    fn evaluate(&mut self, _pos: &[[u8; SQUARES]], priors: &mut [f32], values: &mut [f32]) {
        if self.peaked {
            // A power-law decay away from a random rank-0 square. Not a single spike: one
            // spike lands on an illegal square often enough to leave a player flat, and an
            // exponential decay underflows f32 within a few ranks with the same result. A
            // power law keeps its shape whatever the legal mask removes.
            for row in priors.chunks_mut(SQUARES) {
                let o = self.rng.randint(SQUARES as u64) as usize;
                for j in 0..SQUARES {
                    row[(o + j) % SQUARES] = 1.0 / (1.0 + j as f32).powf(6.0);
                }
            }
        } else {
            for p in priors.iter_mut() {
                *p = 1.0 + self.rng.random() as f32 * 0.05;
            }
        }
        for v in values.iter_mut() {
            *v = if self.zero_values { 0.0 } else { self.rng.random() as f32 * 2.0 - 1.0 };
        }
    }
}

fn arg(args: &[String], key: &str, default: usize) -> usize {
    args.iter().position(|a| a == key).map(|i| args[i + 1].parse().expect("numeric")).unwrap_or(default)
}

fn go<V: Variant>(cfg: Config, cycles: usize, node_bytes: usize, peaked: bool, zero_values: bool) {
    let mut search = Search::<V>::new(cfg.clone(), 0xA1FA);
    let mut model = Stub { rng: Rng::new(7), peaked, zero_values };
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
    let dh: Vec<(usize, String)> = d.depth_hist.iter().enumerate().filter(|(_, &c)| c > 0)
        .map(|(k, &c)| (k, format!("{:.1}%", 100.0 * c as f64 / d.descents.max(1) as f64))).collect();
    println!("  depth spread  {dh:?}");
    println!("  new nodes     {}  ({:.0}% of descents)", d.buffer_unique, 100.0 * d.buffer_unique as f64 / d.descents.max(1) as f64);
    println!("  shared        {}  ({:.0}%)", d.duplicate_hits, 100.0 * d.duplicate_hits as f64 / d.descents.max(1) as f64);
    println!("  terminal      {}  ({:.0}%)", d.terminal_hits, 100.0 * d.terminal_hits as f64 / d.descents.max(1) as f64);
    println!("  root evals    {}   exhausted {}   max-descents {}", d.root_evals, d.exhausted, d.max_descents_hit);

    println!("\nmodel calls (one per cycle)");
    println!("  calls         {}   {:.2} ms of non-model work each", d.cycles,
        (secs * 1e3 - d.t_evaluate as f64 / 1e6) / d.cycles.max(1) as f64);
    println!("  short calls   {} of {}  ({} rows unfilled in total)", d.cycles_short, d.cycles, d.buffer_shortfall);
    println!("  short at      first {:?}{}", d.short_cycles, if d.cycles_short > 64 { " ..." } else { "" });
    println!("                last short cycle {} of {}", d.last_short_cycle, d.cycles);
    println!("  on short cycles: {} descents, {} of them ({:.0}%) landed on a position another",
        d.short_descents, d.short_duplicates,
        100.0 * d.short_duplicates as f64 / d.short_descents.max(1) as f64);
    println!("                   game had already queued, so they filled no row");
    println!("  walk reaches  slot {:.0} on average, {} at worst, of {}",
        d.walk_end_total as f64 / d.cycles.max(1) as f64, d.deepest_slot, cfg.g - 1);

    println!("\ngames");
    println!("  steps {}   moves {moves}   games finished {done}", d.steps);

    println!("\narena");
    println!("  live {}  ({:.0} MB of {:.0} MB budget)   free list {}",
        search.arena.len(), search.arena.len() as f64 * node_bytes as f64 / 1e6,
        cfg.node_capacity as f64 * node_bytes as f64 / 1e6, search.arena.free_depth());
    println!("  created {}   deleted {}   id shifts {}", d.buffer_unique, d.nodes_deleted, search.arena.shifted);
    let (mean, worst) = search.arena.probe_lengths();
    println!("  probe {mean:.2} mean, {worst} worst   id overflows {}", search.arena.overflows);
    println!("  id histogram (cumulative over sweeps) {:?}", d.id_histogram);
    println!("  max ids seen {}   nodes that overflowed {} ({:.3}% of created)",
        d.max_id_count, search.arena.overflow_nodes,
        100.0 * search.arena.overflow_nodes as f64 / d.buffer_unique.max(1) as f64);
    let ply: Vec<(usize, u64)> = search.arena.overflow_ply.iter().enumerate()
        .filter(|(_, &c)| c > 0).map(|(p, &c)| (p, c)).collect();
    println!("  overflows by ply {ply:?}");
    println!("  reach tests {} of which kept {} ({:.1}% hit rate)",
        d.reach_tested, d.reach_kept, 100.0 * d.reach_kept as f64 / d.reach_tested.max(1) as f64);
    let fns: Vec<(usize, u64)> = d.full_node_survivors.iter().enumerate()
        .filter(|(_, &c)| c > 0).map(|(k, &c)| (k, c)).collect();
    println!("  full (K-id) nodes at sweep, by ids surviving: {fns:?}");
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let cycles = arg(&a, "--cycles", 16000);
    let cfg = Config::default();
    println!(
        "G={} B={} T={} S={} capacity={}",
        cfg.g, cfg.b, cfg.t, cfg.s, cfg.node_capacity
    );
    let peaked = a.iter().any(|x| x == "--peaked");
    let zero = a.iter().any(|x| x == "--zero-values");
    println!("model: {} policy, {} values",
        if peaked { "peaked" } else { "flat" },
        if zero { "zero" } else { "random" });
    if a.iter().any(|x| x == "--exp3") {
        go::<Exp3>(cfg, cycles, std::mem::size_of::<Node<Exp3Stats>>(), peaked, zero);
    } else {
        go::<Puct>(cfg, cycles, std::mem::size_of::<Node<PuctStats>>(), peaked, zero);
    }
}
