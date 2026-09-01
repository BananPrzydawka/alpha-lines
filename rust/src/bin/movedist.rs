//! Distribution of per-move cost in the incremental engine.
//!
//! The design predicts five paths whose costs differ by orders of magnitude — a dead insert
//! is a handful of lookups, a slow removal can be hundreds of repair pops. This times every
//! individual game-move and buckets the results, both overall and split by which path the
//! move actually took, to see whether those paths are distinguishable in the distribution
//! or smeared together.
//!
//! Build with `--features stats` to get the per-path split; without it only the overall
//! histogram is produced. The counters cost ~5% of a move, which inflates every bucket
//! uniformly and so does not change the shape.
//!
//! Usage: movedist [--games N] [--seed S]

use std::time::Instant;

use alpha_lines_game::config::{HEIGHT, WIDTH};
use alpha_lines_game::game_kernels::{apply_and_score_kernel, sample_move_kernel};
use alpha_lines_game::incremental::HW;
use alpha_lines_game::rng::Rng;
use alpha_lines_game::{BatchedLinesGame, IncrementalGame};

fn arg_usize(args: &[String], key: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == key)
        .map(|i| args[i + 1].parse().expect("numeric"))
        .unwrap_or(default)
}

/// Which path a single game-move took, derived from the stats deltas it produced.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Path {
    InsertDead,
    InsertFast,
    InsertSlow,
    CollisionCheap,
    CollisionSlow,
    Unknown,
}

impl Path {
    fn label(self) -> &'static str {
        match self {
            Path::InsertDead => "move, both marks dead (no route to border)",
            Path::InsertFast => "move, fast path (live territory)",
            Path::InsertSlow => "move, slow path (revives a dead blob)",
            Path::CollisionCheap => "collision, no slow removal",
            Path::CollisionSlow => "collision, >=1 slow removal",
            Path::Unknown => "unclassified (build with --features stats)",
        }
    }
    fn all() -> [Path; 6] {
        [Path::InsertDead, Path::InsertFast, Path::InsertSlow,
         Path::CollisionCheap, Path::CollisionSlow, Path::Unknown]
    }
}

struct Sample {
    ns: f64,
    path: Path,
    repair_pops: u64,
    /// highest level the repair wrote, and how the dying cells actually died
    repair_max_level: u8,
    capped: u64,
    dead: u64,
    /// how many removals on this move took the slow (component) path
    slow_removals: u64,
}

fn pct(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let i = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[i]
}

/// ASCII histogram over log-spaced buckets, which is the only way a 100x spread reads.
fn histogram(title: &str, samples: &[f64], total_all: usize) {
    if samples.is_empty() {
        return;
    }
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let edges: Vec<f64> = vec![
        0.0, 100.0, 150.0, 200.0, 300.0, 400.0, 600.0, 800.0, 1_200.0, 1_600.0,
        2_400.0, 3_200.0, 4_800.0, 6_400.0, 12_800.0, 25_600.0, f64::INFINITY,
    ];
    let mut counts = vec![0usize; edges.len() - 1];
    for &v in &s {
        let b = edges.iter().position(|&e| v < e).unwrap_or(edges.len() - 1) - 1;
        counts[b] += 1;
    }
    let peak = *counts.iter().max().unwrap_or(&1).max(&1);

    println!(
        "\n{}  ({} moves, {:.1}% of all)",
        title,
        s.len(),
        s.len() as f64 / total_all as f64 * 100.0
    );
    println!(
        "   min {:.0}  p50 {:.0}  p90 {:.0}  p99 {:.0}  p99.9 {:.0}  max {:.0} ns   mean {:.0} ns",
        s[0], pct(&s, 0.50), pct(&s, 0.90), pct(&s, 0.99), pct(&s, 0.999),
        s[s.len() - 1], s.iter().sum::<f64>() / s.len() as f64
    );
    for b in 0..counts.len() {
        if counts[b] == 0 {
            continue;
        }
        let hi = edges[b + 1];
        let label = if hi.is_infinite() {
            format!("{:>6}+   ", edges[b] as u64)
        } else {
            format!("{:>6}-{:<6}", edges[b] as u64, hi as u64)
        };
        let bar = (counts[b] * 56 / peak).max(1);
        println!(
            "  {} {:>8} {:5.1}% {}",
            label,
            counts[b],
            counts[b] as f64 / s.len() as f64 * 100.0,
            "#".repeat(bar)
        );
    }
}

/// Replay a recorded move sequence through the incremental engine, timing every individual
/// game-move. `bfs` picks the slow-removal repair strategy. The sequence is fixed, so two
/// runs with different strategies produce sample vectors that line up index for index.
fn replay(
    rec: &[(Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<bool>)],
    n: usize,
    seed: u64,
    overhead: f64,
    bfs: bool,
) -> (Vec<Sample>, IncrementalGame) {
    let mut inc = IncrementalGame::new(n, seed);
    inc.set_bfs_repair(bfs);
    let mut samples: Vec<Sample> = Vec::with_capacity(n * rec.len());
    for (r0, c0, r1, c1, active) in rec {
        for g in 0..n {
            if !active[g] {
                continue;
            }
            let before = inc.stats();
            let t = Instant::now();
            inc.apply_game(g, r0[g], c0[g], r1[g], c1[g]);
            let ns = t.elapsed().as_secs_f64() * 1e9 - overhead;
            let a = inc.stats();

            let d = |now: u64, was: u64| now - was;
            let collision = (r0[g], c0[g]) == (r1[g], c1[g]);
            let path = if a.inserts + a.removes == before.inserts + before.removes {
                Path::Unknown
            } else if collision {
                if d(a.removes_slow, before.removes_slow) > 0 {
                    Path::CollisionSlow
                } else {
                    Path::CollisionCheap
                }
            } else if d(a.inserts_slow, before.inserts_slow) > 0 {
                Path::InsertSlow
            } else if d(a.inserts_dead, before.inserts_dead) == 2 {
                Path::InsertDead
            } else {
                Path::InsertFast
            };
            samples.push(Sample {
                ns: ns.max(0.0),
                path,
                repair_pops: d(a.repair_pops, before.repair_pops),
                repair_max_level: a.repair_max_level,
                capped: d(a.repair_capped, before.repair_capped),
                dead: d(a.repair_dead, before.repair_dead),
                slow_removals: d(a.removes_slow, before.removes_slow),
            });
        }
    }
    (samples, inc)
}

/// median and mean of a slice, in ns.
fn med_mean(v: &[f64]) -> (f64, f64) {
    if v.is_empty() {
        return (0.0, 0.0);
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (pct(&s, 0.5), v.iter().sum::<f64>() / v.len() as f64)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n = arg_usize(&args, "--games", 2048);
    let seed = arg_usize(&args, "--seed", 12345) as u64;

    // timer overhead, so the smallest bucket can be read honestly
    let probe = 200_000;
    let t = Instant::now();
    let mut acc = 0u64;
    for _ in 0..probe {
        let a = Instant::now();
        acc = acc.wrapping_add(a.elapsed().as_nanos() as u64);
    }
    let overhead = t.elapsed().as_secs_f64() / probe as f64 * 1e9;
    println!("timer overhead ~{overhead:.0} ns per measurement (acc {acc}), subtracted below");

    // record a move sequence with the reference so every game is a normal, legal game
    let mut rng = Rng::new(seed);
    let d0: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
    let d1: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
    let mut rec: Vec<(Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<bool>)> = Vec::new();
    {
        let mut g = BatchedLinesGame::new(n, seed);
        let mut r = Rng::new(seed ^ 0xc0ffee);
        loop {
            let active: Vec<bool> = g.finished.iter().map(|&f| !f).collect();
            if !active.iter().any(|&a| a) {
                break;
            }
            let (m0, m1) = g.raw_masks();
            let (r0, c0) = sample_move_kernel(&d0, &m0, &active, n, HEIGHT, WIDTH, &mut r);
            let (r1, c1) = sample_move_kernel(&d1, &m1, &active, n, HEIGHT, WIDTH, &mut r);
            apply_and_score_kernel(
                &mut g.boards, &mut g.move_counts, &mut g.finished, &mut g.scores,
                &r0, &c0, &r1, &c1, &active, n, HEIGHT, WIDTH,
            );
            rec.push((r0, c0, r1, c1, active));
        }
    }

    // The same recorded games, replayed once per repair strategy. `--bfs-first` reverses
    // the order, which is the only way to tell a real difference from whichever run happens
    // to go second on a warm cache.
    let bfs_first = args.iter().any(|a| a == "--bfs-first");
    let (samples, inc_walk, bfs_samples, inc_bfs) = if bfs_first {
        let (b, ib) = replay(&rec, n, seed, overhead, true);
        let (w, iw) = replay(&rec, n, seed, overhead, false);
        (w, iw, b, ib)
    } else {
        let (w, iw) = replay(&rec, n, seed, overhead, false);
        let (b, ib) = replay(&rec, n, seed, overhead, true);
        (w, iw, b, ib)
    };
    assert_eq!(samples.len(), bfs_samples.len(), "the two replays diverged");
    assert_eq!(inc_walk.boards, inc_bfs.boards, "the two strategies produced different boards");
    assert_eq!(inc_walk.levels, inc_bfs.levels, "the two strategies produced different levels");
    assert_eq!(inc_walk.scores, inc_bfs.scores, "the two strategies produced different scores");

    let total = samples.len();
    let all: Vec<f64> = samples.iter().map(|s| s.ns).collect();
    println!("\n{} games, {} game-moves timed individually", n, total);
    histogram("ALL MOVES", &all, total);

    let classified = samples.iter().any(|s| s.path != Path::Unknown);
    if !classified {
        println!("\n(rebuild with `--features stats` to split this by path)");
        return;
    }

    println!("\n=== split by path ===");
    let mut rows = Vec::new();
    for p in Path::all() {
        let v: Vec<f64> = samples.iter().filter(|s| s.path == p).map(|s| s.ns).collect();
        if v.is_empty() {
            continue;
        }
        histogram(p.label(), &v, total);
        let mut sv = v.clone();
        sv.sort_by(|a, b| a.partial_cmp(b).unwrap());
        rows.push((p, v.len(), pct(&sv, 0.5), v.iter().sum::<f64>() / v.len() as f64,
                   v.iter().sum::<f64>()));
    }

    println!("\n=== summary: where the time actually goes ===");
    println!(
        "{:<44} {:>9} {:>7} {:>9} {:>9} {:>8}",
        "path", "moves", "share", "median", "mean", "of total"
    );
    let grand: f64 = samples.iter().map(|s| s.ns).sum();
    for (p, count, med, mean, sum) in &rows {
        println!(
            "{:<44} {:>9} {:>6.1}% {:>7.0}ns {:>7.0}ns {:>7.1}%",
            p.label(), count, *count as f64 / total as f64 * 100.0, med, mean,
            sum / grand * 100.0
        );
    }

    // is the slow tail really the severed-cycle climb?
    let mut with_pops: Vec<(u64, f64)> =
        samples.iter().filter(|s| s.repair_pops > 0).map(|s| (s.repair_pops, s.ns)).collect();
    if !with_pops.is_empty() {
        with_pops.sort_by_key(|&(p, _)| p);
        let tail: Vec<f64> = samples.iter().filter(|s| s.ns > 2000.0).map(|s| s.ns).collect();
        let tail_pops: u64 =
            samples.iter().filter(|s| s.ns > 2000.0).map(|s| s.repair_pops).sum();
        let all_pops: u64 = samples.iter().map(|s| s.repair_pops).sum();
        println!(
            "\nmoves doing any repair work: {} ({:.2}%), {} repair pops total",
            with_pops.len(),
            with_pops.len() as f64 / total as f64 * 100.0,
            all_pops
        );
        println!(
            "moves slower than 2000 ns:   {} ({:.2}%), holding {:.0}% of all repair pops",
            tail.len(),
            tail.len() as f64 / total as f64 * 100.0,
            if all_pops == 0 { 0.0 } else { tail_pops as f64 / all_pops as f64 * 100.0 }
        );
        let hi = &with_pops[with_pops.len() * 9 / 10..];
        println!(
            "top decile by repair pops:   {} pops median, {:.0} ns median",
            hi[hi.len() / 2].0,
            hi[hi.len() / 2].1
        );
    }

    // ---- does the climb really run all the way to the cap? ----
    let repairs: Vec<&Sample> = samples.iter().filter(|s| s.repair_pops > 0).collect();
    if !repairs.is_empty() {
        let capped: u64 = repairs.iter().map(|s| s.capped).sum();
        let dead: u64 = repairs.iter().map(|s| s.dead).sum();
        println!("\n=== how far do the levels actually climb? ===");
        println!(
            "cells that went INF: {} by the cap firing ({:.1}%), {} by losing every finite \
             neighbour ({:.1}%)",
            capped,
            capped as f64 / (capped + dead).max(1) as f64 * 100.0,
            dead,
            dead as f64 / (capped + dead).max(1) as f64 * 100.0
        );

        let mut lv: Vec<u8> = repairs.iter().map(|s| s.repair_max_level).collect();
        lv.sort_unstable();
        println!(
            "highest level written per repair: min {}  p50 {}  p90 {}  p99 {}  max {}  \
             (MAX_LEVEL = {})",
            lv[0], lv[lv.len() / 2], lv[lv.len() * 9 / 10], lv[lv.len() * 99 / 100],
            lv[lv.len() - 1], alpha_lines_game::incremental::MAX_LEVEL
        );
        let mut buckets = [0usize; 9];
        for &v in &lv {
            let b = (v as usize / 10).min(8);
            buckets[b] += 1;
        }
        for (b, &c) in buckets.iter().enumerate() {
            if c == 0 {
                continue;
            }
            println!(
                "  levels {:>2}-{:<3} {:>7} repairs {:5.1}% {}",
                b * 10, b * 10 + 9, c, c as f64 / lv.len() as f64 * 100.0,
                "#".repeat((c * 50 / lv.len().max(1)).max(1))
            );
        }

        // The climb is bimodal, so the two modes are really two different workloads. Time
        // them separately: a repair that settles immediately and one that runs to the cap
        // should not be averaged together.
        println!("\n=== time distribution of the two climb modes (repair_up) ===");
        let settled: Vec<f64> = samples
            .iter()
            .filter(|s| s.repair_pops > 0 && s.repair_max_level < 20)
            .map(|s| s.ns)
            .collect();
        let climbed: Vec<f64> = samples
            .iter()
            .filter(|s| s.repair_pops > 0 && s.repair_max_level >= 80)
            .map(|s| s.ns)
            .collect();
        histogram("repairs that settled within 20 levels", &settled, total);
        histogram("repairs that ran to the cap (level 80+)", &climbed, total);

        // and the same two sets of moves under the component rebuild
        let pick = |keep: &dyn Fn(&Sample) -> bool| -> Vec<f64> {
            samples
                .iter()
                .zip(bfs_samples.iter())
                .filter(|(o, _)| keep(o))
                .map(|(_, b)| b.ns)
                .collect()
        };
        histogram(
            "^ the same moves, rebuilt: settled within 20 levels",
            &pick(&|s: &Sample| s.repair_pops > 0 && s.repair_max_level < 20),
            total,
        );
        histogram(
            "^ the same moves, rebuilt: ran to the cap",
            &pick(&|s: &Sample| s.repair_pops > 0 && s.repair_max_level >= 80),
            total,
        );

        // are the slowest moves slow because of real work, or because the OS stole the CPU?
        let mut worst: Vec<&Sample> = samples.iter().collect();
        worst.sort_by(|a, b| b.ns.partial_cmp(&a.ns).unwrap());
        println!("\nthe 10 slowest moves, and whether their cost is accounted for:");
        println!("  {:>10} {:>12} {:>12} {:>8}", "ns", "repair pops", "ns per pop", "maxlvl");
        for w in worst.iter().take(10) {
            let per = if w.repair_pops == 0 { f64::NAN } else { w.ns / w.repair_pops as f64 };
            println!("  {:>10.0} {:>12} {:>12.1} {:>8}", w.ns, w.repair_pops, per, w.repair_max_level);
        }
    }

    // ---- repair_up vs. rebuilding the component with a BFS ----
    //
    // Same games, same moves, same order; the only difference is which repair the slow
    // removal path runs. Rows are grouped by what the *old* strategy did, because that is
    // what the choice is being made about: the climb modes only exist in repair_up.
    println!("\n=== slow-removal repair: walking levels up vs. rebuilding the component ===");
    println!(
        "{:<44} {:>8} {:>10} {:>10} {:>10} {:>10} {:>9}",
        "group (classified by the repair_up run)", "moves", "old med", "old mean", "new med",
        "new mean", "mean dx"
    );
    let row = |label: &str, keep: &dyn Fn(&Sample) -> bool| {
        let old: Vec<f64> =
            samples.iter().filter(|s| keep(s)).map(|s| s.ns).collect();
        let new: Vec<f64> = samples
            .iter()
            .zip(bfs_samples.iter())
            .filter(|(o, _)| keep(o))
            .map(|(_, b)| b.ns)
            .collect();
        if old.is_empty() {
            return;
        }
        let (om, oa) = med_mean(&old);
        let (nm, na) = med_mean(&new);
        println!(
            "{:<44} {:>8} {:>8.0}ns {:>8.0}ns {:>8.0}ns {:>8.0}ns {:>8.2}x",
            label, old.len(), om, oa, nm, na, oa / na.max(1e-9)
        );
    };
    row("all moves", &|_: &Sample| true);
    row("moves with a slow removal", &|s: &Sample| s.slow_removals > 0);
    row("  ... that settled within 20 levels", &|s: &Sample| {
        s.repair_pops > 0 && s.repair_max_level < 20
    });
    row("  ... that ran to the cap (level 80+)", &|s: &Sample| {
        s.repair_pops > 0 && s.repair_max_level >= 80
    });
    row("moves with no repair work at all", &|s: &Sample| s.repair_pops == 0);
    for p in Path::all() {
        let lbl = format!("path: {}", p.label());
        row(&lbl, &|s: &Sample| s.path == p);
    }

    let old_total: f64 = samples.iter().map(|s| s.ns).sum();
    let new_total: f64 = bfs_samples.iter().map(|s| s.ns).sum();
    let old_pops: u64 = samples.iter().map(|s| s.repair_pops).sum();
    let new_pops: u64 = bfs_samples.iter().map(|s| s.repair_pops).sum();
    println!(
        "\ntotal move time: {:.1} ms walking up, {:.1} ms rebuilding ({:+.1}%)",
        old_total / 1e6,
        new_total / 1e6,
        (new_total - old_total) / old_total * 100.0
    );
    println!(
        "repair cells touched: {} walking up, {} rebuilding ({:.1}x fewer)",
        old_pops,
        new_pops,
        old_pops as f64 / new_pops.max(1) as f64
    );
}
