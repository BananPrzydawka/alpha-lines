//! Per-function breakdown of a full rollout, for both Rust implementations.
//!
//! `distribution_step` is four distinct pieces of work. This runs the same loop with each
//! piece timed separately, so it is visible which one moves with batch size and which does
//! not:
//!
//!   active   building the ~finished mask (allocates an N-element Vec per step)
//!   masks    legal_masks_kernel        (reference only; the incremental engine's live
//!                                       list replaces it, so this column is 0 there)
//!   sample   the sampler, twice
//!   apply    apply_and_score_kernel, or the incremental apply_step (which also compacts
//!            the live list)
//!
//! Small batches are averaged over enough independent rollouts to cover ~2048 games, since
//! one rollout at n=1 is a sample of a single game.
//!
//! Usage: phases [--seed S] [--target-games G]

use std::time::Instant;

use alpha_lines_game::config::{HEIGHT, WIDTH};
use alpha_lines_game::game_kernels::{apply_and_score_kernel, sample_move_kernel};
use alpha_lines_game::incremental::HW;
use alpha_lines_game::rng::Rng;
use alpha_lines_game::{BatchedLinesGame, IncrementalGame};

#[derive(Default, Clone, Copy)]
struct Phases {
    active: f64,
    masks: f64,
    sample: f64,
    apply: f64,
    steps: usize,
}

impl Phases {
    fn sum(&self) -> f64 {
        self.active + self.masks + self.sample + self.apply
    }
}

fn arg_usize(args: &[String], key: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == key)
        .map(|i| args[i + 1].parse().expect("numeric"))
        .unwrap_or(default)
}

fn reference(n: usize, reps: usize, seed: u64, d0: &[f32], d1: &[f32]) -> Phases {
    let mut games: Vec<BatchedLinesGame> =
        (0..reps).map(|k| BatchedLinesGame::new(n, seed + k as u64)).collect();
    let mut p = Phases::default();
    for g in games.iter_mut() {
        loop {
            let t = Instant::now();
            let active: Vec<bool> = g.finished.iter().map(|&f| !f).collect();
            p.active += t.elapsed().as_secs_f64();
            if !active.iter().any(|&a| a) {
                break;
            }

            let t = Instant::now();
            let (m0, m1) = g.raw_masks();
            p.masks += t.elapsed().as_secs_f64();

            let t = Instant::now();
            let (r0, c0) = sample_move_kernel(d0, &m0, &active, n, HEIGHT, WIDTH, &mut g.rng);
            let (r1, c1) = sample_move_kernel(d1, &m1, &active, n, HEIGHT, WIDTH, &mut g.rng);
            p.sample += t.elapsed().as_secs_f64();

            let t = Instant::now();
            apply_and_score_kernel(
                &mut g.boards, &mut g.move_counts, &mut g.finished, &mut g.scores,
                &r0, &c0, &r1, &c1, &active, n, HEIGHT, WIDTH,
            );
            p.apply += t.elapsed().as_secs_f64();
            p.steps += 1;
        }
    }
    p
}

fn incremental(n: usize, reps: usize, seed: u64, d0: &[f32], d1: &[f32]) -> Phases {
    let mut games: Vec<IncrementalGame> =
        (0..reps).map(|k| IncrementalGame::new(n, seed + k as u64)).collect();
    let mut p = Phases::default();
    let (mut r0, mut c0) = (vec![0i64; n], vec![0i64; n]);
    let (mut r1, mut c1) = (vec![0i64; n], vec![0i64; n]);
    for g in games.iter_mut() {
        loop {
            let t = Instant::now();
            let active: Vec<bool> = g.finished.iter().map(|&f| !f).collect();
            p.active += t.elapsed().as_secs_f64();
            if !active.iter().any(|&a| a) {
                break;
            }

            // The incremental engine never builds the masks: the live list already knows
            // which squares are available, so `masks` is structurally zero here.
            let t = Instant::now();
            g.sample_moves(d0, 0, &active, &mut r0, &mut c0);
            g.sample_moves(d1, 1, &active, &mut r1, &mut c1);
            p.sample += t.elapsed().as_secs_f64();

            let t = Instant::now();
            g.apply_step(&r0, &c0, &r1, &c1, &active);
            p.apply += t.elapsed().as_secs_f64();
            p.steps += 1;
        }
    }
    p
}

/// The same rollout with no timers inside it, to expose the instrumentation overhead.
fn uninstrumented(n: usize, reps: usize, seed: u64, d0: &[f32], d1: &[f32], inc: bool) -> f64 {
    if inc {
        let mut games: Vec<IncrementalGame> =
            (0..reps).map(|k| IncrementalGame::new(n, seed + k as u64)).collect();
        let t = Instant::now();
        for g in games.iter_mut() {
            while !g.finished.iter().all(|&f| f) {
                g.distribution_step(d0, d1);
            }
        }
        t.elapsed().as_secs_f64()
    } else {
        let mut games: Vec<BatchedLinesGame> =
            (0..reps).map(|k| BatchedLinesGame::new(n, seed + k as u64)).collect();
        let t = Instant::now();
        for g in games.iter_mut() {
            while !g.finished.iter().all(|&f| f) {
                g.distribution_step(d0, d1);
            }
        }
        t.elapsed().as_secs_f64()
    }
}

fn table(name: &str, sizes: &[usize], seed: u64, target: usize, inc: bool) {
    println!("\n=== {name}: microseconds per completed game ===");
    println!(
        "{:>7} {:>9}  {:>9} {:>9} {:>9} {:>9}  {:>9} {:>9}",
        "games", "rollouts", "active", "masks", "sample", "apply", "sum", "untimed"
    );
    println!("{}", "-".repeat(82));
    for &n in sizes {
        let reps = (target / n).max(1);
        let mut rng = Rng::new(seed);
        let d0: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
        let d1: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();

        let p = if inc {
            incremental(n, reps, seed, &d0, &d1)
        } else {
            reference(n, reps, seed, &d0, &d1)
        };
        let raw = uninstrumented(n, reps, seed, &d0, &d1, inc);

        let games = (n * reps) as f64;
        let us = |t: f64| t / games * 1e6;
        println!(
            "{:>7} {:>9}  {:>8.1}us {:>8.1}us {:>8.1}us {:>8.1}us  {:>8.1}us {:>8.1}us",
            n, reps, us(p.active), us(p.masks), us(p.sample), us(p.apply), us(p.sum()), us(raw)
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed = arg_usize(&args, "--seed", 12345) as u64;
    let target = arg_usize(&args, "--target-games", 2048);
    let sizes = [1usize, 4, 16, 64, 256, 1024, 2048, 8192];

    println!("per-phase breakdown of a full rollout, averaged over ~{target} completed games");
    println!("'sum' is the four timed phases; 'untimed' is the same rollout with no timers,");
    println!("so the gap between them is the instrumentation cost.");

    table("rust reference", &sizes, seed, target, false);
    table("rust incremental", &sizes, seed, target, true);
}
