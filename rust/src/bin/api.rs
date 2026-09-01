//! Per-call cost of every public function on the engine.
//!
//! The rollout benchmarks answer "how fast is a game"; this answers "how fast is a call",
//! which is the question that matters to whatever drives the engine from outside. Pure
//! accessors are timed on a representative mid-game batch and averaged over `--reps`;
//! mutating calls are timed over a whole rollout and divided by the number of calls, which
//! is the only honest average for something whose cost changes as the board fills.
//!
//! One caveat on the numbers: any call that returns owned `Vec`s allocates and frees on
//! every invocation — 2.6 MB per call for the masks at 2048 games — and that cost depends on
//! the process's heap and cache state as much as on the code. The same `get_legal_masks`
//! measures ~315 ns/game here and ~550 ns/game inside `bench`, which keeps several MB of
//! recorded moves live throughout. The `_into` variants allocate nothing and are the stable
//! comparison; the allocating rows are only comparable against each other, in this table.
//!
//! Usage: api [--games N] [--reps R] [--warmup-moves M] [--seed S]

use std::hint::black_box;
use std::time::Instant;

use alpha_lines_game::incremental::{Rng, HEIGHT, HW, WIDTH};
use alpha_lines_game::IncrementalGame;

fn arg_usize(args: &[String], key: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == key)
        .map(|i| args[i + 1].parse().expect("numeric"))
        .unwrap_or(default)
}

struct Table {
    n: usize,
    rows: Vec<(String, f64, &'static str)>,
}

impl Table {
    /// `secs` covers `calls` invocations over the whole batch of `n` games.
    fn add(&mut self, name: &str, secs: f64, calls: usize, unit: &'static str) {
        self.rows.push((name.to_string(), secs / calls as f64, unit));
    }
    fn print(&self, title: &str) {
        println!("\n=== {title} ===");
        println!("{:<34} {:>14}   {}", "call", "per call", "what one call covers");
        for (name, per_call, unit) in &self.rows {
            println!("{:<34} {:>12.1}us   {}", name, per_call * 1e6, unit);
        }
        let _ = self.n;
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n = arg_usize(&args, "--games", 2048);
    let reps = arg_usize(&args, "--reps", 100);
    let warmup = arg_usize(&args, "--warmup-moves", 10);
    let seed = arg_usize(&args, "--seed", 12345) as u64;

    let mut rng = Rng::new(seed);
    let d0: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
    let d1: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();

    // A mid-game batch: the pure accessors are timed on a board that has been played into,
    // not on the opening position, where the fast paths are unrepresentative.
    let mut mid_inc = IncrementalGame::new(n, seed);
    for _ in 0..warmup {
        mid_inc.distribution_step(&d0, &d1);
    }
    let idx: Vec<usize> = (0..n).step_by(2).collect();

    // ---------------------------------------------------------------- incremental engine
    let mut t = Table { n, rows: Vec::new() };

    let s = Instant::now();
    for _ in 0..reps { black_box(IncrementalGame::new(black_box(n), seed)); }
    t.add("new", s.elapsed().as_secs_f64(), reps, "whole batch");

    let b = mid_inc.boards.clone();
    let s = Instant::now();
    for _ in 0..reps { black_box(IncrementalGame::from_state(b.clone(), 0)); }
    t.add("from_state", s.elapsed().as_secs_f64(), reps, "whole batch");

    let s = Instant::now();
    for _ in 0..reps { for g in 0..n { black_box(mid_inc.legal_bits(black_box(g))); } }
    t.add("legal_bits (every game)", s.elapsed().as_secs_f64(), reps, "whole batch");

    // A yardstick: copying the whole batch state, with no work done on it. Everything above
    // and below has to move at least this much, so it says whether a call is anywhere near
    // being limited by memory rather than by what it computes.
    let state_bytes = mid_inc.boards.len() + mid_inc.levels.len() + n * 16 + n * 8 + n * 4 + n;
    let mut sink_b = mid_inc.boards.clone();
    let mut sink_l = mid_inc.levels.clone();
    let s = Instant::now();
    for _ in 0..reps {
        sink_b.copy_from_slice(black_box(&mid_inc.boards));
        sink_l.copy_from_slice(black_box(&mid_inc.levels));
        black_box(&sink_b);
        black_box(&sink_l);
    }
    let copy = s.elapsed().as_secs_f64() / reps as f64;
    t.add("memcpy boards+levels (yardstick)", s.elapsed().as_secs_f64(), reps, "whole batch");
    println!(
        "(batch state is {} KB; copying the {} KB of boards+levels takes {:.1}us, ~{:.1} GB/s)",
        state_bytes / 1024,
        (mid_inc.boards.len() + mid_inc.levels.len()) / 1024,
        copy * 1e6,
        (mid_inc.boards.len() + mid_inc.levels.len()) as f64 * 2.0 / copy / 1e9,
    );

    let s = Instant::now();
    for _ in 0..reps { black_box(mid_inc.clone_states_to_batch(black_box(&idx))); }
    t.add("clone_states_to_batch (n/2)", s.elapsed().as_secs_f64(), reps, "half the batch");

    // mutating calls: one rollout each, averaged over the calls it took
    let mut g = IncrementalGame::new(n, seed);
    let mut steps = 0usize;
    let s = Instant::now();
    while !g.finished.iter().all(|&f| f) { g.distribution_step(&d0, &d1); steps += 1; }
    t.add("distribution_step", s.elapsed().as_secs_f64(), steps, "one move, all games");

    // a recorded legal sequence, so the sampler and the action path can be timed apart
    let mut rec: Vec<(Vec<i64>, Vec<i64>, Vec<bool>)> = Vec::new();
    {
        let mut probe = IncrementalGame::new(n, seed);
        while !probe.finished.iter().all(|&f| f) {
            let active: Vec<bool> = probe.finished.iter().map(|&x| !x).collect();
            let (mut r0, mut c0v) = (vec![0i64; n], vec![0i64; n]);
            let (mut r1, mut c1v) = (vec![0i64; n], vec![0i64; n]);
            probe.sample_moves(&d0, 0, &active, &mut r0, &mut c0v);
            probe.sample_moves(&d1, 1, &active, &mut r1, &mut c1v);
            let i0: Vec<i64> = (0..n).map(|k| r0[k] * WIDTH as i64 + c0v[k]).collect();
            let i1: Vec<i64> = (0..n).map(|k| r1[k] * WIDTH as i64 + c1v[k]).collect();
            probe.action_step(&i0, &i1).unwrap();
            rec.push((i0, i1, active));
        }
    }

    let mut g = IncrementalGame::new(n, seed);
    let (mut ro, mut co) = (vec![0i64; n], vec![0i64; n]);
    let mut calls = 0usize;
    let s = Instant::now();
    for (_, _, active) in &rec {
        g.sample_moves(&d0, 0, active, &mut ro, &mut co);
        calls += 1;
    }
    t.add("sample_moves (one player)", s.elapsed().as_secs_f64(), calls, "one move, all games");

    let mut g = IncrementalGame::new(n, seed);
    let mut calls = 0usize;
    let s = Instant::now();
    for (i0, i1, _) in &rec {
        g.action_step(i0, i1).unwrap();
        calls += 1;
    }
    t.add("action_step", s.elapsed().as_secs_f64(), calls, "one move, all games");

    t.print(format!("IncrementalGame, {n} games, {reps} reps").as_str());


    println!("\n(HEIGHT x WIDTH = {HEIGHT} x {WIDTH}, batch of {n} games)");
}
