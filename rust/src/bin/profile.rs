//! Batch-size profile, part by part.
//!
//! Answers two questions the benchmark cannot: how each stage scales with the number of
//! games in the batch, and which stages are doing genuinely batched work versus a loop of
//! independent per-game work.
//!
//! Timings come from the default build. Build with `--features stats` to also get the
//! per-path counters for the incremental scorer (those add ~5% to a move, so that run's
//! timings should be ignored).
//!
//! Usage: profile [--reps R] [--seed S] [--max-games N]

use std::hint::black_box;
use std::time::Instant;

use alpha_lines_game::config::{HEIGHT, WIDTH};
use alpha_lines_game::game_kernels::{
    apply_and_score_kernel, legal_masks_kernel, sample_move_kernel,
};
use alpha_lines_game::incremental::HW;
use alpha_lines_game::rng::Rng;
use alpha_lines_game::{BatchedLinesGame, IncrementalGame};

type Move = (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<bool>);

fn arg_usize(args: &[String], key: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == key)
        .map(|i| args[i + 1].parse().expect("numeric argument"))
        .unwrap_or(default)
}

/// Plays a batch to completion with the reference, recording every move so both scorers can
/// be replayed over identical work.
fn record(n: usize, seed: u64, dist_p0: &[f32], dist_p1: &[f32]) -> (Vec<Move>, u64) {
    let mut rng = Rng::new(seed ^ 0xc0ffee);
    let mut g = BatchedLinesGame::new(n, seed);
    let mut out = Vec::new();
    let mut active_moves = 0u64;
    loop {
        let active: Vec<bool> = g.finished.iter().map(|&f| !f).collect();
        let live = active.iter().filter(|&&a| a).count();
        if live == 0 {
            break;
        }
        active_moves += live as u64;
        let (m0, m1) = g.raw_masks();
        let (r0, c0) = sample_move_kernel(dist_p0, &m0, &active, n, HEIGHT, WIDTH, &mut rng);
        let (r1, c1) = sample_move_kernel(dist_p1, &m1, &active, n, HEIGHT, WIDTH, &mut rng);
        apply_and_score_kernel(
            &mut g.boards, &mut g.move_counts, &mut g.finished, &mut g.scores,
            &r0, &c0, &r1, &c1, &active, n, HEIGHT, WIDTH,
        );
        out.push((r0, c0, r1, c1, active));
    }
    (out, active_moves)
}

/// Minimum wall time over `reps` fresh replays. Setup is outside the timed region.
fn min_time<S, T>(reps: usize, mut setup: impl FnMut() -> S, mut run: impl FnMut(&mut S) -> T) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let mut state = setup();
        let t = Instant::now();
        black_box(run(&mut state));
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps = arg_usize(&args, "--reps", 5);
    let seed = arg_usize(&args, "--seed", 12345) as u64;
    let max_games = arg_usize(&args, "--max-games", 8192);

    let sizes: Vec<usize> = [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        .into_iter()
        .filter(|&n| n <= max_games)
        .collect();

    println!("board {}x{}, {} cells, {} playable", HEIGHT, WIDTH, HW, HW / 2);
    println!(
        "per game: reference {} B (board+scores), incremental {} B (board+levels+scores)",
        HW + 8,
        HW + HW + 8
    );
    println!("timings are the minimum of {reps} runs\n");

    println!("--- scoring: ns per active game-move (one game advancing one move) ---");
    println!(
        "{:>7} {:>8} {:>14} {:>14} {:>9}   {:>12} {:>12}",
        "games", "moves", "reference", "incremental", "speedup", "ref total", "inc total"
    );
    let mut rows = Vec::new();
    for &n in &sizes {
        let mut rng = Rng::new(seed);
        let dist_p0: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
        let dist_p1: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
        let (recorded, active_moves) = record(n, seed, &dist_p0, &dist_p1);

        let ref_t = min_time(
            reps,
            || BatchedLinesGame::new(n, seed),
            |g| {
                for (r0, c0, r1, c1, active) in &recorded {
                    apply_and_score_kernel(
                        &mut g.boards, &mut g.move_counts, &mut g.finished, &mut g.scores,
                        r0, c0, r1, c1, active, n, HEIGHT, WIDTH,
                    );
                }
            },
        );
        let inc_t = min_time(
            reps,
            || IncrementalGame::new(n, seed),
            |g| {
                for (r0, c0, r1, c1, active) in &recorded {
                    g.apply_step(r0, c0, r1, c1, active);
                }
            },
        );

        let rn = ref_t / active_moves as f64 * 1e9;
        let inn = inc_t / active_moves as f64 * 1e9;
        println!(
            "{:>7} {:>8} {:>11.1} ns {:>11.1} ns {:>8.1}x   {:>9.2} ms {:>9.2} ms",
            n, active_moves, rn, inn, rn / inn, ref_t * 1e3, inc_t * 1e3
        );
        rows.push((n, rn, inn));
    }

    println!("\n--- surrounding stages: ns per game per step (identical code in both) ---");
    println!(
        "{:>7} {:>16} {:>16} {:>16}",
        "games", "legal_masks", "sample_move", "get_encoded"
    );
    for &n in &sizes {
        let mut rng = Rng::new(seed);
        let dist: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
        let dist2: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
        let mut mid = BatchedLinesGame::new(n, seed);
        for _ in 0..10 {
            mid.distribution_step(&dist, &dist2);
        }
        let active: Vec<bool> = mid.finished.iter().map(|&f| !f).collect();
        let (mask, _, _, _) =
            legal_masks_kernel(&mid.boards, n, &mid.move_counts, mid.half_width, HEIGHT, WIDTH);

        let masks = min_time(reps, || (), |_| {
            legal_masks_kernel(&mid.boards, n, &mid.move_counts, mid.half_width, HEIGHT, WIDTH)
        });
        let mut srng = Rng::new(7);
        let sample = min_time(reps, || (), |_| {
            sample_move_kernel(&dist, &mask, &active, n, HEIGHT, WIDTH, &mut srng)
        });
        let encode = min_time(reps, || (), |_| mid.get_encoded_states(0));

        println!(
            "{:>7} {:>13.1} ns {:>13.1} ns {:>13.1} ns",
            n,
            masks / n as f64 * 1e9,
            sample / n as f64 * 1e9,
            encode / n as f64 * 1e9
        );
    }

    // ---- path distribution ----
    let n = 2048.min(max_games);
    let mut rng = Rng::new(seed);
    let dist_p0: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
    let dist_p1: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
    let (recorded, active_moves) = record(n, seed, &dist_p0, &dist_p1);
    let mut g = IncrementalGame::new(n, seed);
    g.reset_stats();
    for (r0, c0, r1, c1, active) in &recorded {
        g.apply_step(r0, c0, r1, c1, active);
    }
    let st = g.stats();
    if st.inserts == 0 && st.removes == 0 {
        println!("\n(path counters need `--features stats`)");
        return;
    }
    println!("\n--- which path each operation took, over {active_moves} game-moves at n={n} ---");
    let pct = |a: u64, b: u64| if b == 0 { 0.0 } else { a as f64 / b as f64 * 100.0 };
    println!(
        "inserts {:>9}   fast {:>9} ({:5.1}%)  slow {:>7} ({:4.2}%)  dead {:>7} ({:4.2}%)",
        st.inserts,
        st.inserts_fast, pct(st.inserts_fast, st.inserts),
        st.inserts_slow, pct(st.inserts_slow, st.inserts),
        st.inserts_dead, pct(st.inserts_dead, st.inserts),
    );
    println!(
        "removes {:>9}   fast {:>9} ({:5.1}%)  slow {:>7} ({:4.2}%)  dead {:>7} ({:4.2}%)  empty {:>7}",
        st.removes,
        st.removes_fast, pct(st.removes_fast, st.removes),
        st.removes_slow, pct(st.removes_slow, st.removes),
        st.removes_dead, pct(st.removes_dead, st.removes),
        st.removes_nonmark,
    );
    let ops = st.inserts + st.removes;
    println!(
        "level pass work: {} relax pops + {} repair pops = {:.2} per operation",
        st.relax_pops,
        st.repair_pops,
        (st.relax_pops + st.repair_pops) as f64 / ops as f64
    );
    println!(
        "component walks: {} walks, {} cells visited = {:.2} cells per operation",
        st.component_walks,
        st.component_cells,
        st.component_cells as f64 / ops as f64
    );
}
