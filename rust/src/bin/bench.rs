//! Performance driver. Mirrors `rust/xcheck/bench.py` one-for-one so the two numbers are
//! measuring the same work.
//!
//! Usage:
//!   bench [--games N] [--reps R] [--warmup-moves K] [--seed S] [--rollout-reps R]
//!
//! Emits one `name<TAB>seconds<TAB>iters<TAB>unit_ns` line per benchmark on stdout.

use std::hint::black_box;
use std::time::Instant;

use alpha_lines_game::config::{HEIGHT, WIDTH};
use alpha_lines_game::game_kernels::{apply_and_score_kernel, legal_masks_kernel,
                                     sample_move_kernel, score_batch,
                                     PLAYER_0_MARK, PLAYER_1_MARK};
use alpha_lines_game::rng::Rng;
use alpha_lines_game::{BatchedLinesGame, IncrementalGame};

fn arg_usize(args: &[String], key: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == key)
        .map(|i| args[i + 1].parse().expect("numeric argument"))
        .unwrap_or(default)
}

fn report(name: &str, secs: f64, iters: usize) {
    println!("{}\t{:.9}\t{}\t{:.1}", name, secs, iters, secs / iters as f64 * 1e9);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n = arg_usize(&args, "--games", 2048);
    let reps = arg_usize(&args, "--reps", 20);
    let warmup_moves = arg_usize(&args, "--warmup-moves", 10);
    let seed = arg_usize(&args, "--seed", 12345) as u64;
    // Independent rollouts to average over. At small batch sizes a single rollout is a
    // sample of one game, whose cost varies by 12x, so it says nothing on its own.
    let rollout_reps = arg_usize(&args, "--rollout-reps", 1);
    // slow-removal repair strategy for the incremental engine; state is identical either way
    let bfs = args.iter().any(|a| a == "--bfs");

    let hw = HEIGHT * WIDTH;
    let mut rng = Rng::new(seed);

    // Fixed random policy distributions, reused across steps, exactly like the Python bench.
    let dist_p0: Vec<f32> = (0..n * hw).map(|_| rng.random() as f32).collect();
    let dist_p1: Vec<f32> = (0..n * hw).map(|_| rng.random() as f32).collect();

    // ---- full rollouts to completion via distribution_step ----
    // Games are constructed outside the timed region, matching the Python harness.
    let mut games: Vec<BatchedLinesGame> =
        (0..rollout_reps).map(|k| BatchedLinesGame::new(n, seed + k as u64)).collect();
    let t0 = Instant::now();
    let mut steps = 0usize;
    for game in games.iter_mut() {
        while !game.finished.iter().all(|&f| f) {
            game.distribution_step(&dist_p0, &dist_p1);
            steps += 1;
            assert!(steps < 1000 * rollout_reps.max(1), "rollout did not terminate");
        }
    }
    let rollout = t0.elapsed().as_secs_f64();
    let total_games = n * rollout_reps;
    report("rollout_total", rollout, 1);
    report("rollout_per_step", rollout, steps);
    println!("#\tgames\t{}\tsteps\t{}\trollout_games\t{}", n, steps, total_games);

    // ---- the same rollouts, scored incrementally ----
    let mut incs: Vec<IncrementalGame> =
        (0..rollout_reps).map(|k| IncrementalGame::new(n, seed + k as u64)).collect();
    incs.iter_mut().for_each(|i| i.set_bfs_repair(bfs));
    let t0 = Instant::now();
    let mut inc_steps = 0usize;
    for inc in incs.iter_mut() {
        while !inc.finished.iter().all(|&f| f) {
            inc.distribution_step(&dist_p0, &dist_p1);
            inc_steps += 1;
            assert!(inc_steps < 1000 * rollout_reps.max(1), "incremental rollout did not terminate");
        }
    }
    let rollout_inc = t0.elapsed().as_secs_f64();
    report("rollout_incremental_total", rollout_inc, 1);
    report("rollout_incremental_per_step", rollout_inc, inc_steps);

    // ---- scoring alone, with masks and sampling factored out ----
    // A rollout spends most of its time outside the scorer, so replay a pre-recorded move
    // sequence through each scorer's apply path to isolate the part that actually differs.
    let mut recorded: Vec<(Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<bool>)> = Vec::new();
    {
        let mut g = BatchedLinesGame::new(n, seed);
        loop {
            let active: Vec<bool> = g.finished.iter().map(|&f| !f).collect();
            if !active.iter().any(|&a| a) {
                break;
            }
            let (m0, m1) = g.raw_masks();
            let (r0, c0) = sample_move_kernel(&dist_p0, &m0, &active, n, HEIGHT, WIDTH, &mut rng);
            let (r1, c1) = sample_move_kernel(&dist_p1, &m1, &active, n, HEIGHT, WIDTH, &mut rng);
            apply_and_score_kernel(
                &mut g.boards, &mut g.move_counts, &mut g.finished, &mut g.scores,
                &r0, &c0, &r1, &c1, &active, n, HEIGHT, WIDTH,
            );
            recorded.push((r0, c0, r1, c1, active));
        }
    }

    let mut ref_game = BatchedLinesGame::new(n, seed);
    let t = Instant::now();
    for (r0, c0, r1, c1, active) in &recorded {
        apply_and_score_kernel(
            &mut ref_game.boards, &mut ref_game.move_counts, &mut ref_game.finished,
            &mut ref_game.scores, r0, c0, r1, c1, active, n, HEIGHT, WIDTH,
        );
    }
    report("apply_and_score_reference", t.elapsed().as_secs_f64(), recorded.len());

    let mut inc_game = IncrementalGame::new(n, seed);
    inc_game.set_bfs_repair(bfs);
    let t = Instant::now();
    for (r0, c0, r1, c1, active) in &recorded {
        inc_game.apply_step(r0, c0, r1, c1, active);
    }
    report("apply_and_score_incremental", t.elapsed().as_secs_f64(), recorded.len());

    // the two apply paths must have produced identical results, or the numbers are meaningless
    assert_eq!(inc_game.boards, ref_game.boards, "apply paths diverged");
    assert_eq!(inc_game.scores_f32(), ref_game.scores, "apply paths scored differently");

    // ---- micro-benchmarks on a representative mid-game batch ----
    let mut mid = BatchedLinesGame::new(n, seed ^ 0x5eed);
    for _ in 0..warmup_moves {
        mid.distribution_step(&dist_p0, &dist_p1);
    }
    let active: Vec<bool> = mid.finished.iter().map(|&f| !f).collect();
    let (mask_0, _m1, _c0, _c1) =
        legal_masks_kernel(&mid.boards, mid.n, &mid.move_counts, mid.half_width, HEIGHT, WIDTH);

    let t = Instant::now();
    for _ in 0..reps {
        black_box(legal_masks_kernel(
            black_box(&mid.boards),
            mid.n,
            &mid.move_counts,
            mid.half_width,
            HEIGHT,
            WIDTH,
        ));
    }
    report("legal_masks_kernel", t.elapsed().as_secs_f64(), reps);

    let t = Instant::now();
    for _ in 0..reps {
        black_box(score_batch(black_box(&mid.boards), mid.n, PLAYER_0_MARK, HEIGHT, WIDTH));
        black_box(score_batch(black_box(&mid.boards), mid.n, PLAYER_1_MARK, HEIGHT, WIDTH));
    }
    report("score_batch_both_players", t.elapsed().as_secs_f64(), reps);

    let t = Instant::now();
    for _ in 0..reps {
        black_box(sample_move_kernel(
            black_box(&dist_p0),
            &mask_0,
            &active,
            mid.n,
            HEIGHT,
            WIDTH,
            &mut rng,
        ));
    }
    report("sample_move_kernel", t.elapsed().as_secs_f64(), reps);

    let t = Instant::now();
    for _ in 0..reps {
        black_box(mid.get_encoded_states(black_box(0)));
    }
    report("get_encoded_states", t.elapsed().as_secs_f64(), reps);
}
