//! Isolates *batch size* from *memory access order*.
//!
//! The batch-size sweep shows per-game-move cost apparently rising from n=1 to n>=64. Two
//! hypotheses: (a) something about large batches is intrinsically slower, or (b) it is
//! purely cache residency — at n=1 one 320-byte game stays in L1 for its whole life, while
//! step-major order over N games touches every other game before coming back.
//!
//! This replays an identical recorded move sequence over the identical batch, twice:
//!
//!   step-major:  for each step { for each game { apply } }   <- what the game loop does
//!   game-major:  for each game { for each step { apply } }   <- one game start to finish
//!
//! Same n, same work, same final state (asserted). Any difference is access order alone.
//!
//! It also samples n=1 honestly, over many independent single-game batches, because a
//! single game's cost varies by more than 10x and one sample proves nothing.

use std::time::Instant;

use alpha_lines_game::config::{HEIGHT, WIDTH};
use alpha_lines_game::game_kernels::{apply_and_score_kernel, sample_move_kernel};
use alpha_lines_game::incremental::HW;
use alpha_lines_game::rng::Rng;
use alpha_lines_game::{BatchedLinesGame, IncrementalGame};

type Move = (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<bool>);

fn arg_usize(args: &[String], key: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == key)
        .map(|i| args[i + 1].parse().expect("numeric"))
        .unwrap_or(default)
}

fn record(n: usize, seed: u64, d0: &[f32], d1: &[f32]) -> (Vec<Move>, u64) {
    let mut rng = Rng::new(seed ^ 0xc0ffee);
    let mut g = BatchedLinesGame::new(n, seed);
    let mut out = Vec::new();
    let mut moves = 0u64;
    loop {
        let active: Vec<bool> = g.finished.iter().map(|&f| !f).collect();
        let live = active.iter().filter(|&&a| a).count();
        if live == 0 {
            break;
        }
        moves += live as u64;
        let (m0, m1) = g.raw_masks();
        let (r0, c0) = sample_move_kernel(d0, &m0, &active, n, HEIGHT, WIDTH, &mut rng);
        let (r1, c1) = sample_move_kernel(d1, &m1, &active, n, HEIGHT, WIDTH, &mut rng);
        apply_and_score_kernel(
            &mut g.boards, &mut g.move_counts, &mut g.finished, &mut g.scores,
            &r0, &c0, &r1, &c1, &active, n, HEIGHT, WIDTH,
        );
        out.push((r0, c0, r1, c1, active));
    }
    (out, moves)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps = arg_usize(&args, "--reps", 7);
    let seed = arg_usize(&args, "--seed", 12345) as u64;

    println!("same batch, same moves, only the traversal order differs");
    println!("{:>7} {:>8} {:>13} {:>13} {:>9} {:>10}", "games", "moves", "step-major", "game-major", "ratio", "footprint");

    for n in [1usize, 16, 64, 256, 1024, 2048, 8192] {
        let mut rng = Rng::new(seed);
        let d0: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
        let d1: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
        let (rec, moves) = record(n, seed, &d0, &d1);
        let ns = |t: f64| t / moves as f64 * 1e9;

        let mut step_major = f64::INFINITY;
        let mut final_step = None;
        for _ in 0..reps {
            let mut g = IncrementalGame::new(n, seed);
            let t = Instant::now();
            for (r0, c0, r1, c1, active) in &rec {
                g.apply_step(r0, c0, r1, c1, active);
            }
            step_major = step_major.min(t.elapsed().as_secs_f64());
            final_step = Some((g.boards.clone(), g.scores.clone()));
        }

        let mut game_major = f64::INFINITY;
        let mut final_game = None;
        for _ in 0..reps {
            let mut g = IncrementalGame::new(n, seed);
            let t = Instant::now();
            for game in 0..n {
                for (r0, c0, r1, c1, active) in &rec {
                    if active[game] {
                        g.apply_game(game, r0[game], c0[game], r1[game], c1[game]);
                    }
                }
            }
            game_major = game_major.min(t.elapsed().as_secs_f64());
            final_game = Some((g.boards.clone(), g.scores.clone()));
        }

        assert_eq!(final_step, final_game, "traversal order changed the result at n={n}");

        // same two orders for the reference scorer, whose per-move work is nearly
        // independent of board content
        let mut ref_step = f64::INFINITY;
        for _ in 0..reps {
            let mut g = BatchedLinesGame::new(n, seed);
            let t = Instant::now();
            for (r0, c0, r1, c1, active) in &rec {
                apply_and_score_kernel(
                    &mut g.boards, &mut g.move_counts, &mut g.finished, &mut g.scores,
                    r0, c0, r1, c1, active, n, HEIGHT, WIDTH,
                );
            }
            ref_step = ref_step.min(t.elapsed().as_secs_f64());
        }
        let mut ref_game = f64::INFINITY;
        for _ in 0..reps {
            let mut g = BatchedLinesGame::new(n, seed);
            let t = Instant::now();
            for game in 0..n {
                for (r0, c0, r1, c1, active) in &rec {
                    if active[game] {
                        apply_and_score_kernel(
                            &mut g.boards[game * HW..(game + 1) * HW],
                            &mut g.move_counts[game..game + 1],
                            &mut g.finished[game..game + 1],
                            &mut g.scores[game * 2..game * 2 + 2],
                            &[r0[game]], &[c0[game]], &[r1[game]], &[c1[game]],
                            &[true], 1, HEIGHT, WIDTH,
                        );
                    }
                }
            }
            ref_game = ref_game.min(t.elapsed().as_secs_f64());
        }
        println!(
            "        reference: step-major {:>8.1} ns  game-major {:>8.1} ns  ratio {:.2}x",
            ns(ref_step), ns(ref_game), ns(ref_step) / ns(ref_game)
        );

        let footprint = n * (HW + HW + 8);
        println!(
            "{:>7} {:>8} {:>10.1} ns {:>10.1} ns {:>8.2}x {:>8.0} KB",
            n, moves, ns(step_major), ns(game_major),
            ns(step_major) / ns(game_major),
            footprint as f64 / 1024.0
        );
    }

    // How much does the cost of a single game vary? If the spread is wide, then any
    // measurement taken at n=1 is a sample of one game and says nothing about batch size.
    let n = 2048;
    let mut rng = Rng::new(seed);
    let d0: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
    let d1: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
    let (rec, _) = record(n, seed, &d0, &d1);

    let mut per_game: Vec<f64> = Vec::with_capacity(n);
    let mut g = IncrementalGame::new(n, seed);
    for game in 0..n {
        let mut moves = 0u64;
        let t = Instant::now();
        for (r0, c0, r1, c1, active) in &rec {
            if active[game] {
                g.apply_game(game, r0[game], c0[game], r1[game], c1[game]);
                moves += 1;
            }
        }
        per_game.push(t.elapsed().as_secs_f64() / moves as f64 * 1e9);
    }
    per_game.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean: f64 = per_game.iter().sum::<f64>() / n as f64;
    println!("\nper-game cost spread, incremental, {n} games timed individually (ns per move):");
    println!(
        "  min {:.0}   p10 {:.0}   median {:.0}   mean {:.0}   p90 {:.0}   max {:.0}",
        per_game[0],
        per_game[n / 10],
        per_game[n / 2],
        mean,
        per_game[n * 9 / 10],
        per_game[n - 1]
    );

    // Is n=1 systematically cheaper, or was the single game at this seed just a cheap draw?
    // Every n=1 figure quoted so far is the SAME game measured repeatedly. Sample it.
    let trials = 512usize;
    let mut recs = Vec::with_capacity(trials);
    let mut solos = Vec::with_capacity(trials);
    let mut solo_moves = 0u64;
    for k in 0..trials {
        let sd = seed.wrapping_add(k as u64 * 7919);
        let mut r = Rng::new(sd);
        let a: Vec<f32> = (0..HW).map(|_| r.random() as f32).collect();
        let b: Vec<f32> = (0..HW).map(|_| r.random() as f32).collect();
        let (rec, m) = record(1, sd, &a, &b);
        solo_moves += m;
        recs.push(rec);
        solos.push(IncrementalGame::new(1, sd)); // constructed outside the timed region
    }
    let t = Instant::now();
    for (g, rec) in solos.iter_mut().zip(&recs) {
        for (r0, c0, r1, c1, active) in rec {
            g.apply_step(r0, c0, r1, c1, active);
        }
    }
    let solo = t.elapsed().as_secs_f64() / solo_moves as f64 * 1e9;
    println!(
        "\nn=1 averaged over {trials} independent single-game batches: {:.1} ns per move",
        solo
    );

    let cheap = per_game.iter().filter(|&&x| x < 300.0).count();
    println!(
        "  {} of {} games ({:.0}%) come in under 300 ns/move on their own",
        cheap, n, cheap as f64 / n as f64 * 100.0
    );
}
