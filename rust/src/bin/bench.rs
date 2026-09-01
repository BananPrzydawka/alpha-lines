//! Time per game: build a [`Game`], play it to the end with random moves, repeat.
//!
//! Two samplers are timed, because the choice is worth a factor of 1.5:
//!
//! * **From the bitboard.** A uniform draw over the legal set with no distribution at all —
//!   `popcount`, one bounded draw, one select. This is what a rollout should use.
//! * **Weighted.** A scan over an `HW`-float distribution, which is what a policy network
//!   hands you. Timed twice: over one distribution reused by every game, which stays in L1,
//!   and over a fresh slice per game out of a large array, which does not. The gap between
//!   those two rows is memory traffic, not engine work, and the cold row is the one to quote.
//!
//! Construction is measured on its own rather than folded in, so the column means something:
//! a `Game` is built once per game here, and the number says how much of a rollout that is.
//!
//! This engine replaced a batched one — N boards in flat arrays, stepped in lockstep, an
//! `active` mask saying which were still running. Measured against it on this same workload,
//! batching flattened out at ~22.5 us/game by 64 games and got no better at 4096; one game at
//! a time with the same weighted sampler cost the same 22.1 us, and the bitboard sampler beat
//! every batch size by 1.5x. Batching was amortizing per-step allocation, not vectorizing
//! anything — the scorer chases pointers around one board at a time no matter how many boards
//! are in flight.
//!
//! Every configuration is measured several times, interleaved with the others, and the best
//! time is kept. That is not cherry-picking: on a laptop the clock drifts by 20% over the
//! couple of seconds a straight-through run takes, which is larger than every difference
//! being measured here, and it lands entirely on whichever configuration ran last. Sweeping
//! round-robin puts every configuration under the same conditions, and the minimum is the
//! sample least polluted by whatever else the machine was doing.
//!
//! Usage: bench [--games N] [--positions P] [--sweeps K] [--seed S]

use std::hint::black_box;
use std::time::Instant;

use alpha_lines_game::game::{Rng, Scratch, HW};
use alpha_lines_game::Game;

fn arg_usize(args: &[String], key: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == key)
        .map(|i| args[i + 1].parse().expect("numeric"))
        .unwrap_or(default)
}

/// One sweep's measurement of one configuration.
#[derive(Clone, Copy)]
struct Sample {
    games: usize,
    moves: usize,
    /// seconds for everything: construction plus playing to the end
    total: f64,
    /// seconds of that spent constructing, measured on its own
    construct: f64,
}

impl Sample {
    fn per_game(&self) -> f64 {
        self.total / self.games as f64
    }
}

/// The best sweep seen for one configuration, plus the totals over all of them.
struct Row {
    label: String,
    best: Sample,
    games: usize,
    moves: usize,
}

impl Row {
    fn new(label: &str, first: Sample) -> Self {
        Row { label: label.to_string(), best: first, games: first.games, moves: first.moves }
    }

    fn merge(&mut self, s: Sample) {
        self.games += s.games;
        self.moves += s.moves;
        if s.per_game() < self.best.per_game() {
            self.best = s;
        }
    }

    fn print(&self) {
        let b = &self.best;
        let per_game = b.per_game();
        let per_construct = b.construct / b.games as f64;
        let per_move = (b.total - b.construct) / b.moves as f64;
        println!(
            "{:<26} {:>8} {:>7.1} {:>12.2} {:>11.2} {:>11.0} {:>12.0}",
            self.label,
            self.games,
            self.moves as f64 / self.games as f64,
            per_game * 1e6,
            per_construct * 1e6,
            per_move * 1e9,
            1.0 / per_game,
        );
    }
}

fn header(title: &str) {
    println!("\n=== {title} ===");
    println!(
        "{:<26} {:>8} {:>7} {:>12} {:>11} {:>11} {:>12}",
        "config", "games", "moves", "us/game", "of which", "ns/move", "games/s"
    );
    println!(
        "{:<26} {:>8} {:>7} {:>12} {:>11} {:>11} {:>12}",
        "", "", "/game", "(total)", "construct", "(playing)", ""
    );
}

/// Which sampler the rollouts use.
enum Sampler<'a> {
    /// A uniform draw straight out of the legality bitboard: popcount, draw, select.
    Bits,
    /// The weighted scan, over `HW` weights that stay in cache across every game.
    Hot(&'a [f32]),
    /// The weighted scan, over a fresh slice per game out of a large array.
    Cold(&'a [f32]),
}

fn rollouts(games: usize, seed: u64, sampler: Sampler) -> Sample {
    let mut scratch = Scratch::new();
    let mut rng = Rng::new(seed);
    let mut moves = 0usize;
    let slices = match sampler {
        Sampler::Cold(d) => d.len() / HW,
        _ => 1,
    };

    let t = Instant::now();
    for k in 0..games {
        let mut g = Game::new();
        while !g.finished {
            let (i0, i1) = match sampler {
                Sampler::Bits => (g.random_move(0, &mut rng), g.random_move(1, &mut rng)),
                Sampler::Hot(d) => {
                    (g.sample_move(d, 0, &mut rng), g.sample_move(d, 1, &mut rng))
                }
                Sampler::Cold(d) => {
                    let s = &d[(k % slices) * HW..][..HW];
                    (g.sample_move(s, 0, &mut rng), g.sample_move(s, 1, &mut rng))
                }
            };
            g.apply(i0, i1, &mut scratch);
        }
        moves += g.move_count as usize;
        black_box(&g.scores);
    }
    let total = t.elapsed().as_secs_f64();

    let t = Instant::now();
    for _ in 0..games {
        black_box(Game::new());
    }
    let construct = t.elapsed().as_secs_f64();

    Sample { games, moves, total, construct }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let games = arg_usize(&args, "--games", 10_000);
    let positions = arg_usize(&args, "--positions", 4096);
    let sweeps = arg_usize(&args, "--sweeps", 5).max(1);
    let seed = arg_usize(&args, "--seed", 12345) as u64;

    // Uniform: every square carries the same weight, so the weighted sampler's cumulative
    // scan picks a legal square uniformly at random — the same games the bitboard sampler
    // plays, reached by doing much more work, which is the point of comparing them.
    let dist = vec![1.0f32; positions * HW];

    // get the clocks up before anything is recorded
    black_box(rollouts(200, seed, Sampler::Bits).games);

    let mut rows: Vec<Row> = Vec::new();
    for sweep in 0..sweeps {
        // a different seed per sweep, so no configuration is measured on one lucky run
        let s = seed + sweep as u64 * 7919;
        let measured = [
            ("uniform from bitboard", rollouts(games, s, Sampler::Bits)),
            ("weighted, hot dist", rollouts(games, s, Sampler::Hot(&dist[..HW]))),
            ("weighted, cold dist", rollouts(games, s, Sampler::Cold(&dist))),
        ];
        for (k, (label, sample)) in measured.into_iter().enumerate() {
            if sweep == 0 {
                rows.push(Row::new(label, sample));
            } else {
                rows[k].merge(sample);
            }
        }
    }

    println!(
        "\n{sweeps} interleaved sweeps of {games} games; each row is the best sweep, \
         `games` the total played."
    );
    header("Game: random rollouts, one after another");
    for r in &rows {
        r.print();
    }
    println!(
        "\n(a Game is {} bytes, so forking a position is a memcpy. The weighted rows walk a \
         {}-float\n distribution per move; the cold one draws it from {positions} positions' \
         worth, the hot one reuses\n a single vector. The bitboard sampler reads no \
         distribution at all and is {:.2}x faster\n than the cold weighted scan.)",
        std::mem::size_of::<Game>(),
        HW,
        rows[2].best.per_game() / rows[0].best.per_game(),
    );
}
