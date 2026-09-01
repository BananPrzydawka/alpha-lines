//! Time per game: build a `Game`, play it to the end with random moves, repeat.
//!
//! Two ways of choosing the move are timed. `distribution_step` scans an HW-float
//! distribution, which is what a policy network hands you — over one distribution reused by
//! every game (L1) and over a fresh slice per game (not L1). A uniform pick over the mask,
//! done here in the driver, is the floor: what a move costs when choosing it is free.
//!
//! Construction is timed separately so the column means something.
//!
//! Configurations are measured interleaved and the best sweep is kept, because the clock
//! drifts by ~20% over a straight-through run and that lands on whichever ran last.
//!
//! Usage: bench [--games N] [--positions P] [--sweeps K] [--seed S]

use std::hint::black_box;
use std::time::Instant;

use alpha_lines_game::game::{legal_cell, Rng, Scratch, HW};
use alpha_lines_game::Game;

/// A uniformly random legal move for `player`, out of the engine's mask: `popcount` for how
/// many, one bounded draw for which, and a select for where. `x &= x - 1` clears the lowest
/// set bit, so doing it `k` times leaves the k-th at the bottom for `trailing_zeros`.
fn uniform_move(g: &Game, player: usize, rng: &mut Rng) -> usize {
    let w = g.legal_moves(player);
    let count = w[0].count_ones() + w[1].count_ones();
    assert!(count > 0, "no legal move for player {player}");
    let mut k = rng.randint(count as u64) as u32;
    for (wi, &word) in w.iter().enumerate() {
        let c = word.count_ones();
        if k < c {
            let mut x = word;
            for _ in 0..k {
                x &= x - 1;
            }
            return legal_cell((wi << 6) + x.trailing_zeros() as usize);
        }
        k -= c;
    }
    unreachable!("select past the end of the legal set")
}


fn arg_usize(args: &[String], key: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == key)
        .map(|i| args[i + 1].parse().expect("numeric"))
        .unwrap_or(default)
}

/// One sweep of one configuration.
#[derive(Clone, Copy)]
struct Sample {
    games: usize,
    moves: usize,
    /// construction plus playing to the end
    total: f64,
    /// the construction part, timed on its own
    construct: f64,
}

impl Sample {
    fn per_game(&self) -> f64 {
        self.total / self.games as f64
    }
}

/// The best sweep for one configuration, plus totals over all of them.
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

/// How a rollout picks its moves.
enum Sampler<'a> {
    /// Uniform over the mask, in the driver.
    Bits,
    /// `distribution_step` over `HW` weights that stay in cache.
    Hot(&'a [f32]),
    /// `distribution_step` over a fresh slice per game out of a large array.
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
            match sampler {
                Sampler::Bits => {
                    let (i0, i1) = (uniform_move(&g, 0, &mut rng), uniform_move(&g, 1, &mut rng));
                    g.action_step(i0, i1, &mut scratch);
                }
                Sampler::Hot(d) => g.distribution_step(d, d, &mut rng, &mut scratch),
                Sampler::Cold(d) => {
                    let s = &d[(k % slices) * HW..][..HW];
                    g.distribution_step(s, s, &mut rng, &mut scratch);
                }
            }
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

    // every square the same weight, so the weighted scan picks uniformly too — the same
    // games as the driver-side pick, reached by doing much more work
    let dist = vec![1.0f32; positions * HW];

    // warm up the clocks
    black_box(rollouts(200, seed, Sampler::Bits).games);

    let mut rows: Vec<Row> = Vec::new();
    for sweep in 0..sweeps {
        // a different seed per sweep
        let s = seed + sweep as u64 * 7919;
        let measured = [
            ("uniform over the mask", rollouts(games, s, Sampler::Bits)),
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
         worth, the hot one reuses\n a single vector. The uniform pick reads no \
         distribution at all and is {:.2}x faster\n than the cold weighted scan.)",
        std::mem::size_of::<Game>(),
        HW,
        rows[2].best.per_game() / rows[0].best.per_game(),
    );
}
