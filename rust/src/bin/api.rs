//! Per-call cost of every public function on the engine.
//!
//! `bench` answers "how fast is a game"; this answers "how fast is a call", which is the
//! question that matters to a search driving the engine one node at a time. Every row is one
//! call, averaged over `--reps` invocations on a mid-game position — not on the opening
//! position, where the fast paths are unrepresentative.
//!
//! `apply` is the exception and is timed differently, over whole rollouts divided by the
//! moves it took, because its cost changes as the board fills and no single position is a
//! fair sample of it. Its row is therefore the same number `bench` reports as ns/move.
//!
//! Usage: api [--reps R] [--positions P] [--warmup-moves M] [--seed S]

use std::hint::black_box;
use std::time::Instant;

use alpha_lines_game::game::{legal_cell, Rng, Scratch, HEIGHT, HW, WIDTH};
use alpha_lines_game::Game;

/// A uniformly random legal move for `player`, out of the mask the engine hands over.
///
/// The engine deliberately does not offer this — picking moves uniformly is a driver's
/// business, and the real one will be sampling a policy. It is three operations on the two
/// mask words: `popcount` for how many moves there are, one bounded draw for which, and a
/// select for where. `x &= x - 1` clears the lowest set bit, so doing it `k` times leaves the
/// k-th one at the bottom for `trailing_zeros` to read off.
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

struct Table {
    rows: Vec<(String, f64, &'static str)>,
}

impl Table {
    /// `secs` covers `calls` invocations.
    fn add(&mut self, name: &str, secs: f64, calls: usize, unit: &'static str) {
        self.rows.push((name.to_string(), secs / calls as f64, unit));
    }
    fn print(&self, title: &str) {
        println!("\n=== {title} ===");
        println!("{:<30} {:>12}   {}", "call", "per call", "what one call covers");
        for (name, per_call, unit) in &self.rows {
            println!("{:<30} {:>10.1}ns   {}", name, per_call * 1e9, unit);
        }
    }
}

/// A batch of independent mid-game positions to measure the pure accessors on.
///
/// One position would sit in L1 and every row would be a cache hit, which is not the
/// situation a search is in. `--positions` of them, walked in order, is closer: a tree search
/// touches a different node every time it descends.
fn positions(count: usize, warmup: usize, seed: u64, s: &mut Scratch) -> Vec<Game> {
    let mut rng = Rng::new(seed);
    (0..count)
        .map(|_| {
            let mut g = Game::new();
            for _ in 0..warmup {
                if g.finished {
                    break;
                }
                let (i0, i1) = (uniform_move(&g, 0, &mut rng), uniform_move(&g, 1, &mut rng));
                g.action_step(i0, i1, s);
            }
            g
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps = arg_usize(&args, "--reps", 2000);
    let count = arg_usize(&args, "--positions", 512);
    let warmup = arg_usize(&args, "--warmup-moves", 20);
    let seed = arg_usize(&args, "--seed", 12345) as u64;

    let mut scratch = Scratch::new();
    let mut rng = Rng::new(seed ^ 0xfeed);
    let mid = positions(count, warmup, seed, &mut scratch);
    let dist: Vec<f32> = (0..HW).map(|_| rng.random() as f32).collect();
    let calls = reps * count;

    let mut t = Table { rows: Vec::new() };

    let s = Instant::now();
    for _ in 0..reps {
        black_box(Game::new());
    }
    t.add("new", s.elapsed().as_secs_f64(), reps, "the opening position");

    let s = Instant::now();
    for _ in 0..reps {
        for g in &mid {
            black_box(g.clone());
        }
    }
    t.add("clone", s.elapsed().as_secs_f64(), calls, "fork a position for a child node");

    let s = Instant::now();
    for _ in 0..reps {
        for g in &mid {
            black_box(Game::from_cells(g.cells, &mut scratch));
        }
    }
    t.add("from_cells", s.elapsed().as_secs_f64(), calls, "adopt a board: BFS + full rescore");

    let s = Instant::now();
    for _ in 0..reps {
        for g in &mid {
            black_box(g.legal_moves(0));
        }
    }
    t.add("legal_moves", s.elapsed().as_secs_f64(), calls, "the 80-bit mask for one player");

    let s = Instant::now();
    for _ in 0..reps {
        for g in &mid {
            black_box(g.legal_count(0));
        }
    }
    t.add("legal_count", s.elapsed().as_secs_f64(), calls, "how many moves a player has");

    let s = Instant::now();
    for _ in 0..reps {
        for g in &mid {
            black_box(g.sample_move(&dist, 0, &mut rng));
        }
    }
    t.add("sample_move", s.elapsed().as_secs_f64(), calls, "weighted draw over 160 floats");

    // `apply` over whole rollouts: its cost is a function of how full the board is, so the
    // only honest average is over a game, not over one position replayed.
    let rollouts = reps.div_ceil(4).max(1);
    let mut moves = 0usize;
    let s = Instant::now();
    for _ in 0..rollouts {
        let mut g = Game::new();
        while !g.finished {
            let (i0, i1) = (uniform_move(&g, 0, &mut rng), uniform_move(&g, 1, &mut rng));
            g.action_step(i0, i1, &mut scratch);
            moves += 1;
        }
        black_box(&g.scores);
    }
    let played = s.elapsed().as_secs_f64();
    t.add("action_step (+ 2 picks)", played, moves, "one move, averaged over whole games");

    t.print(format!("Game, {count} mid-game positions, {reps} reps").as_str());

    println!(
        "\n(board {HEIGHT} x {WIDTH}; a Game is {} bytes, so the {count} positions above are \
         {} KB\n and do not fit in L1. Rollout: {moves} moves over {rollouts} games, \
         {:.1} moves/game.)",
        std::mem::size_of::<Game>(),
        count * std::mem::size_of::<Game>() / 1024,
        moves as f64 / rollouts as f64,
    );
}
