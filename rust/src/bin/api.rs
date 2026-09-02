//! Per-call cost of every public call on the engine, averaged over `--reps` invocations on
//! mid-game positions.
//!
//! The two step calls are timed over whole rollouts divided by the moves it took, since
//! their cost changes as the board fills and no single position is a fair sample.
//!
//! The `zobrist` rows are what a search pays to look up a child before building one.
//!
//! Usage: api [--reps R] [--positions P] [--warmup-moves M] [--seed S]

use std::hint::black_box;
use std::time::Instant;

use alpha_lines_game::game::{Rng, Scratch, HEIGHT, SQUARES, WIDTH};
use alpha_lines_game::{zobrist, Game};

/// A uniformly random legal move for `player`, out of the engine's mask.
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
            return (wi << 6) + x.trailing_zeros() as usize;
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

/// Independent mid-game positions to measure the accessors on. One would sit in L1 and every
/// row would be a cache hit; a search touches a different node every time it descends.
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
    let dist: Vec<f32> = (0..SQUARES).map(|_| rng.random() as f32).collect();
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

    // over whole rollouts: cost depends on how full the board is
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
    t.add("action_step (+ 2 picks)", played, moves, "one move, over whole games");

    let mut moves = 0usize;
    let s = Instant::now();
    for _ in 0..rollouts {
        let mut g = Game::new();
        while !g.finished {
            g.distribution_step(&dist, &dist, &mut rng, &mut scratch);
            moves += 1;
        }
        black_box(&g.scores);
    }
    t.add("distribution_step", s.elapsed().as_secs_f64(), moves, "one move, over whole games");

    let s = Instant::now();
    for _ in 0..reps {
        for g in &mid {
            black_box(zobrist::hash(&g.cells));
        }
    }
    t.add("zobrist::hash", s.elapsed().as_secs_f64(), calls, "hash a board from scratch");

    // one ordinary move and one collision, since only the collision reads the board
    let s = Instant::now();
    for _ in 0..reps {
        for g in &mid {
            black_box(zobrist::step(black_box(0), &g.cells, black_box(9), black_box(30)));
        }
    }
    t.add("zobrist::step", s.elapsed().as_secs_f64(), calls, "hash a child, ordinary move");

    let s = Instant::now();
    for _ in 0..reps {
        for g in &mid {
            black_box(zobrist::step(black_box(0), &g.cells, black_box(30), black_box(30)));
        }
    }
    t.add("zobrist::step (collision)", s.elapsed().as_secs_f64(), calls, "hash a child, collision");

    t.print(format!("Game, {count} mid-game positions, {reps} reps").as_str());

    println!(
        "\n(board {HEIGHT} x {WIDTH}; a Game is {} bytes, so the {count} positions above are \
         {} KB and do not fit in L1.)",
        std::mem::size_of::<Game>(),
        count * std::mem::size_of::<Game>() / 1024,
    );
}
