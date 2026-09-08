//! Population sweep: eval-depth ± std and nodes built, averaged over repetitions.
//!
//! PUCT: one table per game-slot count G (rows epsilon, columns alpha). EXP3: one table
//! (rows gamma, columns G). Every cell averages R = 4096 / G independent searches, each a
//! single MCTS step-0 run of 200 sims from fresh openings — no step ever fires (T is set
//! above any reachable ready count). B = G, node_capacity scales with G.
//!
//! Two stubs: `onehot` (100% prior on one legal square per player; values constant +0.5
//! unless --random-values) and `grounded` (Dirichlet(0.09) priors per row, values from
//! Normal(0.15, 0.2) clipped to [-1, 1]).
//!
//! The full sweep is ~1.35M searches; shard it: --shard K/N runs every Nth cell starting
//! at K, so N shells each do 1/N of the work. Progress lines go to stderr.
//!
//! Usage: sweep [--exp3] [--grounded] [--random-values] [--shard K/N]

use std::io::Write;
use std::sync::mpsc;
use std::thread;

use alpha_lines_game::game::{Rng, SQUARES, PLAYABLE_SQUARE, ROW, WIDTH};
use alpha_lines_game::mcts::noise::dirichlet;
use alpha_lines_game::mcts::search::{Diagnostics, Evaluate, Search};
use alpha_lines_game::mcts::variant::{Exp3, Puct};
use alpha_lines_game::mcts::Config;

const EPSILONS: [f32; 6] = [0.0, 0.05, 0.1057, 0.2236, 0.4729, 1.0];
const ALPHAS: [f32; 5] = [0.05, 0.1057, 0.2236, 0.4729, 1.0];
const GAMMAS: [f32; 5] = [0.05, 0.1057, 0.2236, 0.4729, 1.0];
const GS: [usize; 13] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
const SIMS: u32 = 200;

#[derive(Clone, Copy, PartialEq)]
enum StubKind {
    OneHot,
    Grounded,
}

struct Stub {
    rng: Rng,
    kind: StubKind,
    positive: bool,
}

fn in_half(k: usize, player: usize) -> bool {
    let (r, j) = (k / ROW, k % ROW);
    let c = 2 * j + (r & 1);
    if player == 0 { c < WIDTH / 2 } else { c >= WIDTH / 2 }
}

fn onehot_square(cells: &[u8; SQUARES], player: usize) -> usize {
    let opening = cells.iter().all(|&c| c == PLAYABLE_SQUARE);
    let pref = if player == 0 { 0 } else { 4 };
    if cells[pref] == PLAYABLE_SQUARE && (!opening || in_half(pref, player)) {
        return pref;
    }
    for k in 0..SQUARES {
        if cells[k] == PLAYABLE_SQUARE && (!opening || in_half(k, player)) {
            return k;
        }
    }
    (0..SQUARES).find(|&k| cells[k] == PLAYABLE_SQUARE).unwrap_or(0)
}

/// One standard normal, by Box-Muller.
fn normal(rng: &mut Rng) -> f64 {
    let u1 = rng.random().max(f64::MIN_POSITIVE);
    let u2 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

impl Evaluate for Stub {
    fn evaluate(&mut self, positions: &[[u8; SQUARES]], priors: &mut [f32], values: &mut [f32]) {
        match self.kind {
            StubKind::OneHot => {
                let width = priors.len() / (2 * SQUARES);
                let nrows = priors.len() / SQUARES;
                for idx in 0..nrows {
                    let player = if idx >= width { 1 } else { 0 };
                    let spike = onehot_square(&positions[idx % width], player);
                    let row = &mut priors[idx * SQUARES..(idx + 1) * SQUARES];
                    for v in row.iter_mut() {
                        *v = 0.0;
                    }
                    row[spike] = 1.0;
                }
                if self.positive {
                    for v in values.iter_mut() {
                        *v = 0.5;
                    }
                } else {
                    for v in values.iter_mut() {
                        *v = self.rng.random() as f32 * 2.0 - 1.0;
                    }
                }
            }
            StubKind::Grounded => {
                // Concentrated priors: one Dirichlet(0.09) draw per row, written straight
                // into the row so both players share the same shape (legal masking happens
                // at set_priors, per player).
                let mut draw = [0.0f32; SQUARES];
                for row in priors.chunks_mut(SQUARES) {
                    dirichlet(&mut self.rng, 0.09, &mut draw);
                    row.copy_from_slice(&draw);
                }
                // Mostly positive values: Normal(0.15, 0.2), clipped into [-1, 1].
                for v in values.iter_mut() {
                    let x = 0.15 + 0.2 * normal(&mut self.rng);
                    *v = (x as f32).clamp(-1.0, 1.0);
                }
            }
        }
    }
}

fn cfg_for(g: usize) -> Config {
    Config {
        g,
        b: g,
        t: usize::MAX,
        s: SIMS,
        max_descents: 4,
        node_capacity: (g * SIMS as usize * 5 / 8).max(4096),
        ..Config::default()
    }
}

/// One repetition: fresh search, 199 cycles (S-1), step-0 only. Returns eval-depth mean
/// over fresh nodes, fresh-node count, total descents, and whether a step fired anyway.
fn one_rep_puct(eps: f32, alpha: f32, g: usize, seed: u64, kind: StubKind, positive: bool)
    -> (f64, u64, u64, bool)
{
    let mut scfg = cfg_for(g);
    scfg.epsilon = eps;
    scfg.alpha = alpha;
    let mut s = Search::<Puct>::new(scfg, seed);
    let mut model = Stub { rng: Rng::new(seed ^ 0xE1A4), kind, positive };
    for _ in 0..(SIMS - 1) {
        s.cycle(&mut model);
    }
    let stepped = s.diag.steps > 0 || s.diag.games_finished > 0;
    let (m, _) = Diagnostics::mean_of(&s.diag.depth_new);
    (m, s.diag.buffer_unique, s.diag.descents, stepped)
}

fn one_rep_exp3(gamma: f32, g: usize, seed: u64, kind: StubKind, positive: bool)
    -> (f64, u64, u64, bool)
{
    let mut scfg = cfg_for(g);
    scfg.exp3_gamma = gamma;
    let mut s = Search::<Exp3>::new(scfg, seed);
    let mut model = Stub { rng: Rng::new(seed ^ 0xE1A4), kind, positive };
    for _ in 0..(SIMS - 1) {
        s.cycle(&mut model);
    }
    let stepped = s.diag.steps > 0 || s.diag.games_finished > 0;
    let (m, _) = Diagnostics::mean_of(&s.diag.depth_new);
    (m, s.diag.buffer_unique, s.diag.descents, stepped)
}

struct Cell {
    depth_mean: f64,
    depth_std: f64,
    nodes_mean: f64,
    desc_mean: f64,
    stepped: bool,
    reps: usize,
}

fn aggregate(depths: &[f64], nodes: &[u64], descs: &[u64], stepped: bool) -> Cell {
    let n = depths.len() as f64;
    let mean = depths.iter().sum::<f64>() / n;
    let var = depths.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n;
    Cell {
        depth_mean: mean,
        depth_std: var.sqrt(),
        nodes_mean: nodes.iter().sum::<u64>() as f64 / n,
        desc_mean: descs.iter().sum::<u64>() as f64 / n,
        stepped,
        reps: depths.len(),
    }
}

fn parse_shard(s: &str) -> (usize, usize) {
    let (k, n) = s.split_once('/').expect("--shard K/N");
    (k.parse().expect("numeric K"), n.parse().expect("numeric N"))
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let exp3 = a.iter().any(|x| x == "--exp3");
    let grounded = a.iter().any(|x| x == "--grounded");
    let positive = !a.iter().any(|x| x == "--random-values");
    let (shard_k, shard_n) = a.iter().position(|x| x == "--shard")
        .and_then(|i| a.get(i + 1).map(|s| parse_shard(s)))
        .unwrap_or((0, 1));
    let kind = if grounded { StubKind::Grounded } else { StubKind::OneHot };
    let stub_name = match (kind, positive) {
        (StubKind::OneHot, true) => "one-hot priors, constant +0.5 values",
        (StubKind::OneHot, false) => "one-hot priors, random values",
        (StubKind::Grounded, _) => "Dirichlet(0.09) priors, Normal(0.15,0.2)-clipped values",
    };

    // Job list: PUCT (g, eps_i, al_i) or EXP3 (g, gam_i). Shard k of n takes every nth
    // job from k. Seeds are job-derived, so a cell's numbers are identical however the
    // sweep was sharded.
    enum Job { Puct { g: usize, e: usize, a: usize }, Exp3 { g: usize, m: usize } }
    let mut jobs: Vec<Job> = Vec::new();
    if exp3 {
        for &g in &GS {
            for m in 0..GAMMAS.len() {
                jobs.push(Job::Exp3 { g, m });
            }
        }
    } else {
        for &g in &GS {
            for e in 0..EPSILONS.len() {
                for a_idx in 0..ALPHAS.len() {
                    jobs.push(Job::Puct { g, e, a: a_idx });
                }
            }
        }
    }
    let mine: Vec<(usize, &Job)> = jobs.iter().enumerate()
        .filter(|(i, _)| i % shard_n == shard_k)
        .collect();
    eprintln!("sweep: {} jobs total, shard {shard_k}/{shard_n} runs {} ({stub_name})",
        jobs.len(), mine.len());

    // Single shared job queue behind a mutex, 7 worker threads on 8 cores.
    let queue: std::sync::Arc<std::sync::Mutex<Vec<(usize, JobDesc)>>> =
        std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    struct JobDesc { g: usize, e: usize, a: usize, m: usize, is_exp3: bool }
    {
        let mut q = queue.lock().unwrap();
        for (i, job) in mine.iter() {
            match job {
                Job::Puct { g, e, a } => q.push((*i, JobDesc { g: *g, e: *e, a: *a, m: 0, is_exp3: false })),
                Job::Exp3 { g, m } => q.push((*i, JobDesc { g: *g, e: 0, a: 0, m: *m, is_exp3: true })),
            }
        }
    }
    let workers: usize = a.iter().position(|x| x == "--jobs")
        .and_then(|i| a.get(i + 1)?.parse().ok())
        .unwrap_or(7);
    eprintln!("sweep: {workers} worker threads");
    let (res_tx, res_rx) = mpsc::channel::<(usize, Cell)>();
    let mut handles = Vec::new();
    for _ in 0..workers {
        let queue = std::sync::Arc::clone(&queue);
        let res_tx = res_tx.clone();
        handles.push(thread::spawn(move || {
            loop {
                let item = queue.lock().unwrap().pop();
                let Some((idx, desc)) = item else { break };
                let reps = 4096 / desc.g;
                let mut depths = Vec::with_capacity(reps);
                let mut nodes = Vec::with_capacity(reps);
                let mut descs = Vec::with_capacity(reps);
                let mut stepped = false;
                for r in 0..reps {
                    // Job- and rep-derived seed: deterministic under any sharding.
                    let seed = 0x5EED_0000 ^ ((idx as u64) << 20) ^ (r as u64);
                    let (d, n, dc, st) = if desc.is_exp3 {
                        one_rep_exp3(GAMMAS[desc.m], desc.g, seed, kind, positive)
                    } else {
                        one_rep_puct(EPSILONS[desc.e], ALPHAS[desc.a], desc.g, seed, kind, positive)
                    };
                    depths.push(d);
                    nodes.push(n);
                    descs.push(dc);
                    stepped |= st;
                }
                res_tx.send((idx, aggregate(&depths, &nodes, &descs, stepped))).unwrap();
            }
        }));
    }
    drop(res_tx);

    let mut cells: Vec<Option<Cell>> = (0..jobs.len()).map(|_| None).collect();
    let mut done = 0;
    for (idx, cell) in res_rx {
        cells[idx] = Some(cell);
        done += 1;
        if done % 20 == 0 || done == mine.len() {
            eprintln!("  {done}/{} cells", mine.len());
        }
    }
    for h in handles {
        h.join().unwrap();
    }

    // Tables. Missing cells (other shards) print as `--`.
    let out = std::io::stdout();
    let mut f = std::io::BufWriter::new(out.lock());
    writeln!(f, "stub: {stub_name}. S={SIMS}, B=G, T=disabled (step-0 only), R=4096/G reps.").unwrap();
    writeln!(f, "cell: eval-depth mean ± std / mean nodes built (* = a repetition stepped).").unwrap();
    if exp3 {
        write!(f, "{:>12}", "gamma \\ G").unwrap();
        for &g in &GS {
            write!(f, "{:>30}", format!("{g}")).unwrap();
        }
        writeln!(f).unwrap();
        for m in 0..GAMMAS.len() {
            write!(f, "{:>12}", format!("{}", GAMMAS[m])).unwrap();
            for (gi, _) in GS.iter().enumerate() {
                let idx = gi * GAMMAS.len() + m;
                match &cells[idx] {
                    Some(c) => write!(f, "{:>30}",
                        format!("{:.2}±{:.2}/{:.0}{}", c.depth_mean, c.depth_std,
                            c.nodes_mean, if c.stepped { "*" } else { "" })).unwrap(),
                    None => write!(f, "{:>30}", "--").unwrap(),
                }
            }
            writeln!(f).unwrap();
        }
    } else {
        for (gi, &g) in GS.iter().enumerate() {
            let reps = 4096 / g;
            writeln!(f, "\nG={g} (R={reps} reps, B={g}):").unwrap();
            write!(f, "{:>12}", "eps \\ alpha").unwrap();
            for &al in &ALPHAS {
                write!(f, "{:>30}", format!("{al}")).unwrap();
            }
            writeln!(f).unwrap();
            for e in 0..EPSILONS.len() {
                write!(f, "{:>12}", format!("{}", EPSILONS[e])).unwrap();
                for a_idx in 0..ALPHAS.len() {
                    let idx = gi * EPSILONS.len() * ALPHAS.len() + e * ALPHAS.len() + a_idx;
                    match &cells[idx] {
                        Some(c) => write!(f, "{:>30}",
                            format!("{:.2}±{:.2}/{:.0}{}", c.depth_mean, c.depth_std,
                                c.nodes_mean, if c.stepped { "*" } else { "" })).unwrap(),
                        None => write!(f, "{:>30}", "--").unwrap(),
                    }
                }
                writeln!(f).unwrap();
            }
        }
    }
}
