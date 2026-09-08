//! Where the depth goes, over exactly one move.
//!
//! Runs `S - 1` cycles so no step ever fires: every statistic is from the first move's 100
//! simulations, before any promotion or sweep can muddy it. Two sweep modes:
//!
//! - epsilon sweep (default): the knob that spreads a game across sibling children.
//! - `--grid`: PUCT epsilon x alpha, compact tables of eval-depth and node counts.
//!
//! The stub assigns a one-hot prior — 100% of the mass on one legal square per player — the
//! purest case for the zero-noise collapse question. `--powerlaw` restores the older peaked
//! stub. Values are a constant +0.5 unless `--random-values` is passed: a realistic trained
//! net agrees with its own priors more often than not, and constant-positive removes the
//! punishment flips that otherwise snap shared chains sideways. (Zero-sum is deliberately
//! broken: both players' `q` converge to +0.5.)
//!
//! `--alpha-sweep` fixes epsilon at 0.25 and sweeps the Dirichlet concentration instead.
//! `--trace` runs the zero-noise setting with a per-cycle line (fresh-node depths, root
//! argmax flock, children present) instead of the sweep.
//! `--solo N` runs N isolated single-game searches (G=1, B=1, T=1) instead of the population.
//!
//! Usage: depth [--exp3] [--powerlaw] [--random-values] [--alpha-sweep] [--grid] [--trace] [--solo N]

use alpha_lines_game::game::{Rng, SQUARES, PLAYABLE_SQUARE, ROW, WIDTH, LEGAL_WORDS};
use alpha_lines_game::mcts::search::{Evaluate, Search};
use alpha_lines_game::mcts::variant::{squares, Exp3, Exp3Stats, Puct, PuctStats, Variant};
use alpha_lines_game::mcts::search::Diagnostics;
use alpha_lines_game::mcts::Config;
use alpha_lines_game::zobrist;

struct Stub {
    rng: Rng,
    onehot: bool,
    positive: bool,
}

/// Is square `k` in `player`'s opening half? `c = 2j + (r & 1)`, left half is `c < WIDTH/2`.
fn in_half(k: usize, player: usize) -> bool {
    let (r, j) = (k / ROW, k % ROW);
    let c = 2 * j + (r & 1);
    if player == 0 { c < WIDTH / 2 } else { c >= WIDTH / 2 }
}

/// The one-hot spike: the fixed square if it is playable (and legal, at the opening),
/// else the first square that is.
fn spike_square(cells: &[u8; SQUARES], player: usize) -> usize {
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
    // No playable square in the half: fall back to any playable square. Unreachable while
    // the game is live, and the row is ignored when the buffer slot is unfilled anyway.
    (0..SQUARES).find(|&k| cells[k] == PLAYABLE_SQUARE).unwrap_or(0)
}

impl Evaluate for Stub {
    fn evaluate(&mut self, positions: &[[u8; SQUARES]], priors: &mut [f32], values: &mut [f32]) {
        if self.onehot {
            let width = priors.len() / (2 * SQUARES);
            let nrows = priors.len() / SQUARES;
            for idx in 0..nrows {
                let player = if idx >= width { 1 } else { 0 };
                let spike = spike_square(&positions[idx % width], player);
                let row = &mut priors[idx * SQUARES..(idx + 1) * SQUARES];
                for v in row.iter_mut() {
                    *v = 0.0;
                }
                row[spike] = 1.0;
            }
        } else {
            for row in priors.chunks_mut(SQUARES) {
                let o = self.rng.randint(SQUARES as u64) as usize;
                for j in 0..SQUARES {
                    row[(o + j) % SQUARES] = 1.0 / (1.0 + j as f32).powf(6.0);
                }
            }
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
}

/// Every bucket that holds anything, as a percentage.
fn spread(h: &[u64], total: u64) -> String {
    let last = h.iter().rposition(|&c| c > 0).unwrap_or(0);
    (0..=last)
        .map(|k| format!("{:.1}", 100.0 * h[k] as f64 / total.max(1) as f64))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Per-game root branching, averaged over a 64-slot sample: for each sampled slot, how many
/// distinct squares per player its root statistics touched, plus joint children.
///
/// `visited` is "was this square ever selected at this slot's root": `visit > 0` for PUCT,
/// `strategy_sum > 0` for EXP3. Kids counts arena nodes matching `step(root.key, x, y)` for
/// touched `x`, `y` — an upper bound on evaluated children, since pending (unevaluated)
/// nodes match too. All three are properties of the private per-slot roots, so at G=8192
/// identical openings the sample is 64 copies of the same number.
fn root_width<V: Variant>(s: &Search<V>, visited: impl Fn(&Search<V>, usize, usize, usize) -> bool) -> (f64, f64, f64) {
    let sample = 64.min(s.slots.len());
    let (mut a0, mut a1, mut kids) = (0.0, 0.0, 0.0);
    for i in 0..sample {
        let g = &s.slots[i].root.game;
        let key = s.slots[i].root.key;
        let p0: Vec<usize> = (0..SQUARES).filter(|&sq| visited(s, i, 0, sq)).collect();
        let p1: Vec<usize> = (0..SQUARES).filter(|&sq| visited(s, i, 1, sq)).collect();
        let mut k = 0;
        for &x in &p0 {
            for &y in &p1 {
                if s.arena.get(zobrist::step(key, &g.cells, x, y)).is_some() {
                    k += 1;
                }
            }
        }
        a0 += p0.len() as f64;
        a1 += p1.len() as f64;
        kids += k as f64;
    }
    (a0 / sample as f64, a1 / sample as f64, kids / sample as f64)
}

/// PUCT's argmax, mirroring `select` without the rng: lowest square wins ties.
fn argmax_puct(stats: &PuctStats, legal: [[u64; LEGAL_WORDS]; 2], c_puct: f32) -> [usize; 2] {
    let mut out = [0usize; 2];
    for player in 0..2 {
        let explore = c_puct * (1.0 + stats.total[player] as f32).sqrt();
        let mut best = f32::NEG_INFINITY;
        let mut act = 0usize;
        let mut first = true;
        for sq in squares(legal[player]) {
            let score = stats.q[player][sq]
                + explore * stats.prior[player][sq] / (1.0 + stats.visit[player][sq] as f32);
            if first || score > best {
                best = score;
                act = sq;
                first = false;
            }
        }
        out[player] = act;
    }
    out
}

/// EXP3's modal move: the max log weight, which at gamma = 0 is also the sampling peak.
fn argmax_exp3(stats: &Exp3Stats, legal: [[u64; LEGAL_WORDS]; 2], _c: f32) -> [usize; 2] {
    let mut out = [0usize; 2];
    for player in 0..2 {
        let mut best = f32::NEG_INFINITY;
        let mut act = 0usize;
        let mut first = true;
        for sq in squares(legal[player]) {
            let w = stats.log_w[player][sq];
            if first || w > best {
                best = w;
                act = sq;
                first = false;
            }
        }
        out[player] = act;
    }
    out
}

/// Joint children of slot 0's root present in the arena.
fn kid_count<V: Variant>(s: &Search<V>) -> usize {
    let r = &s.slots[0].root;
    let l0: Vec<usize> = squares(r.game.legal_moves(0)).collect();
    let l1: Vec<usize> = squares(r.game.legal_moves(1)).collect();
    let mut n = 0;
    for &x in &l0 {
        for &y in &l1 {
            if s.arena.get(zobrist::step(r.key, &r.game.cells, x, y)).is_some() {
                n += 1;
            }
        }
    }
    n
}

fn make_stub(rng_seed: u64, onehot: bool, positive: bool) -> Stub {
    Stub { rng: Rng::new(rng_seed), onehot, positive }
}

/// One line per cycle at zero noise: fresh-node depths, the root argmax flock, children
/// present. `zero_noise` zeroes the variant's exploration knob; `label` names the setting.
fn run_trace<V: Variant>(
    scfg: Config,
    onehot: bool,
    positive: bool,
    label: &str,
    argmax: fn(&V::Stats, [[u64; LEGAL_WORDS]; 2], f32) -> [usize; 2],
) {
    let mut s = Search::<V>::new(scfg, 0xA1FA);
    let mut model = make_stub(7, onehot, positive);
    println!("trace {label}, one-hot priors, {} values. {} slots, B={}.",
        if positive { "constant +0.5" } else { "random" }, scfg.g, scfg.b);
    println!("fresh/dup/term are this cycle's deltas; freshdepths lists depth:count over");
    println!("nodes this cycle built (depth_new delta); slot0 is the argmax joint move at");
    println!("slot 0's root; distinct counts distinct argmax pairs over slots 0..64;");
    println!("kids counts joint children of the opening present in the arena.\n");
    for c in 0..(scfg.s - 1) {
        let (bu0, dup0, tm0) = (s.diag.buffer_unique, s.diag.duplicate_hits, s.diag.terminal_hits);
        let dn0 = s.diag.depth_new.clone();
        s.cycle(&mut model);
        let d = &s.diag;
        let fresh = d.buffer_unique - bu0;
        let dup = d.duplicate_hits - dup0;
        let term = d.terminal_hits - tm0;
        let mut fdeps = Vec::new();
        for (depth, (&now, &was)) in d.depth_new.iter().zip(dn0.iter()).enumerate() {
            if now > was {
                fdeps.push(format!("{depth}:{}", now - was));
            }
        }
        let g0 = &s.slots[0].root.game;
        let legal0 = [g0.legal_moves(0), g0.legal_moves(1)];
        let a0 = argmax(&s.slots[0].root.stats, legal0, scfg.c_puct);
        let n = 64.min(s.slots.len());
        let mut seen = Vec::new();
        for i in 0..n {
            let g = &s.slots[i].root.game;
            let lg = [g.legal_moves(0), g.legal_moves(1)];
            let x = argmax(&s.slots[i].root.stats, lg, scfg.c_puct);
            if !seen.contains(&x) {
                seen.push(x);
            }
        }
        let kids = kid_count(&s);
        let fd = if fdeps.is_empty() { "-".to_string() } else { fdeps.join(",") };
        println!("c{c:3} fresh={fresh:5} dup={dup:5} term={term:4} desc={} arena={:6} kids={kids:4} \
            freshdepths=[{fd}] slot0=({},{}) distinct={}",
            d.descents, s.arena.len(), a0[0], a0[1], seen.len());
    }
}

/// Eval-depth and node count for one config: the two cells of the grid.
fn grid_cell<V: Variant>(scfg: Config, onehot: bool, positive: bool) -> (f64, u64, u64) {
    let mut s = Search::<V>::new(scfg, 0xA1FA);
    let mut model = make_stub(7, onehot, positive);
    for _ in 0..(scfg.s - 1) {
        s.cycle(&mut model);
    }
    let (m, _) = Diagnostics::mean_of(&s.diag.depth_new);
    (m, s.diag.buffer_unique, s.diag.descents)
}

fn run_sweep<V: Variant>(
    scfg: Config,
    onehot: bool,
    positive: bool,
    tag: &str,
    visited: impl Fn(&Search<V>, usize, usize, usize) -> bool,
) {
    let mut s = Search::<V>::new(scfg, 0xA1FA);
    let mut model = make_stub(7, onehot, positive);
    let mut per_cycle = Vec::new();
    let (mut pn, mut pd) = (0u64, 0u64);
    for c in 0..(scfg.s - 1) {
        s.cycle(&mut model);
        let (n, dp) = (s.diag.buffer_unique, s.diag.duplicate_hits);
        if [0u32, 1, 2, 5, 10, 25, 50, 98].contains(&c) {
            per_cycle.push((c, n - pn, dp - pd));
        }
        pn = n; pd = dp;
    }
    let d = &s.diag;
    let (w0, w1, kids) = root_width(&s, visited);
    println!("{tag}");
    if d.steps > 0 || d.games_finished > 0 {
        println!("  WARNING: a step fired mid-sweep (steps={}, finished={}) — stats mix two moves",
            d.steps, d.games_finished);
    }
    let (mn, cn) = Diagnostics::mean_of(&d.depth_new);
    let (mp, cp) = Diagnostics::mean_of(&d.depth_dup);
    println!("  descents {}  nodes {}", d.descents, s.arena.len());
    println!("  mean depth: all {:.2} | at eval {mn:.2} (over {cn}) | pending {mp:.2} (over {cp})",
        d.mean_depth());
    let line: Vec<String> = per_cycle.iter()
        .map(|(c, n, dp)| format!("c{c}:{}%", 100 * n / (n + dp).max(1)))
        .collect();
    println!("  share of descents that built a node, by cycle: {}", line.join(" "));
    println!("  root width: p0 {w0:.1} moves tried, p1 {w1:.1}, {kids:.1} joint children exist");
    println!("  (root width: distinct squares the slot's own root stats touched per player,");
    println!("   plus joint children present in the arena; pending nodes count too)");
    println!("  all      {}", spread(&d.depth_hist, d.descents));
    println!("  new node {}", spread(&d.depth_new, d.descents));
    println!("  pending  {}", spread(&d.depth_dup, d.descents));
    println!("  terminal {}\n", spread(&d.depth_terminal, d.descents));
}

/// N isolated single-game searches (G=1, B=1, T=1, S=100): each builds its own 99-node
/// chain with no population pressure. Reports eval-depth and fresh-node counts per game
/// plus the mean, so alpha's effect on solo branching is directly visible.
fn run_solo(onehot: bool, positive: bool, games: usize, epsilons: &[f32], alphas: &[f32]) {
    println!("solo: {games} isolated games (G=1,B=1,T=1,S=100), one move each.\n");
    for &eps in epsilons {
        for &al in alphas {
            let mut depths = Vec::new();
            let mut nodes = Vec::new();
            for g in 0..games {
                let scfg = Config {
                    g: 1, b: 1, t: 1, node_capacity: 4096,
                    epsilon: eps, alpha: al, ..Config::default()
                };
                let mut s = Search::<Puct>::new(scfg, 0xA1FA + g as u64);
                let mut model = make_stub(7 + g as u64, onehot, positive);
                for _ in 0..(scfg.s - 1) {
                    s.cycle(&mut model);
                }
                let (m, _) = Diagnostics::mean_of(&s.diag.depth_new);
                depths.push(m);
                nodes.push(s.diag.buffer_unique);
            }
            let md: f64 = depths.iter().sum::<f64>() / depths.len() as f64;
            let mn: f64 = nodes.iter().sum::<u64>() as f64 / nodes.len() as f64;
            println!("  eps={eps} alpha={al}: eval-depth mean {md:.2} (per-game {:?}), nodes/game mean {mn:.1}",
                depths.iter().map(|d| format!("{d:.1}")).collect::<Vec<_>>());
        }
    }
}

fn arg_usize(args: &[String], key: &str, default: usize) -> usize {
    args.iter().position(|a| a == key)
        .and_then(|i| args.get(i + 1)?.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let exp3 = a.iter().any(|x| x == "--exp3");
    let alpha_sweep = a.iter().any(|x| x == "--alpha-sweep");
    let grid = a.iter().any(|x| x == "--grid");
    let do_trace = a.iter().any(|x| x == "--trace");
    let solo = arg_usize(&a, "--solo", 0);
    if exp3 && (alpha_sweep || grid || solo > 0) {
        eprintln!("alpha/epsilon grid and solo runs are PUCT-only; --exp3 with them is meaningless");
        std::process::exit(1);
    }
    // One-hot is the standard now; --powerlaw restores the older peaked stub.
    let onehot = !a.iter().any(|x| x == "--powerlaw");
    let positive = !a.iter().any(|x| x == "--random-values");

    if do_trace {
        if exp3 {
            let mut scfg = Config::default();
            scfg.exp3_gamma = 0.0;
            run_trace::<Exp3>(scfg, onehot, positive, "gamma=0", argmax_exp3);
        } else {
            let mut scfg = Config::default();
            scfg.epsilon = 0.0;
            run_trace::<Puct>(scfg, onehot, positive, "epsilon=0", argmax_puct);
        }
        return;
    }

    if solo > 0 {
        run_solo(onehot, positive, solo, &[0.0, 0.25, 1.0], &[0.03, 0.1, 0.3, 1.0]);
        return;
    }

    if grid {
        let epsilons = [0.0f32, 0.05, 0.25, 0.5, 1.0];
        let alphas = [0.03f32, 0.1, 0.3, 1.0];
        let base = Config::default();
        println!("PUCT epsilon x alpha grid, one-hot priors, {} values. One move, no step.",
            if positive { "constant +0.5" } else { "random" });
        println!("each cell: eval-depth (over fresh nodes) / nodes built.\n");
        print!("{:>12}", "eps \\ alpha");
        for &al in &alphas {
            print!("{:>22}", format!("{al}"));
        }
        println!();
        for &eps in &epsilons {
            print!("{:>12}", format!("{eps}"));
            for &al in &alphas {
                let mut scfg = base;
                scfg.epsilon = eps;
                scfg.alpha = al;
                let (m, n, _) = grid_cell::<Puct>(scfg, onehot, positive);
                print!("{:>22}", format!("{m:.2} / {n}"));
            }
            println!();
        }
        return;
    }

    let base = Config::default();
    println!("one move only: {} cycles, no step fires. {} slots, B={}.",
        base.s - 1, base.g, base.b);
    println!("stub: {}, {} values.",
        if onehot { "one-hot (100% on one legal square per player)" } else { "power-law peak" },
        if positive { "constant +0.5" } else { "random" });
    println!("depth spread is % of descents at depth 0,1,2,... with no truncation.\n");

    if alpha_sweep {
        for &al in &[0.03f32, 0.1, 0.3, 1.0] {
            let mut scfg = base;
            scfg.alpha = al;
            let tag = format!("alpha = {al} (epsilon = 0.25)");
            run_sweep::<Puct>(scfg, onehot, positive, &tag,
                |s: &Search<Puct>, i: usize, p: usize, sq: usize| s.slots[i].root.stats.visit[p][sq] > 0);
        }
        return;
    }

    let knobs: &[f32] = if exp3 { &[0.0, 0.01, 0.05, 0.1, 0.25] } else { &[0.0, 0.05, 0.25, 0.5, 1.0] };
    let name = if exp3 { "gamma" } else { "epsilon" };

    for &k in knobs {
        let mut scfg = base;
        if exp3 { scfg.exp3_gamma = k } else { scfg.epsilon = k }
        let tag = format!("{name} = {k}");

        if exp3 {
            run_sweep::<Exp3>(scfg, onehot, positive, &tag,
                |s: &Search<Exp3>, i: usize, p: usize, sq: usize| s.slots[i].root.stats.strategy_sum[p][sq] > 0.0);
        } else {
            run_sweep::<Puct>(scfg, onehot, positive, &tag,
                |s: &Search<Puct>, i: usize, p: usize, sq: usize| s.slots[i].root.stats.visit[p][sq] > 0);
        }
    }
}
