//! Search trace for manual inspection, through the real batched search loop.
//!
//! G games share one arena; each cycle runs the true collection walk (descend in slot
//! order until B rows fill or slots run out), one model call (one-hot priors, constant
//! +0.5 values), scatter, and backup of every waiter. T is disabled: no step ever fires,
//! the whole trace is step 0. Runs until every game has completed `--sims` sims.
//!
//! Per cycle the file shows, in slot order, each game's descent with readable per-node
//! stat tables — snapshotted at selection time, before the child lookup, so the numbers
//! are what `select` saw — then one `eval:` line naming the buffer rows the model call
//! resolved. PUCT tables show the q/prior/visit/score ranking; EXP3 tables show the
//! log-weights and the mixed strategy the game sampled from.
//!
//! Config: one-hot priors, constant +0.5 values, variant from `--exp3` (default PUCT),
//! epsilon/alpha/B from flags (B defaults to G).
//!
//! Usage: chain [--sims N] [--eps E] [--alpha A] [--games G] [--buffer B] [--t T]
//!   [--steps K] [--grounded] [--exp3]
//! Prints only the output file name; the traces go in the file.

use std::collections::HashMap;
use std::fmt::Write as _;

use alpha_lines_game::game::{
    Rng, HEIGHT, PLAYER_0_MARK, PLAYER_1_MARK, PLAYABLE_SQUARE, SQUARES, WIDTH,
};
use alpha_lines_game::mcts::descent::Descent;
use alpha_lines_game::mcts::search::{Evaluate, Search};
use alpha_lines_game::mcts::slot::ROOT;
use alpha_lines_game::mcts::variant::{squares, Choice, Exp3, Exp3Stats, Puct, PuctStats, Variant};
use alpha_lines_game::mcts::Config;
use alpha_lines_game::zobrist;

/// Which stub model to run.
#[derive(Clone, Copy, PartialEq)]
enum StubKind {
    OneHot,
    Grounded,
}

struct Stub {
    rng: Rng,
    kind: StubKind,
}

/// One standard normal, by Box-Muller.
fn normal(rng: &mut Rng) -> f64 {
    let u1 = rng.random().max(f64::MIN_POSITIVE);
    let u2 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Prior decay base: square s has weight R^s, so the field spans 80 squares without
/// underflowing and peaks visibly at the target corner.
const GRADIENT_R: f32 = 0.9;
/// Per-eval noise: lognormal jitter on priors (sigma in log space), Gaussian on
/// values. Keeps the gradient direction but stops the stub being a perfect oracle —
/// same position reads slightly differently every eval, like a real model.
const PRIOR_NOISE: f64 = 0.75;
const VALUE_NOISE: f32 = 0.3;

/// Gradient weight table plus prefix sums: GRAD[s] = R^s, TOP_K[k] = sum of the k
/// largest weights (R^0 + ... + R^(k-1)), the denominator of the coverage ratio.
struct Gradient {
    w: [f32; SQUARES],
    top_k: [f32; SQUARES + 1],
}

impl Gradient {
    fn build() -> Self {
        let mut w = [0.0f32; SQUARES];
        let mut acc = 1.0f32;
        for s in 0..SQUARES {
            w[s] = acc;
            acc *= GRADIENT_R;
        }
        let mut top_k = [0.0f32; SQUARES + 1];
        for k in 1..=SQUARES {
            top_k[k] = top_k[k - 1] + w[k - 1];
        }
        Gradient { w, top_k }
    }
}

/// Coverage ratio for one player: gradient mass on their marks over the best
/// achievable mass with that many marks. Empty board reads 0.5 (a draw).
fn coverage(cells: &[u8; SQUARES], mark: u8, g: &Gradient, flip: bool) -> f32 {
    let mut num = 0.0f32;
    let mut k = 0usize;
    for sq in 0..SQUARES {
        if cells[sq] == mark {
            let d = if flip { SQUARES - 1 - sq } else { sq };
            num += g.w[d];
            k += 1;
        }
    }
    if k == 0 {
        return 0.5;
    }
    num / g.top_k[k].max(1e-8)
}

impl Evaluate for Stub {
    fn evaluate(&mut self, positions: &[[u8; SQUARES]], priors: &mut [f32], values: &mut [f32]) {
        match self.kind {
            // One-hot is kept only for the old onehot trace files; the grounded stub
            // below is the realistic one. Roots go through evaluate like any node.
            StubKind::OneHot => {
                for row in priors.chunks_mut(SQUARES) {
                    for v in row.iter_mut() {
                        *v = 0.0;
                    }
                    row[0] = 1.0;
                }
                for v in values.iter_mut() {
                    *v = 0.5;
                }
            }
            StubKind::Grounded => {
                let g = Gradient::build();
                // player rows come in pairs: rows [2r] is P0's view of position r,
                // rows [2r+1] P1's. P0's corner is square 0, P1's square 79.
                // Noise is drawn fresh every eval (not hashed from the position): a
                // real model jitters between calls, and dup-hits sharing one eval row
                // stay consistent regardless.
                let npos = positions.len().min(priors.len() / SQUARES / 2);
                let (p0_half, p1_half) = priors.split_at_mut(npos * SQUARES);
                for (r, cells) in positions.iter().enumerate().take(npos) {
                    let p0_row = &mut p0_half[r * SQUARES..][..SQUARES];
                    let p1_row = &mut p1_half[r * SQUARES..][..SQUARES];
                    for sq in 0..SQUARES {
                        let j0 = (-0.5 * PRIOR_NOISE * PRIOR_NOISE
                            + PRIOR_NOISE * normal(&mut self.rng))
                        .exp() as f32;
                        let j1 = (-0.5 * PRIOR_NOISE * PRIOR_NOISE
                            + PRIOR_NOISE * normal(&mut self.rng))
                        .exp() as f32;
                        p0_row[sq] = g.w[sq] * j0;
                        p1_row[sq] = g.w[SQUARES - 1 - sq] * j1;
                    }
                    let n0 = VALUE_NOISE * normal(&mut self.rng) as f32;
                    let n1 = VALUE_NOISE * normal(&mut self.rng) as f32;
                    values[0 * npos + r] = (((coverage(cells, PLAYER_0_MARK, &g, false) - 0.5)
                        * 2.5)
                        .tanh()
                        + n0)
                        .clamp(-1.0, 1.0);
                    values[1 * npos + r] = (((coverage(cells, PLAYER_1_MARK, &g, true) - 0.5)
                        * 2.5)
                        .tanh()
                        + n1)
                        .clamp(-1.0, 1.0);
                }
            }
        }
    }
}

/// Board art in the `print_state` style, one neutral view per node: `00`/`11` for the
/// marks, `--` playable, `xx` removed, blank unplayable.
fn board_art(out: &mut String, indent: &str, cells: &[u8; SQUARES]) {
    let border = format!("{indent}+{}+", "-".repeat(WIDTH * 2));
    let _ = writeln!(out, "{border}");
    for r in 0..HEIGHT {
        let mut line = String::from("|");
        for c in 0..WIDTH {
            if (r + c) % 2 != 0 {
                line.push_str("  ");
                continue;
            }
            // square index of board cell (r, c): row r holds WIDTH/2 playable cells
            let sq = r * (WIDTH / 2) + (c - (r & 1)) / 2;
            let v = cells[sq];
            line.push_str(if v == PLAYER_0_MARK {
                "00"
            } else if v == PLAYER_1_MARK {
                "11"
            } else if v == PLAYABLE_SQUARE {
                "--"
            } else {
                "xx"
            });
        }
        line.push('|');
        let _ = writeln!(out, "{indent}{line}");
    }
    let _ = writeln!(out, "{border}");
}

/// Readable dump of one node's stats: the board once, then per player the top-10
/// squares by prior, with q/visit/score beside them so the eye can check the argmax.
/// The choice is marked with `*` and named in the player header; `depth` indents the
/// node one level deeper per ply.
fn dump_node(
    out: &mut String,
    label: &str,
    stats: &PuctStats,
    cells: &[u8; SQUARES],
    legal0: [u64; 2],
    legal1: [u64; 2],
    choice: [u8; 2],
    c_puct: f32,
    depth: usize,
) {
    let pad = "    ".repeat(2 + depth);
    let inner = "    ".repeat(3 + depth);
    let _ = writeln!(out, "{pad}node {label}:");
    board_art(out, &inner, cells);
    for player in 0..2 {
        let legal = if player == 0 { legal0 } else { legal1 };
        let explore = c_puct as f64 * (1.0 + stats.total[player] as f64).sqrt();
        let _ = writeln!(out, "{inner}player {player} picked {}:", choice[player]);
        // (square, prior, score); top 10 by prior, ties by square
        let mut rows: Vec<(usize, f32, f64)> = squares(legal)
            .map(|sq| {
                let score = stats.q[player][sq]
                    + explore * stats.prior[player][sq] as f64
                        / (1.0 + stats.visit[player][sq] as f64);
                (sq, stats.prior[player][sq], score)
            })
            .collect();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        let _ = writeln!(out, "{inner}sq     q      prior   visit   score");
        for (sq, p, score) in rows.into_iter().take(10) {
            let q = stats.q[player][sq];
            let v = stats.visit[player][sq];
            let mark = if sq as u8 == choice[player] { "*" } else { " " };
            let _ = writeln!(out, "{inner}{sq:>3}{mark}  {q:.4}  {p:.4}  {v:>5}  {score:.4}");
        }
    }
}

/// Readable dump of one EXP3 node's stats: the board once, then per player the top-10
/// squares by mixed-strategy probability, with the log-weight beside them. The sampled
/// choice is marked with `*` and named in the player header; the `prob` column is what
/// the importance weight in backup uses. `depth` indents the node one level per ply.
fn dump_node_exp3(
    out: &mut String,
    label: &str,
    stats: &Exp3Stats,
    cells: &[u8; SQUARES],
    legal0: [u64; 2],
    legal1: [u64; 2],
    choice: [Choice; 2],
    gamma: f32,
    depth: usize,
) {
    let pad = "    ".repeat(2 + depth);
    let inner = "    ".repeat(3 + depth);
    let _ = writeln!(out, "{pad}node {label}:");
    board_art(out, &inner, cells);
    let mut probs = [0.0f64; SQUARES];
    for player in 0..2 {
        let legal = if player == 0 { legal0 } else { legal1 };
        Exp3::mixed(stats, legal, player, gamma, &mut probs);
        // (square, prob); top 10 by prob, ties by square
        let mut rows: Vec<(usize, f64)> =
            squares(legal).map(|sq| (sq, probs[sq])).collect();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        let _ = writeln!(out, "{inner}player {player} picked {}:", choice[player].action);
        let _ = writeln!(out, "{inner}sq     log_w    prob");
        for (sq, prob) in rows.into_iter().take(10) {
            let w = stats.log_w[player][sq];
            let mark = if choice[player].action as usize == sq { "*" } else { " " };
            let _ = writeln!(out, "{inner}{sq:>3}{mark}  {w:.4}  {prob:.4}");
        }
    }
}

/// The per-cycle body, generic over the variant. `dump` renders one snapshot's stats;
/// `tag` names the variant in the header.
fn run<V: Variant>(
    cfg: Config,
    games: usize,
    buf: usize,
    sims: usize,
    steps: usize,
    tag: &str,
    stub_name: &str,
    fname_mid: &str,
    model: &mut Stub,
    dump: impl Fn(&mut String, &str, &V::Stats, &[u8; SQUARES], [u64; 2], [u64; 2], [Choice; 2], &Config, usize),
) {
    // Roots start pending, exactly as Search::new leaves them: cycle 0 can only queue
    // the roots (buffer underfill by construction), the stub evaluates them through
    // the same path as every other node, and children start in cycle 1. No hand-rolled
    // seeding — roots are treated like any other node.
    let mut s = Search::<V>::new(cfg, 0xA1FA);

    // arena slot -> short name (n1, n2, ...) in creation order
    let mut names: HashMap<u32, String> = HashMap::new();
    let mut counter = 0u32;

    let mut out = String::new();
    let _ = writeln!(out, "chain: G={games} B={buf} {tag} {stub_name} sims={sims}/game\n");

    let mut cycle = 0usize;
    let mut step_done = 0usize;
    loop {
        if s.slots.iter().all(|sl| sl.sim_count >= sims as u32) {
            if step_done + 1 >= steps {
                break;
            }
            // like Search::cycle: step when enough games are ready, then keep tracing
            let before = s.arena.len();
            let records = s.step();
            step_done += 1;
            let _ = writeln!(out, "step {step_done}: arena nodes before={before} after={} (sweep dropped {})",
                s.arena.len(), before.saturating_sub(s.arena.len()));
            if records.is_empty() {
                let _ = writeln!(out, "    (no games ready — trace ends here)");
                break;
            }
            for r in &records {
                let _ = writeln!(out, "    game {}: played ({},{}) finished={} values={:?}",
                    r.id, r.played[0], r.played[1], r.finished, r.values);
            }
            let _ = writeln!(out, "");
            continue;
        }
        let (collected, snapshots) = s.collect_traced();
        // index snapshots by (slot, node) in selection order
        let mut snap_idx: HashMap<(usize, u32), Vec<usize>> = HashMap::new();
        for (k, sn) in snapshots.iter().enumerate() {
            snap_idx.entry((sn.slot, sn.node)).or_default().push(k);
        }
        let mut snap_cursor: HashMap<(usize, u32), usize> = HashMap::new();
        // name fresh nodes in slot order, so nK matches creation order
        for c in &collected {
            if let Some(node) = c.fresh {
                if !names.contains_key(&node) {
                    counter += 1;
                    names.insert(node, format!("n{counter}"));
                }
            }
        }
        let name_of = |node: u32| -> String {
            if node == ROOT {
                "r".to_string()
            } else {
                names.get(&node).cloned().unwrap_or_else(|| format!("?{node}"))
            }
        };
        let _ = writeln!(out, "cycle {cycle}:");
        for c in &collected {
            if c.queued_root {
                let _ = writeln!(out, "    game {}:", c.slot);
                if c.queued_root_shared {
                    let _ = writeln!(out, "        node r already in buffer");
                } else {
                    let _ = writeln!(out, "        queued node r for eval");
                }
                continue;
            }
            // rebuild the named path from the slot's stored path (still in flight)
            let slot = &s.slots[c.slot];
            let mut path = String::from("r");
            let mut labels = vec![String::from("r")];
            let mut cur_key = slot.root.key;
            let mut cur_cells = slot.root.game.cells;
            for k in 0..slot.path.len() {
                let step = slot.path.step(k);
                let (a0, a1) =
                    (step.choice[0].action as usize, step.choice[1].action as usize);
                let child_key = zobrist::step(cur_key, &cur_cells, a0, a1);
                match s.arena.get(child_key) {
                    Some(child) => {
                        let nm = name_of(child);
                        path.push_str(&format!(" -({a0},{a1})-> {nm}"));
                        labels.push(nm);
                        if Some(child) != c.fresh {
                            let n = s.arena.node(child);
                            cur_key = n.key;
                            cur_cells = n.game.cells;
                        }
                    }
                    None => {
                        // abandoned node: removed already, name by move only
                        path.push_str(&format!(" -({a0},{a1})-> (abandoned)"));
                        labels.push("(abandoned)".to_string());
                        break;
                    }
                }
            }
            match c.outcome {
                Descent::Entry { fresh, .. } => {
                    let _ = writeln!(out, "    game {slot}:", slot = c.slot);
                    for (k, (node, choice)) in c.selections.iter().enumerate() {
                        let key = (c.slot, *node);
                        let at = snap_cursor.entry(key).or_insert(0);
                        if let Some(&si) = snap_idx.get(&key).and_then(|v| v.get(*at)) {
                            *at += 1;
                            let sn = &snapshots[si];
                            let lb = labels.get(k).cloned().unwrap_or_else(|| "?".into());
                            dump(&mut out, &lb, &sn.stats, &sn.cells, sn.legal[0], sn.legal[1], *choice, &cfg, k);
                        }
                    }
                    let _ = writeln!(out, "");
                    if fresh {
                        if c.abandoned {
                            let _ = writeln!(out, "        (abandoned: buffer full, node given back)");
                        } else {
                            let nm = c.fresh.map(&name_of).unwrap_or_else(|| "(abandoned)".to_string());
                            let _ = writeln!(out, "        queued node {nm} for eval");
                        }
                    } else {
                        let nm = c.duplicate_of.map(&name_of).unwrap_or_else(|| "?".to_string());
                        let _ = writeln!(out, "        node {nm} already in buffer");
                    }
                    let _ = writeln!(out, "");
                }
                Descent::NoEntry => {
                    let _ = writeln!(out, "    game {}: terminal at depth {} (backed up on the spot)",
                        c.slot, c.selections.len());
                }
                Descent::Exhausted => {
                    let _ = writeln!(out, "    game {}: EXHAUSTED (node stack full)", c.slot);
                }
            }
        }
        // the model call + scatter + backups, with row labels and returned values
        let name_lookup = |node: u32| names.get(&node).cloned().unwrap_or_else(|| format!("?{node}"));
        let rows = s.evaluate_scatter_backup(model, name_lookup);
        let _ = writeln!(out, "    eval: {} of {} rows", rows.len(), buf);
        for (row, _, label, values) in &rows {
            let _ = writeln!(out, "        row {row}: {label} values=[{:.4}, {:.4}]", values[0], values[1]);
        }
        let _ = writeln!(out, "");
        cycle += 1;
        if cycle > 100_000 {
            let _ = writeln!(out, "STOP: 100k cycles without finishing {sims} sims/game — aborting");
            break;
        }
    }

    let fname = format!("target/tmp/chain_{fname_mid}_{sims}sims_g{games}_b{buf}.txt");
    std::fs::create_dir_all("target/tmp").expect("mkdir target/tmp");
    std::fs::write(&fname, &out).expect("write trace file");
    println!("{fname}");
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let sims: usize = arg(&a, "--sims").and_then(|s| s.parse().ok()).unwrap_or(100);
    let eps: f32 = arg(&a, "--eps").and_then(|s| s.parse().ok()).unwrap_or(0.25);
    let alpha: f32 = arg(&a, "--alpha").and_then(|s| s.parse().ok()).unwrap_or(0.3);
    let gamma: f32 = arg(&a, "--gamma").and_then(|s| s.parse().ok()).unwrap_or(0.1);
    let games: usize = arg(&a, "--games").and_then(|s| s.parse().ok()).unwrap_or(1);
    let buf: usize = arg(&a, "--buffer").and_then(|s| s.parse().ok()).unwrap_or(games);
    let thresh: usize = arg(&a, "--t").and_then(|s| s.parse().ok()).unwrap_or(games);
    let steps: usize = arg(&a, "--steps").and_then(|s| s.parse().ok()).unwrap_or(1);
    let grounded = a.iter().any(|x| x == "--grounded");
    let exp3 = a.iter().any(|x| x == "--exp3");
    let kind = if grounded { StubKind::Grounded } else { StubKind::OneHot };

    let mut cfg = Config {
        g: games,
        b: buf,
        t: thresh,
        s: sims as u32,
        node_capacity: 4096,
        epsilon: eps,
        alpha,
        exp3_gamma: gamma,
        ..Config::default()
    };
    // one-hot priors put 100% on square 0, which EXP3's log() turns into -inf
    // everywhere else; grounded keeps every square samplable
    if exp3 && !grounded {
        cfg.epsilon = 1.0;
    }
    let stub_name = if grounded {
        "exp-gradient priors (r=0.9), tanh coverage values"
    } else {
        "one-hot priors, values=+0.5"
    };
    let mut model = Stub { rng: Rng::new(0xE1A4), kind };
    if exp3 {
        let tag = format!("EXP3 gamma={gamma}");
        let mid = format!("exp3_g{gamma}{}", if grounded { "_grounded" } else { "" });
        run::<Exp3>(cfg, games, buf, sims, steps, &tag, stub_name, &mid, &mut model,
            |o, lb, st, cells, l0, l1, ch, c, d| dump_node_exp3(o, lb, st, cells, l0, l1, ch, c.exp3_gamma, d));
    } else {
        let tag = format!("PUCT eps={eps} alpha={alpha} c_puct={}", cfg.c_puct);
        let mid = format!("eps{eps}_al{alpha}{}", if grounded { "_grounded" } else { "" });
        run::<Puct>(cfg, games, buf, sims, steps, &tag, stub_name, &mid, &mut model,
            |o, lb, st, cells, l0, l1, ch, c, d| {
                dump_node(o, lb, st, cells, l0, l1, [ch[0].action, ch[1].action], c.c_puct, d)
            });
    }
}
