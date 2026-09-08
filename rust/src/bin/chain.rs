//! PUCT trace for manual inspection, through the real batched search loop.
//!
//! G games share one arena; each cycle runs the true collection walk (descend in slot
//! order until B rows fill or slots run out), one model call (one-hot priors, constant
//! +0.5 values), scatter, and backup of every waiter. T is disabled: no step ever fires,
//! the whole trace is step 0. Runs until every game has completed `--sims` sims.
//!
//! Per cycle the file shows, in slot order, each game's descent with the same readable
//! per-node stat tables the old sequential tracer printed — snapshotted at selection
//! time, before the child lookup, so the numbers are what `select` saw — then one
//! `eval:` line naming the buffer rows the model call resolved.
//!
//! Config: one-hot priors, constant +0.5 values, epsilon/alpha/B from flags (B defaults
//! to G).
//!
//! Usage: chain [--sims N] [--eps E] [--alpha A] [--games G] [--buffer B] [--grounded]
//! Prints only the output file name; the traces go in the file.

use std::collections::HashMap;
use std::fmt::Write as _;

use alpha_lines_game::game::{Rng, SQUARES};
use alpha_lines_game::mcts::descent::Descent;
use alpha_lines_game::mcts::noise::dirichlet;
use alpha_lines_game::mcts::search::{Evaluate, Search};
use alpha_lines_game::mcts::slot::ROOT;
use alpha_lines_game::mcts::variant::{squares, Puct, PuctStats};
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

impl Evaluate for Stub {
    fn evaluate(&mut self, _positions: &[[u8; SQUARES]], priors: &mut [f32], values: &mut [f32]) {
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
                // concentrated priors: one Dirichlet(0.2) draw per row
                let mut draw = [0.0f32; SQUARES];
                for row in priors.chunks_mut(SQUARES) {
                    dirichlet(&mut self.rng, 0.2, &mut draw);
                    row.copy_from_slice(&draw);
                }
                // mostly positive values: Normal(0.15, 0.2), clipped into [-1, 1]
                for v in values.iter_mut() {
                    let x = 0.15 + 0.2 * normal(&mut self.rng);
                    *v = (x as f32).clamp(-1.0, 1.0);
                }
            }
        }
    }
}

/// Readable dump of one node's stats: per player, the legal squares ranked by
/// PUCT score, so the eye can check the argmax. Untouched zero-q squares collapse
/// into one "(N more)" line.
fn dump_node(
    out: &mut String,
    label: &str,
    stats: &PuctStats,
    legal0: [u64; 2],
    legal1: [u64; 2],
    choice: [u8; 2],
    c_puct: f32,
) {
    let _ = writeln!(out, "  node {label}:");
    for player in 0..2 {
        let legal = if player == 0 { legal0 } else { legal1 };
        let explore = c_puct * (1.0 + stats.total[player] as f32).sqrt();
        // (square, score); sort best first, ties by square
        let mut rows: Vec<(usize, f32)> = squares(legal)
            .map(|sq| {
                let score = stats.q[player][sq]
                    + explore * stats.prior[player][sq]
                        / (1.0 + stats.visit[player][sq] as f32);
                (sq, score)
            })
            .collect();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        let _ = writeln!(
            out,
            "    player {player}: total={} explore={explore:.4} chosen={} (marked *)",
            stats.total[player], choice[player],
        );
        let _ = writeln!(out, "      sq     q      prior   visit   score");
        let mut hidden = 0usize;
        for (sq, score) in rows {
            let q = stats.q[player][sq];
            let p = stats.prior[player][sq];
            let v = stats.visit[player][sq];
            // the eye cares about visited squares, the prior peak, and the choice;
            // untouched zero-q squares collapse into one line
            let interesting =
                sq as u8 == choice[player] || q != 0.0 || v != 0 || p >= 0.01;
            if !interesting {
                hidden += 1;
                continue;
            }
            let mark = if sq as u8 == choice[player] { "*" } else { " " };
            let _ = writeln!(out, "     {sq:>3}{mark}  {q:.4}  {p:.4}  {v:>5}  {score:.4}");
        }
        if hidden > 0 {
            let _ = writeln!(out, "      (... {hidden} more legal squares: q=0, visit=0, prior<0.01)");
        }
    }
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let sims: usize = arg(&a, "--sims").and_then(|s| s.parse().ok()).unwrap_or(100);
    let eps: f32 = arg(&a, "--eps").and_then(|s| s.parse().ok()).unwrap_or(0.4729);
    let alpha: f32 = arg(&a, "--alpha").and_then(|s| s.parse().ok()).unwrap_or(0.2236);
    let games: usize = arg(&a, "--games").and_then(|s| s.parse().ok()).unwrap_or(2);
    let buf: usize = arg(&a, "--buffer").and_then(|s| s.parse().ok()).unwrap_or(games);
    let grounded = a.iter().any(|x| x == "--grounded");
    let kind = if grounded { StubKind::Grounded } else { StubKind::OneHot };

    let cfg = Config {
        g: games,
        b: buf,
        t: usize::MAX,
        s: sims as u32,
        node_capacity: 4096,
        epsilon: eps,
        alpha,
        ..Config::default()
    };
    // Roots start pending, exactly as Search::new leaves them: cycle 0 can only queue
    // the roots (buffer underfill by construction), the stub evaluates them through
    // the same path as every other node, and children start in cycle 1. No hand-rolled
    // seeding — roots are treated like any other node.
    let mut s = Search::<Puct>::new(cfg, 0xA1FA);
    let mut model = Stub { rng: Rng::new(0xE1A4), kind };

    // arena slot -> short name (n1, n2, ...) in creation order
    let mut names: HashMap<u32, String> = HashMap::new();
    let mut counter = 0u32;

    let mut out = String::new();
    let stub_name = if grounded {
        "Dirichlet(0.2) priors, Normal(0.15,0.2)-clipped values"
    } else {
        "one-hot priors, values=+0.5"
    };
    let _ = writeln!(out, "chain: G={games} B={buf} PUCT eps={eps} alpha={alpha} c_puct={} {stub_name} sims={sims}/game", cfg.c_puct);
    let _ = writeln!(out, "batched: collect in slot order until B rows fill or slots run out, one eval, backups land together");
    let _ = writeln!(out, "stats snapshotted at selection time, before the child lookup — what `select` saw");
    let _ = writeln!(out, "path names: r = game root, nK = K-th arena node built (shared across games)\n");

    let mut cycle = 0usize;
    loop {
        if s.slots.iter().all(|sl| sl.sim_count >= sims as u32) {
            break;
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
        for c in &collected {
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
                    if fresh {
                        let nm = c.fresh.map(&name_of).unwrap_or_else(|| "(abandoned)".to_string());
                        let extra = if c.abandoned { " (abandoned: buffer full, node given back)" } else { "" };
                        let _ = writeln!(out, "cycle {cycle} sim game {}: {path}  (fresh -> {nm}, depth={}){extra}",
                            c.slot, slot.path.len());
                    } else {
                        let nm = c.duplicate_of.map(&name_of).unwrap_or_else(|| "?".to_string());
                        let _ = writeln!(out, "cycle {cycle} sim game {}: {path}  (dup-hit on {nm}, rides its eval, depth={})",
                            c.slot, slot.path.len());
                    }
                    for (k, (node, choice)) in c.selections.iter().enumerate() {
                        let ch = [choice[0].action, choice[1].action];
                        let key = (c.slot, *node);
                        let at = snap_cursor.entry(key).or_insert(0);
                        if let Some(&si) = snap_idx.get(&key).and_then(|v| v.get(*at)) {
                            *at += 1;
                            let sn = &snapshots[si];
                            let lb = labels.get(k).cloned().unwrap_or_else(|| "?".into());
                            dump_node(&mut out, &lb, &sn.stats, sn.legal[0], sn.legal[1], ch, cfg.c_puct);
                        }
                    }
                    if c.fresh.is_some() && !c.abandoned {
                        let nm = c.fresh.map(&name_of).unwrap_or_default();
                        let _ = writeln!(out, "  leaf {nm} fresh — no stats yet, owes this cycle's eval");
                    }
                    let _ = writeln!(out, "  (backup lands with the eval below; sim_count not yet incremented)\n");
                }
                Descent::NoEntry => {
                    let _ = writeln!(out, "cycle {cycle} sim game {}: terminal at depth {} (backed up on the spot, sim_count={})\n",
                        c.slot, slot.last_depth, slot.sim_count);
                }
                Descent::Exhausted => {
                    let _ = writeln!(out, "cycle {cycle} sim game {}: EXHAUSTED (node stack full)\n", c.slot);
                }
            }
        }
        // the model call + scatter + backups, with row labels
        let name_lookup = |node: u32| names.get(&node).cloned().unwrap_or_else(|| format!("?{node}"));
        let rows = s.evaluate_scatter_backup(&mut model, name_lookup);
        let _ = writeln!(out, "  eval cycle {cycle}: {} row(s)", rows.len());
        for (row, _, label) in &rows {
            let _ = writeln!(out, "    row {row}: {label}");
        }
        let counts: Vec<String> =
            s.slots.iter().enumerate().map(|(i, sl)| format!("g{i}={}", sl.sim_count)).collect();
        let _ = writeln!(out, "    sim_counts after backup: {}\n", counts.join(" "));
        cycle += 1;
        if cycle > 100_000 {
            let _ = writeln!(out, "STOP: 100k cycles without finishing {sims} sims/game — aborting");
            break;
        }
    }

    let stub_tag = if grounded { "_grounded" } else { "" };
    let fname = format!("target/tmp/chain_eps{eps}_al{alpha}_{sims}sims_g{games}_b{buf}{stub_tag}.txt");
    std::fs::create_dir_all("target/tmp").expect("mkdir target/tmp");
    std::fs::write(&fname, &out).expect("write trace file");
    println!("{fname}");
}
