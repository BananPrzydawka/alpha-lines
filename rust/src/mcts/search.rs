//! The search loop: one cycle is one model call.
//!
//! A cycle collects positions from as many games as it takes to fill the evaluation buffer,
//! runs the model once, writes the results back, backs up every game that was waiting, and
//! steps the games that have done their share of simulations. Games are never advanced
//! individually; they advance in a batch, which is what keeps the model call wide.

use std::collections::HashMap;
use std::time::Instant;

use crate::game::{Rng, Scratch, LEGAL_WORDS, SQUARES};
use crate::mcts::arena::Arena;
use crate::mcts::config::{Config, PENDING};
use crate::mcts::descent::{back_up, descend, Descent};
use crate::mcts::slot::Slot;
use crate::mcts::variant::{squares, Variant};
use crate::{zobrist, Game};

/// Depth buckets in the diagnostics; the last one is "this deep or deeper".
pub const DEPTH_BUCKETS: usize = 24;

/// The model. One call scores `positions`, both players at once.
///
/// `priors` is `2 * width * SQUARES` long and `values` is `2 * width`: player `p`'s row for
/// position `r` starts at `(p * width + r) * SQUARES`, matching the `(2B, 80)` the training
/// side produces. Rows past the ones filled this cycle are zero and their output is ignored.
pub trait Evaluate {
    fn evaluate(&mut self, positions: &[[u8; SQUARES]], priors: &mut [f32], values: &mut [f32]);
}

/// Where an evaluated row belongs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Target {
    /// A node in the arena, named by slot.
    Node(u32),
    /// A root state, named by key — several games may be waiting on the same one.
    Root(u64),
}

/// What one game produced when it stepped: the position it moved from, the search's policy
/// there, and the move drawn from it.
#[derive(Clone, Debug)]
pub struct StepRecord {
    pub id: u16,
    pub cells: [u8; SQUARES],
    pub policy: [[f32; SQUARES]; 2],
    pub played: [u8; 2],
    /// The game ended on this move, and `values` is its outcome.
    pub finished: bool,
    pub values: [f32; 2],
}

/// Counters the search keeps about itself (spec §9). Never read by the search.
#[derive(Clone, Debug, Default)]
pub struct Diagnostics {
    pub cycles: u64,
    pub steps: u64,
    pub descents: u64,
    /// Positions that earned a buffer entry.
    pub buffer_unique: u64,
    /// Rows the collection walk could not fill, summed over cycles.
    pub buffer_shortfall: u64,
    /// Cycles that ran the model on a part-empty buffer.
    pub cycles_short: u64,
    /// Which cycles those were, up to the first 64, plus the last one seen — together they
    /// separate a startup effect from one that recurs at every step.
    pub short_cycles: Vec<u64>,
    pub last_short_cycle: u64,
    /// Descents on short cycles that produced no buffer row because the position was already
    /// queued by another game. This is what makes a cycle come up short: the walk uses slots
    /// but they contribute nothing to the batch.
    pub short_duplicates: u64,
    /// Descents on short cycles in total, for the ratio.
    pub short_descents: u64,
    /// How many descents ended at each depth.
    pub depth_hist: Vec<u64>,
    /// Reach tests the sweep ran, and how many kept the id.
    pub reach_tested: u64,
    pub reach_kept: u64,
    /// For nodes whose id list was full at sweep time, how many ids survived the sweep.
    pub full_node_survivors: Vec<u64>,
    /// Last slot index the walk reached, summed over cycles, for the mean.
    pub walk_end_total: u64,
    /// Descents that landed on a position already queued by another game.
    pub duplicate_hits: u64,
    pub terminal_hits: u64,
    pub root_evals: u64,
    /// Descents abandoned because the node stack was full.
    pub exhausted: u64,
    pub max_descents_hit: u64,
    /// Highest slot index the collection walk reached.
    pub deepest_slot: usize,
    pub nodes_deleted: u64,
    /// Games played to the end, which is also the number of slots reseeded.
    pub games_finished: u64,
    /// How many nodes carry each `id_count`, as of the last sweep.
    pub id_histogram: Vec<u64>,
    pub max_id_count: u8,
    /// Path lengths summed over descents, for the mean.
    pub depth_total: u64,
    /// Wall time per phase, in nanoseconds.
    pub t_collect: u128,
    pub t_evaluate: u128,
    pub t_scatter: u128,
    pub t_backup: u128,
    pub t_step: u128,
}

impl Diagnostics {
    pub fn mean_depth(&self) -> f64 {
        self.depth_total as f64 / self.descents.max(1) as f64
    }
}

/// The evaluation buffer: a fixed `B` rows, of which the first `len` are filled.
struct Buffer {
    positions: Vec<[u8; SQUARES]>,
    targets: Vec<Target>,
    priors: Vec<f32>,
    values: Vec<f32>,
    width: usize,
    len: usize,
}

impl Buffer {
    fn new(width: usize) -> Self {
        Buffer {
            positions: vec![[0u8; SQUARES]; width],
            targets: Vec::with_capacity(width),
            priors: vec![0.0; 2 * width * SQUARES],
            values: vec![0.0; 2 * width],
            width,
            len: 0,
        }
    }
    fn clear(&mut self) {
        // rows past `len` stay zero, which is what the model is handed for the shortfall
        for p in self.positions[..self.len].iter_mut() {
            *p = [0u8; SQUARES];
        }
        self.targets.clear();
        self.len = 0;
    }
    fn is_full(&self) -> bool {
        self.len == self.width
    }
    fn push(&mut self, cells: [u8; SQUARES], target: Target) {
        debug_assert!(!self.is_full(), "pushed into a full buffer");
        self.positions[self.len] = cells;
        self.targets.push(target);
        self.len += 1;
    }
    fn prior_row(&self, row: usize, player: usize) -> &[f32] {
        let at = (player * self.width + row) * SQUARES;
        &self.priors[at..at + SQUARES]
    }
    fn value(&self, row: usize, player: usize) -> f32 {
        self.values[player * self.width + row]
    }
}

pub struct Search<V: Variant> {
    pub cfg: Config,
    pub arena: Arena<V::Stats>,
    pub slots: Vec<Slot<V::Stats>>,
    pub diag: Diagnostics,
    buffer: Buffer,
    /// Key of an unevaluated root to the games waiting on it, so one root state costs one
    /// buffer row however many games happen to be sitting on it (spec §3.4).
    pending_roots: HashMap<u64, Vec<u16>>,
    /// Values from this cycle's evaluation, by node. Only ever read in the same cycle it is
    /// written, because a game resolves its path before any sweep can move a slot.
    evaluated: HashMap<u32, [f32; 2]>,
    rng: Rng,
    scratch: Scratch,
}

impl<V: Variant> Search<V> {
    pub fn new(cfg: Config, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut scratch = Scratch::new();
        let arena = Arena::new(&cfg);
        let mut slots: Vec<Slot<V::Stats>> = (0..cfg.g).map(|_| Slot::default()).collect();
        for s in slots.iter_mut() {
            seed_slot::<V>(s, &cfg, &mut rng);
        }
        let _ = &mut scratch;
        Search {
            buffer: Buffer::new(cfg.b),
            pending_roots: HashMap::new(),
            evaluated: HashMap::new(),
            diag: Diagnostics {
                id_histogram: vec![0; crate::mcts::config::K + 1],
                full_node_survivors: vec![0; crate::mcts::config::K + 1],
                depth_hist: vec![0; DEPTH_BUCKETS],
                ..Default::default()
            },
            cfg,
            arena,
            slots,
            rng,
            scratch,
        }
    }

    /// Games that have done their share of simulations and are waiting to move.
    pub fn ready(&self) -> usize {
        self.slots.iter().filter(|s| s.ready(&self.cfg)).count()
    }

    /// One model call: collect, evaluate, scatter, back up, and step if enough games are
    /// ready. Returns the training records from the step, if one happened.
    pub fn cycle<E: Evaluate>(&mut self, model: &mut E) -> Vec<StepRecord> {
        let t = Instant::now();
        self.collect();
        self.diag.t_collect += t.elapsed().as_nanos();

        let t = Instant::now();
        model.evaluate(&self.buffer.positions, &mut self.buffer.priors, &mut self.buffer.values);
        self.diag.t_evaluate += t.elapsed().as_nanos();

        let t = Instant::now();
        self.scatter();
        self.diag.t_scatter += t.elapsed().as_nanos();

        let t = Instant::now();
        self.back_up_waiters();
        self.diag.t_backup += t.elapsed().as_nanos();
        self.diag.cycles += 1;

        if self.ready() >= self.cfg.t {
            let t = Instant::now();
            let out = self.step();
            self.diag.t_step += t.elapsed().as_nanos();
            out
        } else {
            Vec::new()
        }
    }

    /// Walk the slots from 0, descending each eligible game until it produces a buffer entry
    /// or runs out of descents, and stop when the buffer is full.
    fn collect(&mut self) {
        self.buffer.clear();
        self.pending_roots.clear();
        self.evaluated.clear();
        for s in self.slots.iter_mut() {
            s.descents = 0;
        }

        let (dup_at_start, descents_at_start) = (self.diag.duplicate_hits, self.diag.descents);
        let mut end = 0usize;
        for i in 0..self.slots.len() {
            if self.buffer.is_full() {
                break;
            }
            end = i;
            self.diag.deepest_slot = self.diag.deepest_slot.max(i);
            if !self.slots[i].collectable(&self.cfg) {
                continue;
            }

            // §7: an unevaluated root needs the model too, but only once per root state
            if self.slots[i].root.pending {
                let key = self.slots[i].root.key;
                if let Some(waiters) = self.pending_roots.get_mut(&key) {
                    waiters.push(i as u16);
                } else {
                    let cells = self.slots[i].root.game.cells;
                    self.buffer.push(cells, Target::Root(key));
                    self.pending_roots.insert(key, vec![i as u16]);
                    self.diag.root_evals += 1;
                    if self.buffer.is_full() {
                        continue;
                    }
                }
            }

            loop {
                let r = descend::<V>(
                    i as u16,
                    &mut self.slots[i],
                    &mut self.arena,
                    &self.cfg,
                    &mut self.rng,
                    &mut self.scratch,
                );
                self.diag.descents += 1;
                let d = self.slots[i].last_depth as usize;
                self.diag.depth_total += d as u64;
                self.diag.depth_hist[d.min(DEPTH_BUCKETS - 1)] += 1;
                self.slots[i].descents += 1;
                match r {
                    Descent::Entry { node, fresh } => {
                        if fresh {
                            if self.buffer.is_full() {
                                // no room to ask: give the node back rather than leave it
                                // pending with nobody queued to evaluate it
                                self.abandon(i, node);
                            } else {
                                let cells = self.arena.node(node).game.cells;
                                self.buffer.push(cells, Target::Node(node));
                                self.diag.buffer_unique += 1;
                            }
                        } else {
                            self.diag.duplicate_hits += 1;
                        }
                        break;
                    }
                    Descent::NoEntry => {
                        self.diag.terminal_hits += 1;
                        if self.slots[i].sim_count >= self.cfg.s {
                            break;
                        }
                        if self.slots[i].descents >= self.cfg.max_descents {
                            self.diag.max_descents_hit += 1;
                            break;
                        }
                    }
                    Descent::Exhausted => {
                        self.diag.exhausted += 1;
                        break;
                    }
                }
            }
        }
        let short = self.cfg.b - self.buffer.len;
        self.diag.buffer_shortfall += short as u64;
        if short > 0 {
            self.diag.short_duplicates += self.diag.duplicate_hits - dup_at_start;
            self.diag.short_descents += self.diag.descents - descents_at_start;
            self.diag.cycles_short += 1;
            if self.diag.short_cycles.len() < 64 {
                self.diag.short_cycles.push(self.diag.cycles);
            }
            self.diag.last_short_cycle = self.diag.cycles;
        }
        self.diag.walk_end_total += end as u64;
    }

    /// Undo a descent that produced a node the buffer has no room to ask about.
    fn abandon(&mut self, slot: usize, node: u32) {
        self.slots[slot].waiting_on = None;
        self.slots[slot].path.clear();
        if self.arena.drop_id(node, slot as u16) && self.arena.is_unused(node) {
            self.arena.remove(node);
        }
    }

    /// Write the model's output where it belongs and clear `pending`.
    fn scatter(&mut self) {
        for row in 0..self.buffer.len {
            match self.buffer.targets[row] {
                Target::Node(node) => {
                    let n = self.arena.node_mut(node);
                    n.flags &= !PENDING;
                    let legal = [n.game.legal_moves(0), n.game.legal_moves(1)];
                    for p in 0..2 {
                        let at = (p * self.buffer.width + row) * SQUARES;
                        let priors = &self.buffer.priors[at..at + SQUARES];
                        V::set_priors(&mut n.stats, priors, legal[p], p);
                    }
                    self.evaluated
                        .insert(node, [self.buffer.value(row, 0), self.buffer.value(row, 1)]);
                }
                Target::Root(key) => {
                    let waiters = self.pending_roots.remove(&key).unwrap_or_default();
                    for g in waiters {
                        let s = &mut self.slots[g as usize];
                        let legal = [s.root.game.legal_moves(0), s.root.game.legal_moves(1)];
                        for p in 0..2 {
                            V::set_priors(&mut s.root.stats, self.buffer.prior_row(row, p), legal[p], p);
                        }
                        // each waiter applies its own noise, so shared roots still diverge
                        V::make_root(&mut s.root.stats, legal, &self.cfg, &mut self.rng);
                        s.root.pending = false;
                    }
                }
            }
        }
    }

    /// Back up every game whose awaited node was evaluated this cycle.
    fn back_up_waiters(&mut self) {
        for i in 0..self.slots.len() {
            let Some(node) = self.slots[i].waiting_on else { continue };
            let Some(&values) = self.evaluated.get(&node) else {
                debug_assert!(false, "a game waited on a node nobody evaluated");
                continue;
            };
            back_up::<V>(&mut self.slots[i], &mut self.arena, values, &self.cfg);
        }
    }

    /// Advance every ready game by one move, then sweep and reseed. The order is mandatory:
    /// moves first, so the sweep knows every game's new root; sweep next, so a finished
    /// game's ids are gone before its slot is reused; seeding last.
    pub fn step(&mut self) -> Vec<StepRecord> {
        let g = self.slots.len();
        let mut stepping = vec![false; g];
        let mut finished = vec![false; g];
        let mut records = Vec::new();
        let mut policy = [[0.0f32; SQUARES]; 2];

        // 8.1, 8.2: emit the target, play the move, mark games that ended
        for i in 0..g {
            if !self.slots[i].ready(&self.cfg) {
                continue;
            }
            let legal = {
                let game = &self.slots[i].root.game;
                [game.legal_moves(0), game.legal_moves(1)]
            };
            V::target(&self.slots[i].root.stats, legal, &mut policy);
            let played = [
                draw(&policy[0], legal[0], &mut self.rng),
                draw(&policy[1], legal[1], &mut self.rng),
            ];
            let cells = self.slots[i].root.game.cells;
            let key = zobrist::step(self.slots[i].root.key, &cells, played[0], played[1]);

            let s = &mut self.slots[i];
            s.root.game.action_step(played[0], played[1], &mut self.scratch);
            s.root.key = key;
            s.sim_count = 0;
            s.path.clear();
            s.waiting_on = None;
            stepping[i] = true;

            let done = s.root.game.finished;
            let values = if done { s.root.game.terminal_values() } else { [0.0; 2] };
            if done {
                s.game_over = true;
                finished[i] = true;
            }
            records.push(StepRecord {
                id: i as u16,
                cells,
                policy,
                played: [played[0] as u8, played[1] as u8],
                finished: done,
                values,
            });
        }

        // 8.3: adopt the new root, inheriting the node for it if the search already built one
        for i in 0..g {
            if !stepping[i] || finished[i] {
                continue;
            }
            let key = self.slots[i].root.key;
            match self.arena.get(key) {
                Some(node) => {
                    self.slots[i].root.stats = self.arena.node(node).stats.clone();
                    self.slots[i].root.pending = false;
                }
                None => {
                    self.slots[i].root.stats = V::Stats::default();
                    self.slots[i].root.pending = true;
                }
            }
            let legal = {
                let game = &self.slots[i].root.game;
                [game.legal_moves(0), game.legal_moves(1)]
            };
            V::make_root(&mut self.slots[i].root.stats, legal, &self.cfg, &mut self.rng);
        }

        self.sweep(&stepping, &finished);

        // 8.5: reuse of a slot index is safe only because the sweep just removed that id
        for i in 0..g {
            if self.slots[i].game_over {
                seed_slot::<V>(&mut self.slots[i], &self.cfg, &mut self.rng);
                self.diag.games_finished += 1;
            }
        }

        self.diag.steps += 1;
        records
    }

    /// Drop each node's claim from games that can no longer reach it, and delete the nodes
    /// nobody claims. A node is only ever tested against the games in its own id list.
    fn sweep(&mut self, stepping: &[bool], finished: &[bool]) {
        // cumulative across sweeps, so it answers how much id pressure the search ever put
        // on a node rather than what happened to survive the most recent step
        let hist = &mut self.diag.id_histogram;
        let mut max_ids = 0u8;

        for slot in 0..self.arena.slot_count() as u32 {
            if self.arena.is_unused(slot) {
                continue; // a free slot, not a live node
            }
            let mut ids = [0u16; crate::mcts::config::K];
            let n = self.arena.node(slot);
            let count = n.ids().len();
            ids[..count].copy_from_slice(n.ids());
            // recorded before pruning: the point of the histogram is to say whether K is big
            // enough for the pressure the search actually puts on a node, and after the sweep
            // that pressure is already gone
            hist[count] += 1;
            max_ids = max_ids.max(count as u8);

            for &id in &ids[..count] {
                let i = id as usize;
                let stale = if finished[i] {
                    true
                } else if stepping[i] {
                    let keep =
                        self.arena.node(slot).game.reachable_from(&self.slots[i].root.game);
                    self.diag.reach_tested += 1;
                    self.diag.reach_kept += u64::from(keep);
                    !keep
                } else {
                    false // that game's root did not move, so everything it holds is still live
                };
                if stale {
                    self.arena.drop_id(slot, id);
                }
            }

            let survivors = if self.arena.is_unused(slot) { 0 } else { self.arena.node(slot).ids().len() };
            if count == crate::mcts::config::K {
                // how close a full node came to being deleted: 1 means it was on the edge and
                // more id slots would have kept it alive longer
                self.diag.full_node_survivors[survivors] += 1;
            }
            if survivors == 0 {
                self.arena.remove(slot);
                self.diag.nodes_deleted += 1;
            }
        }
        self.diag.max_id_count = self.diag.max_id_count.max(max_ids);
    }
}

/// Put a slot back to a fresh game at the opening, with an unevaluated root.
fn seed_slot<V: Variant>(s: &mut Slot<V::Stats>, cfg: &Config, rng: &mut Rng) {
    s.root.game = Game::new();
    s.root.key = zobrist::hash(&s.root.game.cells);
    s.root.stats = V::Stats::default();
    s.root.pending = true;
    s.occupied = true;
    s.game_over = false;
    s.sim_count = 0;
    s.descents = 0;
    s.path.clear();
    s.waiting_on = None;
    let legal = [s.root.game.legal_moves(0), s.root.game.legal_moves(1)];
    // with no priors yet this is pure noise, which is what §7 asks an unevaluated root to
    // select on — and what stops 8192 identical roots all choosing the same square
    V::make_root(&mut s.root.stats, legal, cfg, rng);
}

/// Sample a square from a policy, by inverse CDF over the legal squares.
fn draw(policy: &[f32; SQUARES], legal: [u64; LEGAL_WORDS], rng: &mut Rng) -> usize {
    let total: f32 = squares(legal).map(|sq| policy[sq]).sum();
    let threshold = rng.random() as f32 * total;
    let mut cum = 0.0f32;
    let mut last = 0usize;
    for sq in squares(legal) {
        cum += policy[sq];
        last = sq;
        if cum > threshold {
            return sq;
        }
    }
    last
}
