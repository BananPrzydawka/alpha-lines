//! One simulation: walk down from a game's root until the search needs something it does not
//! have, then stop and say what it is waiting for.
//!
//! Descending costs no game logic at all. A child is identified by stepping the parent's
//! Zobrist key with the move, and looked up by that key; only when the lookup misses does the
//! engine run, once, to build the position the new node owns. So a descent through `d`
//! existing nodes is `d` hash steps and `d` probes, and the engine is touched once per node
//! ever created rather than once per edge ever traversed.

use crate::game::{Game, Rng, Scratch};
use crate::mcts::arena::Arena;
use crate::mcts::config::{Config, PENDING};
use crate::mcts::slot::{Slot, ROOT};
use crate::mcts::variant::{Choice, Variant};
use crate::zobrist;

/// How a descent ended.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Descent {
    /// Reached a terminal, whose values are already backed up and counted. The game is free
    /// to descend again this cycle.
    NoEntry,
    /// The game now holds an unresolved path and is out of the cycle until the batch returns.
    /// `fresh` when this descent created the node, and so owes the buffer an entry; a pending
    /// hit rides on the entry another game already made.
    Entry { node: u32, fresh: bool },
    /// The node stack is full: nothing created, nothing backed up, no path left in flight.
    /// A budget reached, not a broken invariant — the caller counts it and moves on.
    Exhausted,
}

/// The game's result, if it has one.
#[inline]
fn outcome(g: &Game) -> Option<[f32; 2]> {
    g.finished.then(|| g.terminal_values())
}

/// The current position and its key, wherever it lives.
#[inline]
fn view<'a, S>(slot: &'a Slot<S>, arena: &'a Arena<S>, cur: u32) -> (u64, &'a Game) {
    if cur == ROOT {
        (slot.root.key, &slot.root.game)
    } else {
        let n = arena.node(cur);
        (n.key, &n.game)
    }
}

/// Fold `values` into every node on the path, then retire the simulation.
///
/// Deepest first, which is the spec's order; it makes no difference to the arithmetic, since
/// bits are only ever set and so no position can repeat on one path.
pub fn back_up<V: Variant>(
    slot: &mut Slot<V::Stats>,
    arena: &mut Arena<V::Stats>,
    values: [f32; 2],
    cfg: &Config,
) {
    for k in (0..slot.path.len()).rev() {
        let step = slot.path.step(k);
        if step.node == ROOT {
            let g = &slot.root.game;
            let counts = [g.legal_count(0), g.legal_count(1)];
            V::backup(&mut slot.root.stats, step.choice, values, counts, cfg);
        } else {
            let n = arena.node_mut(step.node);
            let counts = [n.game.legal_count(0), n.game.legal_count(1)];
            V::backup(&mut n.stats, step.choice, values, counts, cfg);
        }
    }
    slot.sim_count += 1;
    slot.path.clear();
    slot.waiting_on = None;
}

/// One selection step, for the trace hook: which node was selected on, and what
/// the selection chose there. Snapshots are taken by the caller before calling this.
#[derive(Clone, Copy)]
pub struct Selection {
    pub node: u32,
    pub choice: [Choice; 2],
}

/// Run one simulation from `slot`'s root. `id` is the slot's own index.
///
/// When `trace` is `Some`, it is called once per selection, after the choice is made but
/// before the child is looked up — so the caller can snapshot the stats `select` just saw.
pub fn descend<V: Variant>(
    id: u16,
    slot: &mut Slot<V::Stats>,
    arena: &mut Arena<V::Stats>,
    cfg: &Config,
    rng: &mut Rng,
    scratch: &mut Scratch,
) -> Descent {
    descend_traced::<V>(id, slot, arena, cfg, rng, scratch, None)
}

/// Traced variant of [`descend`]; see [`Selection`].
pub fn descend_traced<V: Variant>(
    id: u16,
    slot: &mut Slot<V::Stats>,
    arena: &mut Arena<V::Stats>,
    cfg: &Config,
    rng: &mut Rng,
    scratch: &mut Scratch,
    mut trace: Option<&mut dyn FnMut(Selection)>,
) -> Descent {
    debug_assert!(slot.occupied && !slot.game_over, "descent from a slot that is not playing");
    debug_assert!(slot.waiting_on.is_none(), "a game may hold only one path at a time");
    slot.path.clear();

    let mut cur = ROOT;
    loop {
        // A terminal needs no network: its values are the game's own, so back up and stop.
        if let Some(values) = outcome(view(slot, arena, cur).1) {
            slot.last_depth = slot.path.len() as u8;
            back_up::<V>(slot, arena, values, cfg);
            return Descent::NoEntry;
        }

        let (key, legal) = {
            let (key, g) = view(slot, arena, cur);
            (key, [g.legal_moves(0), g.legal_moves(1)])
        };
        let choice = if cur == ROOT {
            V::select(&mut slot.root.stats, legal, cfg, rng)
        } else {
            V::select(&mut arena.node_mut(cur).stats, legal, cfg, rng)
        };
        if let Some(ref mut t) = trace {
            t(Selection { node: cur, choice });
        }
        let (a0, a1) = (choice[0].action as usize, choice[1].action as usize);

        // the child's key without the child's board
        let child_key = zobrist::step(key, &view(slot, arena, cur).1.cells, a0, a1);
        slot.path.push(cur, choice);

        if let Some(child) = arena.get(child_key) {
            arena.touch(child, id);
            if arena.node(child).is_pending() {
                // already queued by someone else: ride their entry rather than duplicating
                // the work, and rather than re-rolling, which would let other games bend
                // this one's search
                slot.last_depth = slot.path.len() as u8;
                slot.waiting_on = Some(child);
                return Descent::Entry { node: child, fresh: false };
            }
            cur = child;
            continue;
        }

        // The one place the engine runs: build the position this new node will own.
        let mut game = view(slot, arena, cur).1.clone();
        game.action_step(a0, a1, scratch);
        // a terminal child is never pending, so later arrivals read its values instead of
        // queueing behind an evaluation that will never come
        let flags = if game.finished { 0 } else { PENDING };
        let Some(child) = arena.insert(child_key, game, flags, id) else {
            slot.last_depth = slot.path.len() as u8;
            slot.path.clear();
            return Descent::Exhausted;
        };

        if let Some(values) = outcome(&arena.node(child).game) {
            slot.last_depth = slot.path.len() as u8;
            back_up::<V>(slot, arena, values, cfg);
            return Descent::NoEntry;
        }
        slot.last_depth = slot.path.len() as u8;
        slot.waiting_on = Some(child);
        return Descent::Entry { node: child, fresh: true };
    }
}
