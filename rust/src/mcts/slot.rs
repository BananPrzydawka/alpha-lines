//! Per-game state: the root it searches from, and the one in-flight simulation it holds.
//!
//! A root is a private copy of a position, never the shared node the arena holds for the same
//! state. That is forced by Dirichlet noise: noise applies to roots only, and writing it into
//! a shared node would push one game's exploration into every other game that reaches the
//! state. Two games sitting on the same root each keep their own.
//!
//! The path is the only place tree structure exists, and it lives here rather than in the
//! nodes because it belongs to one simulation, not to a position: backup is deferred until
//! the batched evaluation returns, so the route taken has to survive in the meantime.

use crate::game::{Game, SQUARES};
use crate::mcts::config::Config;
use crate::mcts::variant::Choice;

/// Deepest a descent can go. A joint move sets exactly two of the board's 160 bits and bits
/// are never cleared, so no game runs past `SQUARES` plies and no path is longer.
pub const MAX_PLY: usize = SQUARES;

/// Stands in for the game's own root in a path step, since a root has no arena slot.
pub const ROOT: u32 = u32::MAX;

/// The position a game is searching from, with its own statistics.
#[derive(Clone, Debug)]
pub struct Root<S> {
    pub key: u64,
    pub game: Game,
    pub stats: S,
    /// No priors yet: the game does not descend until the evaluation lands.
    pub pending: bool,
}

/// One node on the route of a simulation, and what was played there.
#[derive(Clone, Copy, Debug)]
pub struct PathStep {
    /// Arena slot, or [`ROOT`].
    pub node: u32,
    pub choice: [Choice; 2],
}

/// The route of the one in-flight simulation, root first.
#[derive(Clone, Debug)]
pub struct Path {
    steps: [PathStep; MAX_PLY],
    len: usize,
}

impl Default for Path {
    fn default() -> Self {
        let blank = PathStep { node: ROOT, choice: [Choice { action: 0, prob: 0.0 }; 2] };
        Path { steps: [blank; MAX_PLY], len: 0 }
    }
}

impl Path {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn clear(&mut self) {
        self.len = 0;
    }
    /// The `k`-th step from the root.
    pub fn step(&self, k: usize) -> PathStep {
        self.steps[k]
    }
    pub fn push(&mut self, node: u32, choice: [Choice; 2]) {
        assert!(self.len < MAX_PLY, "a descent went past {MAX_PLY} plies");
        self.steps[self.len] = PathStep { node, choice };
        self.len += 1;
    }
}

/// One game. The slot index is the game id the arena records on nodes.
#[derive(Clone, Debug)]
pub struct Slot<S> {
    pub occupied: bool,
    pub game_over: bool,
    pub root: Root<S>,
    /// Simulations completed at this root.
    pub sim_count: u32,
    pub path: Path,
    /// The node whose evaluation this game's backup is blocked on.
    pub waiting_on: Option<u32>,
}

impl<S: Default> Default for Slot<S> {
    fn default() -> Self {
        Slot {
            occupied: false,
            game_over: false,
            root: Root { key: 0, game: Game::new(), stats: S::default(), pending: true },
            sim_count: 0,
            path: Path::default(),
            waiting_on: None,
        }
    }
}

impl<S> Slot<S> {
    /// Has done its share of simulations and is waiting for a step.
    ///
    /// `==` holds exactly: one descent per cycle completes at most one simulation, and a
    /// ready game stops descending, so `sim_count` lands on `s` rather than past it.
    pub fn ready(&self, cfg: &Config) -> bool {
        self.occupied && !self.game_over && self.sim_count == cfg.s
    }

    /// Eligible for a descent this cycle: playing, not already waiting, still short of sims.
    pub fn collectable(&self, cfg: &Config) -> bool {
        self.occupied && !self.game_over && self.waiting_on.is_none() && self.sim_count < cfg.s
    }
}
