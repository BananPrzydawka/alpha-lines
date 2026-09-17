//! Search tunables, with defaults from the shared project config.json.
use crate::config as defaults;

/// Game-id slots per node. Compile-time because it sizes an array on every node.
pub const K: usize = defaults::MCTS_K;
const _: () = assert!(K > 0 && K <= u8::MAX as usize, "mcts.k must fit the node's u8 count");

/// Bits in `Node::flags`. Terminality is not a flag: a node owns its game, so it is
/// `game.finished`, and a stored copy could only drift from it.
pub const PENDING: u8 = 1 << 0;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Live game slots.
    pub g: usize,
    /// Evaluation buffer width: positions per model call.
    pub b: usize,
    /// Ready-to-step threshold.
    pub t: usize,
    /// Simulations per game per move.
    pub s: u32,
    /// Node stack size.
    pub node_capacity: usize,

    pub c_puct: f32,
    pub alpha: f32,
    pub epsilon: f32,
    pub exp3_gamma: f32,
}

impl Config {
    /// Arena budget with reserve above the grounded-model calibration peaks.
    /// This is not a worst-case bound; the factor is set in config.json.
    pub fn recommended_node_capacity(g: usize, s: u32) -> usize {
        g.checked_mul(s as usize)
            .and_then(|n| n.checked_mul(defaults::MCTS_NODE_CAPACITY_FACTOR))
            .expect("arena capacity overflow")
    }
}

impl Default for Config {
    fn default() -> Self {
        let g = defaults::MCTS_G;
        let s = defaults::MCTS_S;
        Config {
            g,
            b: defaults::MCTS_B,
            t: defaults::MCTS_T,
            s,
            node_capacity: Self::recommended_node_capacity(g, s),
            c_puct: defaults::MCTS_C_PUCT,
            alpha: defaults::MCTS_ALPHA,
            epsilon: defaults::MCTS_EPSILON,
            exp3_gamma: defaults::MCTS_EXP3_GAMMA,
        }
    }
}
