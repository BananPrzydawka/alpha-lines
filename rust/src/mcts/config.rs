//! Search tunables.

/// Game-id slots per node. Compile-time because it sizes an array on every node.
pub const K: usize = 64;
/// The later MCTS calibration used eight times G*S, roughly four times its
/// largest 50-step measured peak. The training runner may override this.
pub const NODE_CAPACITY_FACTOR: usize = 8;

/// Bits in `Node::flags`. Terminality is not a flag: a node owns its game, so it is
/// `game.finished`, and a stored copy could only drift from it.
pub const PENDING: u8 = 1 << 0;
/// This node's id list has been full at least once, so some game's claim on it was dropped.
pub const OVERFLOWED: u8 = 1 << 1;

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

impl Default for Config {
    fn default() -> Self {
        let g = 8192;
        let s = 100;
        Config {
            g,
            b: 2048,
            t: 2048,
            s,
            // Grounded-model calibration: see commit fbb2f56 and its 50-step report.
            node_capacity: g * s as usize * NODE_CAPACITY_FACTOR,
            c_puct: 1.5,
            alpha: 0.3,
            epsilon: 0.25,
            exp3_gamma: 0.1,
        }
    }
}
