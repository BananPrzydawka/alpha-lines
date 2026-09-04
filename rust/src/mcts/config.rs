//! Search tunables.

/// Game-id slots per node. Compile-time because it sizes an array on every node.
pub const K: usize = 16;

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
    /// Descents per game per cycle.
    pub max_descents: u32,
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
            max_descents: 4,
            // steady state trends to G * S / 2 (spec section 10); headroom on top
            node_capacity: g * s as usize / 2 * 5 / 4,
            c_puct: 1.5,
            alpha: 0.3,
            epsilon: 0.25,
            exp3_gamma: 0.1,
        }
    }
}

impl Config {
    /// Bytes the node stack will occupy at full capacity, for a node of `node_bytes`.
    pub fn arena_bytes(&self, node_bytes: usize) -> usize {
        self.node_capacity * node_bytes
    }
}
