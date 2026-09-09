//! Search tunables.

/// Game-id slots per node. Compile-time because it sizes an array on every node.
pub const K: usize = 64;

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

impl Config {
    /// Arena budget with reserve above the grounded-model calibration peaks.
    /// See `rust/benchmarks/arena-calibration-50.md`; this is not a worst-case bound.
    pub fn recommended_node_capacity(g: usize, s: u32) -> usize {
        g.checked_mul(s as usize)
            .and_then(|n| n.checked_mul(8))
            .expect("arena capacity overflow")
    }
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
            node_capacity: Self::recommended_node_capacity(g, s),
            c_puct: 1.5,
            alpha: 0.3,
            epsilon: 0.25,
            exp3_gamma: 0.1,
        }
    }
}

