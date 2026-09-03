//! Batched MCTS over shared, hash-addressed node storage.
//!
//! There is no tree structure: a node is located only by Zobrist key, so identical states
//! reached by different move orders or different games are one node whose statistics
//! accumulate across all of them.

pub mod arena;
pub mod config;

pub use arena::{Arena, Node};
pub use config::{Config, K, PENDING};
