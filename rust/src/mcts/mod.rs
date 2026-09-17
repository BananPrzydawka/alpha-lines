//! Batched MCTS over shared, hash-addressed node storage.
//!
//! There is no tree structure: a node is located only by Zobrist key, so identical states
//! reached by different move orders or different games are one node whose statistics
//! accumulate across all of them.

pub mod arena;
pub mod config;
pub mod descent;
pub mod noise;
pub mod search;
pub mod slot;
pub mod variant;

pub use arena::{Arena, Node};
pub use descent::{back_up, descend, Descent};
pub use slot::{Path, Root, Slot, MAX_PLY, ROOT};
pub use config::{Config, K, PENDING};
pub use search::{Evaluate, Search, StepRecord, Target};
pub use variant::{Choice, Exp3, Exp3Stats, Puct, PuctStats, Variant};
