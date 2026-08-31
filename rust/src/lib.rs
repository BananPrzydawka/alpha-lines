//! Reference Rust port of the alpha-lines batched game.
//!
//! A straight translation of `main/game.py` and `main/game_kernels.py` with no algorithmic
//! changes, intended as the baseline that future optimized implementations are measured
//! against for both correctness and speed.

pub mod config;
pub mod game;
pub mod game_kernels;
pub mod incremental;
pub mod rng;

pub use config::{BOARD_SIZE, HEIGHT, WIDTH};
pub use game::BatchedLinesGame;
pub use incremental::IncrementalGame;
