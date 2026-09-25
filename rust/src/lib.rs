//! The alpha-lines game engine.
//!
//! [`Game`] is one board: plain data, cloned freely, with the caller owning the [`Scratch`]
//! workspace and lending it to the calls that need it.
//!
/// Defaults generated from the repository's config.json at build time.
pub mod config {
    include!(concat!(env!("OUT_DIR"), "/config.rs"));
}

pub mod game;
pub mod mcts;
pub mod zobrist;

pub use game::{Game, Scratch, HEIGHT, HW, ROW, SQUARES, WIDTH};


pub mod klent;
