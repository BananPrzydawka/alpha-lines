//! The alpha-lines game engine.
//!
//! [`Game`] is one board: plain data, cloned freely, with the caller owning the [`Scratch`]
//! workspace and lending it to the calls that need it. It stores and indexes only the 80
//! playable squares, in the same index space as the legality mask.
//!
//! The board is scored by a level-based incremental scorer that maintains the score as moves
//! are applied rather than rescoring from scratch. See [`game`] for how levels work.
//!
/// Defaults generated from the repository's config.json at build time.
pub mod config {
    include!(concat!(env!("OUT_DIR"), "/config.rs"));
}

pub mod game;

pub use game::{Game, Scratch, HEIGHT, HW, ROW, SQUARES, WIDTH};


pub mod klent;
