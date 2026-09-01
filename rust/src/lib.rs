//! The alpha-lines game engine.
//!
//! [`Game`] is one board: plain data, cloned freely, with the caller owning the [`Scratch`]
//! workspace and lending it to the calls that need it.
//!
//! The board is scored by a level-based incremental scorer that maintains the score as moves
//! are applied rather than rescoring from scratch. See [`game`] for how levels work.

pub mod game;

pub use game::{Game, Scratch, HEIGHT, HW, WIDTH};
