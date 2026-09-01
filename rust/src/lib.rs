//! The alpha-lines batched game engine.
//!
//! One implementation: a level-based incremental scorer that maintains each board's score
//! as moves are applied, instead of rescoring from scratch. See [`incremental`] for how the
//! levels work and why they cannot form a cycle.
//!
//! The module has no internal dependencies — board shape, square encoding and RNG are all
//! defined in it — so it can be lifted out of this crate as a single file. The port of the
//! original numba kernels it was checked against lives in `tests/oracle`, not here.

pub mod incremental;

pub use incremental::{IncrementalGame, HEIGHT, HW, WIDTH};
