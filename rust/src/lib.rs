//! The alpha-lines game engine.
//!
//! One game, one struct: [`game::Game`] is a single board of plain data, cloned freely, with
//! the caller owning the scratch workspace — the shape a tree search wants.
//!
//! It is scored by a level-based incremental scorer, which maintains the board's score as
//! moves are applied instead of rescoring it from scratch. See [`game`] for how the levels
//! work and why they cannot form a cycle.
//!
//! There used to be a batched form alongside it — N boards in flat arrays, stepped in
//! lockstep, inherited from the numba kernels this replaced. It was measured against this
//! one and bought nothing: batching flattened out by 64 games at ~22.5 us/game, which is
//! exactly what one game at a time costs when handed the same sampler, and a rollout that
//! picks its move straight out of the legality bitboard beats every batch size by 1.5x. So
//! the batch dimension is gone, along with the active masks, the per-game offsets and the
//! per-step allocations that came with it.
//!
//! The module has no internal dependencies — board shape, square encoding and RNG are all
//! defined in it — so it can be lifted out of this crate as a single file. The port of the
//! original numba kernels it was checked against lives in `tests/oracle`, not here.

pub mod game;

pub use game::{Game, Scratch, HEIGHT, HW, WIDTH};
