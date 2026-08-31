//! Mirror of the game-relevant half of `main/config.py`.
//!
//! The training/model hyper-parameters in the Python config have no bearing on the game
//! rules, so only the board geometry and the kernel-parallelism switch are ported.

pub const HEIGHT: usize = 10; // 10
pub const WIDTH: usize = 16; // 16
pub const BOARD_SIZE: usize = HEIGHT * WIDTH;

/// `game_kernels_parralel` in `main/config.py`. It is `False` there, so the numba kernels
/// run single-threaded; this port is single-threaded to match.
pub const GAME_KERNELS_PARALLEL: bool = false;
