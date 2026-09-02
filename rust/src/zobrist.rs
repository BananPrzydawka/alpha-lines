//! Zobrist hashing: a 64-bit key per position, updated by XOR as moves are played.
//!
//! The point is transpositions. Two different move orders that reach the same board are the
//! same position, and a search keyed by hash merges them into one node instead of searching
//! both. So the key is a function of the board alone.
//!
//! The board is the whole state: levels, scores and the legal mask are all derived from it,
//! and `move_count` matters only as "has anything been played", which is `cells != opening`.
//! Moves are simultaneous, so there is no side to move to fold in either.
//!
//! [`step`] takes a hash, the board *before* the move, and the move, and returns the hash
//! after — without touching [`crate::Game`]. That is what lets a search look up a child node
//! before deciding whether to build one.
//!
//! Keys are compile-time constants, so there is nothing to construct and no table to pass
//! around, and a given board hashes the same in every run.

use crate::game::{
    NEIGHBOURS, OFF_BOARD, PLAYER_0_MARK, PLAYER_1_MARK, REMOVED_SQUARE, SQUARES,
};

/// One slot per value a square can hold, indexed by the encoding itself.
const STATES: usize = 5;

/// SplitMix64, as a const fn: `(next state, output)`.
const fn next(s: u64) -> (u64, u64) {
    let s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (s, z ^ (z >> 31))
}

/// `KEYS[square][state]`, with the playable and non-playable states left at zero.
///
/// Zeroing the playable state is what makes the opening position hash to [`OPENING`] and a
/// placed mark cost one XOR instead of two: there is no old key to cancel.
const KEYS: [[u64; STATES]; SQUARES] = {
    let mut k = [[0u64; STATES]; SQUARES];
    let mut seed = 0x5EED_A1F4_1E50_2B17;
    let mut sq = 0;
    while sq < SQUARES {
        let mut st = 0;
        while st < STATES {
            if st == REMOVED_SQUARE as usize
                || st == PLAYER_0_MARK as usize
                || st == PLAYER_1_MARK as usize
            {
                let (s, v) = next(seed);
                seed = s;
                k[sq][st] = v;
            }
            st += 1;
        }
        sq += 1;
    }
    k
};

/// `PLACE[square][player]`: the XOR that turns a playable square into that player's mark.
const PLACE: [[u64; 2]; SQUARES] = {
    let mut p = [[0u64; 2]; SQUARES];
    let mut sq = 0;
    while sq < SQUARES {
        p[sq][0] = KEYS[sq][PLAYER_0_MARK as usize];
        p[sq][1] = KEYS[sq][PLAYER_1_MARK as usize];
        sq += 1;
    }
    p
};

/// `BLAST[square][state]`: the XOR that turns a square holding `state` into a removed one.
///
/// Zero when the square already holds [`REMOVED_SQUARE`], so a collision can XOR all five of
/// its victims unconditionally. The extra row is [`OFF_BOARD`] and is all zeros, so a
/// diagonal that leaves the board costs an XOR with nothing rather than a branch.
const BLAST: [[u64; STATES]; SQUARES + 1] = {
    let mut b = [[0u64; STATES]; SQUARES + 1];
    let mut sq = 0;
    while sq < SQUARES {
        let mut st = 0;
        while st < STATES {
            b[sq][st] = KEYS[sq][st] ^ KEYS[sq][REMOVED_SQUARE as usize];
            st += 1;
        }
        sq += 1;
    }
    b
};

/// The hash of the opening position, where every square is playable.
pub const OPENING: u64 = 0;

/// Hash a board from scratch. For a root position, or to check an incremental chain.
pub fn hash(cells: &[i8; SQUARES]) -> u64 {
    let mut h = OPENING;
    for (sq, &v) in cells.iter().enumerate() {
        h ^= KEYS[sq][v as usize];
    }
    h
}

/// The hash after playing `i0` and `i1` on the board `cells` currently holds.
///
/// `cells` must be the position *before* the move and `h` its hash. Both squares must be
/// legal, which the caller knows from the mask.
///
/// An ordinary move is two XORs and never looks at the board: each square goes from playable,
/// whose key is zero, to a mark. A collision is the only case that has to read anything —
/// five squares are cleared and what they held decides the key to cancel.
#[inline]
pub fn step(h: u64, cells: &[i8; SQUARES], i0: usize, i1: usize) -> u64 {
    debug_assert!(i0 < SQUARES && i1 < SQUARES, "move off the board");

    if i0 != i1 {
        return h ^ PLACE[i0][0] ^ PLACE[i1][1];
    }

    let mut h = h ^ BLAST[i0][cells[i0] as usize];
    for &n in &NEIGHBOURS[i0] {
        // For OFF_BOARD the state read is meaningless, but its BLAST row is all zeros, so
        // the XOR contributes nothing. Clamping beats branching on a path this short.
        let state = cells[(n as usize).min(SQUARES - 1)] as usize;
        h ^= BLAST[n as usize][state];
    }
    debug_assert_eq!(OFF_BOARD as usize, SQUARES, "the sentinel must index the zero row");
    h
}
