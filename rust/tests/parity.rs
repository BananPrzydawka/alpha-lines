//! The one test: play a large batch of games and require the incremental engine to match
//! the reference bit for bit, at every move, in every observable field.
//!
//! The reference (`game.rs` + `game_kernels.rs`) is a straight port of `main/game.py` and is
//! itself verified byte-for-byte against the Python by `xcheck/xcheck.py`. So parity here is
//! parity with the Python, transitively — which is why this single test can stand in for a
//! pile of unit tests: anything the incremental engine gets wrong about a board, a score, a
//! move count or a terminal flag shows up as a mismatch within a move or two of happening.
//!
//! Three things a bare parity check would *not* catch, and how this covers them:
//!
//! * **Levels.** They are internal to the incremental engine, so the reference has nothing
//!   to compare against and a corrupt level can sit there until it eventually poisons a
//!   score. `check_levels_and_scores` re-derives every level from scratch and re-scores
//!   with the reference's own scorer, so the internal state is pinned too, not just the
//!   output.
//! * **The masks.** The reference builds them by scanning; the incremental engine packs the
//!   same information into 80 bits. The dense form is compared here through the engine's
//!   reference view, and the bitboard against the board itself in the API test below.
//! * **The rare paths.** Collisions are what drive the slow removal path, but only if there
//!   is structure for them to cut: forcing one on every move flattens the board and the slow
//!   path then runs *zero* times, measured. Pure random play is in fact the best exerciser of
//!   it, and already collides on ~7% of moves. So the batch is mostly free play, and only the
//!   even-numbered games are nudged — one forced collision every sixth step, which triples
//!   the collision rate while leaving the blobs six moves to grow back. Both halves are
//!   counted separately and both are asserted, so neither can quietly stop testing anything.
//!
//! Meant to be run as `cargo test --release`; a debug build is ~50x slower. `PARITY_GAMES`
//! overrides the batch size.

use alpha_lines_game::config::{HEIGHT, WIDTH};
use alpha_lines_game::game_kernels::{score_player, PLAYABLE_SQUARE, PLAYER_0_MARK, PLAYER_1_MARK};
use alpha_lines_game::incremental::{rebuild_levels, Scratch, HW, INF, MAX_LEVEL};
use alpha_lines_game::rng::Rng;
use alpha_lines_game::{BatchedLinesGame, IncrementalGame};

// ---------------------------------------------------------------- checks the engine does not carry
//
// The incremental engine ships no self-verification: re-deriving levels and re-scoring a
// board is test code, and test code belongs here. Both checkers work off the public surface
// — `boards`, `levels`, `scores`, `legal_bits` — plus the reference's own scorer, which is
// what makes them evidence rather than the engine agreeing with itself.

/// Every level must equal what a from-scratch BFS would produce, and every running score
/// must equal what the reference scorer says about the same board.
fn check_levels_and_scores(inc: &IncrementalGame, where_: &str) {
    let mut scratch = Scratch::new();
    let mut fresh = vec![0u8; HW];
    for g in 0..inc.n {
        let cells = &inc.boards[g * HW..(g + 1) * HW];
        rebuild_levels(cells, &mut fresh, &mut scratch);
        assert_eq!(
            &inc.levels[g * HW..(g + 1) * HW],
            &fresh[..],
            "{where_}: game {g} levels drifted from a from-scratch rebuild"
        );
        for (k, mark) in [PLAYER_0_MARK, PLAYER_1_MARK].into_iter().enumerate() {
            let want = score_player(cells, 0, mark, HEIGHT, WIDTH) as i32;
            assert_eq!(
                inc.scores[g * 2 + k], want,
                "{where_}: game {g} player {k} running score is wrong"
            );
        }
    }
}

/// Every legality bit must match the board, and `finished` must match whether any bit is
/// left. This also pins the bit packing itself, since the two sides are built by different
/// code.
fn check_legality(inc: &IncrementalGame, where_: &str) {
    for g in 0..inc.n {
        let cells = &inc.boards[g * HW..(g + 1) * HW];
        let bits = inc.legal_bits(g);
        let mut any = false;
        for i in 0..HW {
            let k = i >> 1;
            let playable = cells[i] == PLAYABLE_SQUARE;
            let parity = i % 2 == (i / WIDTH) % 2;
            let set = parity && bits[k >> 6] >> (k & 63) & 1 == 1;
            assert_eq!(set, playable, "{where_}: game {g} cell {i} legality bit");
            any |= playable;
        }
        assert_eq!(inc.finished[g], !any, "{where_}: game {g} finished flag");
    }
}

/// The incremental engine deliberately knows nothing about the reference type, so the
/// conversion the comparison needs lives here, in the code doing the comparing.
fn as_reference(inc: &IncrementalGame) -> BatchedLinesGame {
    let mut r = BatchedLinesGame::new(inc.n, 0);
    r.boards.copy_from_slice(&inc.boards);
    for (dst, &src) in r.scores.iter_mut().zip(inc.scores.iter()) {
        *dst = src as f32;
    }
    r.move_counts.copy_from_slice(&inc.move_counts);
    r.finished.copy_from_slice(&inc.finished);
    r
}


/// One uniformly random legal index per game; 0 for finished games, which `action_step`
/// range-checks but never applies.
fn pick(mask: &[f32], finished: &[bool], n: usize, rng: &mut Rng) -> Vec<i64> {
    let mut out = vec![0i64; n];
    let mut legal: Vec<i64> = Vec::with_capacity(HW);
    for g in 0..n {
        if finished[g] {
            continue;
        }
        legal.clear();
        for i in 0..HW {
            if mask[g * HW + i] == 1.0 {
                legal.push(i as i64);
            }
        }
        assert!(!legal.is_empty(), "active game {g} has no legal moves");
        out[g] = legal[rng.randint(legal.len() as u64) as usize];
    }
    out
}

#[test]
fn the_incremental_engine_matches_the_reference_bit_for_bit() {
    let n: usize = std::env::var("PARITY_GAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);
    let seed = 0xa1f4_1e5u64;

    let mut refg = BatchedLinesGame::new(n, seed);
    let mut inc = IncrementalGame::new(n, seed);

    let mut rng = Rng::new(seed ^ 0x9e37_79b9);
    // [random, forced] — the odd-numbered games and the even-numbered ones
    let mut moves = [0usize; 2];
    let mut collisions = [0usize; 2];
    let mut steps = 0usize;

    while !refg.finished.iter().all(|&f| f) {
        let (m0, m1, _, _) = refg.get_legal_masks();
        let idx0 = pick(&m0, &refg.finished, n, &mut rng);
        let mut idx1 = pick(&m1, &refg.finished, n, &mut rng);

        // Every sixth step, steer the even-numbered games onto their opponent's square
        // wherever that square is legal for both. On the opening move the halves are disjoint
        // so no collision is possible and the independent draw stands. The odd-numbered games
        // are never touched: they are ordinary random play, and they are both the volume of
        // the evidence and, measurably, the heaviest user of the slow paths.
        if steps % 6 == 0 {
            for g in (0..n).step_by(2) {
                if !refg.finished[g] && m1[g * HW + idx0[g] as usize] == 1.0 {
                    idx1[g] = idx0[g];
                }
            }
        }
        for g in 0..n {
            if !refg.finished[g] {
                let half = 1 - g % 2; // 0 = random, 1 = forced
                moves[half] += 1;
                if idx0[g] == idx1[g] {
                    collisions[half] += 1;
                }
            }
        }

        refg.action_step(&idx0, &idx1).unwrap();
        inc.action_step(&idx0, &idx1).unwrap();

        let view = as_reference(&inc);
        assert_eq!(view.boards, refg.boards, "boards diverged at step {steps}");
        assert_eq!(view.scores, refg.scores, "scores diverged at step {steps}");
        assert_eq!(view.move_counts, refg.move_counts, "move counts at step {steps}");
        assert_eq!(view.finished, refg.finished, "finished flags at step {steps}");
        assert_eq!(view.get_legal_masks(), refg.get_legal_masks(), "masks at step {steps}");
        check_levels_and_scores(&inc, &format!("step {steps}"));
        check_legality(&inc, &format!("step {steps}"));

        steps += 1;
        assert!(steps < 200, "rollout did not terminate");
    }

    // the terminal outcome is derived state and gets its own comparison, once, at the end
    assert_eq!(as_reference(&inc).get_terminal_outcomes(), refg.get_terminal_outcomes());

    // If either half ever stops reaching the interesting paths, fail loudly rather than
    // silently testing nothing. The random half is held to the collision rate free play
    // actually produces (~7%); the nudged half to the higher rate it exists to create.
    assert!(moves[0] + moves[1] > n * 30, "only {} moves over {n} games", moves[0] + moves[1]);
    assert!(
        collisions[0] * 100 > moves[0],
        "random play produced only {} collisions in {} moves",
        collisions[0], moves[0]
    );
    assert!(
        collisions[1] * 8 > moves[1],
        "the nudged half produced only {} collisions in {} moves",
        collisions[1], moves[1]
    );
    println!(
        "{n} games, {steps} steps\n  free play: {} moves, {} collisions ({:.1}%)\n  nudged:    \
         {} moves, {} collisions ({:.1}%)",
        moves[0], collisions[0], collisions[0] as f64 / moves[0] as f64 * 100.0,
        moves[1], collisions[1], collisions[1] as f64 / moves[1] as f64 * 100.0,
    );
}

/// The sampler needs its own driver, which is why this cannot fold into the test above.
///
/// That one feeds both engines an explicit move index so it can control what gets played;
/// it therefore never runs the sampler at all. Here both sides pick their own moves from the
/// same distributions and the same RNG seed, so a single differing draw sends the two games
/// down permanently different paths — which makes divergence loud rather than subtle.
///
/// The incremental sampler walks set bits in a bitboard; the reference walks 160 squares
/// against a materialized f32 mask. Same order, same accumulation, same RNG consumption,
/// including the degenerate all-zero-weight branch — this is what pins that down.
#[test]
fn the_bitboard_sampler_picks_the_same_moves_as_the_reference_sampler() {
    let n: usize = std::env::var("PARITY_GAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000)
        .min(2_000);
    let seed = 0x5a3_1e5u64;

    // Both distributions are strictly positive on purpose. If every *legal* square carries
    // exactly zero weight, both samplers take the same documented fallback — a uniform draw
    // over all 160 squares — which can land on a square that is not playable at all. The
    // reference absorbs that because it rescores the board from scratch; the incremental
    // engine's `insert` assumes the square it is handed was playable, so from there the two
    // legitimately disagree. That is inherited from the Python and is not what this test is
    // about; a softmax policy never produces it.
    let mut rng = Rng::new(seed);
    let d0: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
    let d1: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();

    let mut refg = BatchedLinesGame::new(n, seed);
    let mut inc = IncrementalGame::new(n, seed);
    let mut steps = 0usize;

    while !refg.finished.iter().all(|&f| f) {
        refg.distribution_step(&d0, &d1);
        inc.distribution_step(&d0, &d1);
        assert_eq!(inc.boards, refg.boards, "sampler diverged at step {steps}");
        assert_eq!(as_reference(&inc).scores, refg.scores, "scores diverged at step {steps}");
        check_levels_and_scores(&inc, &format!("step {steps}"));
        check_legality(&inc, &format!("step {steps}"));
        steps += 1;
        assert!(steps < 200, "rollout did not terminate");
    }
    assert_eq!(steps, 44, "a full rollout should take 44 steps, took {steps}");
}

/// Every externally visible function, checked against the reference on many game states.
///
/// The two tests above cover the engine's *behaviour* — that playing a game produces the
/// same board and the same score. This one covers its *surface*: for each public call, on
/// each of a rollout's states, does the incremental engine hand back what the reference
/// hands back. That is a different question, and the mask path in particular is now the
/// engine's own code rather than the shared kernel, so nothing else pins it down.
///
/// `legal_bits` is the one call that returns a different representation rather than different
/// data: it packs into 80 bits what the reference spreads over 160 floats. So it is checked
/// for logical equivalence against the board itself, and the reference's dense masks are
/// separately checked to be exactly that set intersected with the first-move half rule.
#[test]
fn every_public_call_agrees_with_the_reference() {
    use alpha_lines_game::config::WIDTH;
    use alpha_lines_game::game_kernels::PLAYABLE_SQUARE;
    use alpha_lines_game::incremental::LEGAL_WORDS;

    let n: usize = 512;
    let seed = 0xc0de_5eedu64;
    let half = WIDTH / 2;

    let mut rng = Rng::new(seed);
    let d0: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
    let d1: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();

    let mut refg = BatchedLinesGame::new(n, seed);
    let mut inc = IncrementalGame::new(n, seed);
    let mut steps = 0usize;

    loop {
        // --- state the two engines both own outright ---
        assert_eq!(inc.boards, refg.boards, "boards at step {steps}");
        assert_eq!(inc.move_counts, refg.move_counts, "move_counts at step {steps}");
        assert_eq!(inc.finished, refg.finished, "finished at step {steps}");
        assert_eq!(inc.scores.len(), 2 * n, "one integer score per player per game");
        let (want0, want1, _, _) = refg.get_legal_masks();

        // --- legal_bits: a different representation, so checked for logical equivalence ---
        for g in 0..n {
            let bits = inc.legal_bits(g);
            let first = refg.move_counts[g] == 0;
            for i in 0..HW {
                let k = i >> 1;
                let set = i % 2 == (i / WIDTH) % 2
                    && bits[k >> 6] >> (k & 63) & 1 == 1;
                let playable = refg.boards[g * HW + i] == PLAYABLE_SQUARE;
                assert_eq!(set, playable, "legal_bits g{g} cell {i} at step {steps}");
                let c = i % WIDTH;
                let want_0 = playable && !(first && c >= half);
                let want_1 = playable && !(first && c < half);
                assert_eq!(want0[g * HW + i] == 1.0, want_0, "mask0 g{g} cell {i}");
                assert_eq!(want1[g * HW + i] == 1.0, want_1, "mask1 g{g} cell {i}");
            }
            assert_eq!(bits.len(), LEGAL_WORDS);
        }

        // --- everything reached through the reference-shaped view ---
        let view = as_reference(&inc);
        assert_eq!(view.scores, refg.scores, "scores at step {steps}");
        assert_eq!(view.get_legal_masks(), refg.get_legal_masks(), "masks at step {steps}");
        assert_eq!(view.boards, refg.boards, "to_reference boards at step {steps}");
        assert_eq!(view.scores, refg.scores, "to_reference scores at step {steps}");
        assert_eq!(view.move_counts, refg.move_counts, "to_reference counts at step {steps}");
        assert_eq!(view.finished, refg.finished, "to_reference finished at step {steps}");
        for player in 0..2 {
            assert_eq!(view.get_encoded_states(player), refg.get_encoded_states(player),
                       "get_encoded_states({player}) at step {steps}");
            assert_eq!(view.format_state(None, player), refg.format_state(None, player),
                       "format_state({player}) at step {steps}");
        }

        // --- from_state has to reconstruct levels and scores from the board alone ---
        // it is handed the board and nothing else, so the two flags have to come back too
        let rebuilt = IncrementalGame::from_state(inc.boards.clone(), seed);
        assert_eq!(rebuilt.scores, inc.scores, "from_state scores at step {steps}");
        assert_eq!(rebuilt.levels, inc.levels, "from_state levels at step {steps}");
        assert_eq!(rebuilt.finished, inc.finished, "from_state finished at step {steps}");
        assert_eq!(
            rebuilt.move_counts.iter().map(|&m| m.min(1)).collect::<Vec<_>>(),
            inc.move_counts.iter().map(|&m| m.min(1)).collect::<Vec<_>>(),
            "from_state first-move flag at step {steps}"
        );
        check_levels_and_scores(&rebuilt, "from_state");

        if refg.finished.iter().all(|&f| f) {
            break;
        }
        refg.distribution_step(&d0, &d1);
        inc.distribution_step(&d0, &d1);
        steps += 1;
        assert!(steps < 200, "rollout did not terminate");
    }

    // --- terminal-only calls ---
    let view = as_reference(&inc);
    assert_eq!(view.get_terminal_outcomes(), refg.get_terminal_outcomes());

    // --- selecting a subset of games must detach the same way from both ---
    let picked: Vec<usize> = (0..n).step_by(3).collect();
    let a = refg.clone_states_to_batch(&picked);
    let b = as_reference(&inc).clone_states_to_batch(&picked);
    assert_eq!((a.boards, a.scores, a.move_counts, a.finished),
               (b.boards, b.scores, b.move_counts, b.finished), "clone_states_to_batch");

    // --- the printed form has to round-trip identically through both ---
    let text = refg.format_state(Some(&picked[..4]), 0);
    let from_ref = BatchedLinesGame::import_prints(&text, 0, seed).unwrap();
    let reimported = IncrementalGame::from_state(from_ref.boards.clone(), seed);
    assert_eq!(reimported.boards, from_ref.boards, "import_prints boards");
    assert_eq!(as_reference(&reimported).scores, from_ref.scores, "import_prints scores");
    assert_eq!(reimported.move_counts, from_ref.move_counts, "import_prints move_counts");
    assert_eq!(reimported.finished, from_ref.finished, "import_prints finished");
    check_levels_and_scores(&reimported, "import_prints");
    check_legality(&reimported, "import_prints");

    // --- an illegal action must be rejected the same way, with the same message ---
    let mut a = BatchedLinesGame::new(n, seed);
    let mut b = IncrementalGame::new(n, seed);
    let bad = vec![1i64; n]; // (0,1) is not a playable square
    assert_eq!(a.action_step(&bad, &bad).unwrap_err(), b.action_step(&bad, &bad).unwrap_err());
    assert_eq!(a.boards, b.boards, "a rejected action_step must leave the batch untouched");

    println!("{n} games, {steps} steps, every public call compared at every step");
}

/// `clone_states_to_batch` copies derived state — levels and the legality bitboard — rather
/// than recomputing it. That is the whole reason it is cheap, and also the whole reason it
/// can be wrong in a way `from_state` cannot: a copy that drops or shifts a field produces a
/// batch that looks fine until it is played. So the clone is checked three ways: against a
/// from-scratch rebuild of the same boards, against its parent field by field, and by being
/// played forward to make sure it behaves like the games it came from.
#[test]
fn a_clone_carries_the_whole_state_and_keeps_playing_correctly() {
    let n = 256;
    let seed = 0xc10e_5eedu64;
    let mut rng = Rng::new(seed);
    let d0: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
    let d1: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();

    let mut inc = IncrementalGame::new(n, seed);
    for _ in 0..12 {
        inc.distribution_step(&d0, &d1);
    }

    // naming a game twice must give two independent copies, so the list is deliberately
    // out of order and has a repeat in it
    let picked: Vec<usize> = (0..n).step_by(5).chain([3, 3, 1, 0]).collect();
    let clone = inc.clone_states_to_batch(&picked);
    assert_eq!(clone.n, picked.len());

    check_levels_and_scores(&clone, "clone");
    check_legality(&clone, "clone");

    // every field must equal the game it was copied from
    for (d, &g) in picked.iter().enumerate() {
        assert_eq!(&clone.boards[d * HW..(d + 1) * HW], &inc.boards[g * HW..(g + 1) * HW],
                   "clone board {d} <- {g}");
        assert_eq!(&clone.levels[d * HW..(d + 1) * HW], &inc.levels[g * HW..(g + 1) * HW],
                   "clone levels {d} <- {g}");
        assert_eq!(clone.legal_bits(d), inc.legal_bits(g), "clone legality {d} <- {g}");
        assert_eq!(&clone.scores[d * 2..d * 2 + 2], &inc.scores[g * 2..g * 2 + 2],
                   "clone scores {d} <- {g}");
        assert_eq!(clone.move_counts[d], inc.move_counts[g], "clone move_count {d} <- {g}");
        assert_eq!(clone.finished[d], inc.finished[g], "clone finished {d} <- {g}");
    }

    // and it must be indistinguishable from adopting the same boards the slow way
    let rebuilt = IncrementalGame::from_state(clone.boards.clone(), seed);
    assert_eq!(clone.levels, rebuilt.levels, "clone levels differ from a from-scratch rebuild");
    assert_eq!(clone.scores, rebuilt.scores, "clone scores differ from a from-scratch rebuild");
    assert_eq!(clone.finished, rebuilt.finished, "clone finished differs");

    // finally: play it out, and require it to stay correct the whole way
    let mut clone = clone;
    let cn = clone.n;
    let e0: Vec<f32> = (0..cn * HW).map(|_| rng.random() as f32).collect();
    let e1: Vec<f32> = (0..cn * HW).map(|_| rng.random() as f32).collect();
    let mut steps = 0;
    while !clone.finished.iter().all(|&f| f) {
        clone.distribution_step(&e0, &e1);
        check_levels_and_scores(&clone, &format!("clone, step {steps}"));
        check_legality(&clone, &format!("clone, step {steps}"));
        steps += 1;
        assert!(steps < 200, "clone did not terminate");
    }
    assert!(steps > 10, "the clone finished suspiciously fast");
}

/// The incremental engine now defines its own board shape, square encoding and RNG so that
/// it depends on nothing else in the crate. The reference port still has its own copies.
/// Two definitions of the same constant is exactly the kind of thing that drifts, so this
/// pins them together for as long as both exist — and it is a compile-time check, so it
/// costs nothing to keep.
#[test]
fn the_engine_and_the_reference_still_agree_on_the_board_and_the_encoding() {
    use alpha_lines_game::incremental as inc;

    const _: () = assert!(inc::HEIGHT == alpha_lines_game::config::HEIGHT);
    const _: () = assert!(inc::WIDTH == alpha_lines_game::config::WIDTH);
    const _: () = assert!(inc::NON_PLAYABLE_SQUARE == alpha_lines_game::game_kernels::NON_PLAYABLE_SQUARE);
    const _: () = assert!(inc::PLAYABLE_SQUARE == alpha_lines_game::game_kernels::PLAYABLE_SQUARE);
    const _: () = assert!(inc::REMOVED_SQUARE == alpha_lines_game::game_kernels::REMOVED_SQUARE);
    const _: () = assert!(inc::PLAYER_0_MARK == alpha_lines_game::game_kernels::PLAYER_0_MARK);
    const _: () = assert!(inc::PLAYER_1_MARK == alpha_lines_game::game_kernels::PLAYER_1_MARK);

    // and the two RNGs must be the same generator, or the samplers would diverge
    let (mut a, mut b) = (inc::Rng::new(0xabc), Rng::new(0xabc));
    for k in 0..1000 {
        assert_eq!(a.next_u64(), b.next_u64(), "rng streams diverged at draw {k}");
    }
}

/// `MAX_LEVEL` is a bound derived on paper: a level counts steps through one player's marks,
/// a shortest path visits each cell once, and no player can ever hold more than a quarter of
/// the board. This checks the paper against a lot of real games — if the argument were wrong,
/// levels would reach the cap and cells would be declared unreachable that are not.
#[test]
fn no_real_level_ever_comes_close_to_the_cap() {
    let n = 2048;
    let seed = 0x1e5e_1u64;
    let mut rng = Rng::new(seed);
    let d0: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
    let d1: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();

    let mut inc = IncrementalGame::new(n, seed);
    let mut highest = 0u8;
    while !inc.finished.iter().all(|&f| f) {
        inc.distribution_step(&d0, &d1);
        for &l in &inc.levels {
            if l != INF && l > highest {
                highest = l;
            }
        }
    }
    assert!(
        highest < MAX_LEVEL,
        "a real level reached {highest}, but the cap is {MAX_LEVEL} — the bound is wrong"
    );
    // and the run has to actually reach deep levels, or the check above proves nothing
    assert!(highest > 5, "highest level was only {highest}; this run is not testing the bound");
    println!("highest real level over {n} games: {highest} (cap {MAX_LEVEL})");
}
