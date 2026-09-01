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
//!   score. `check_invariants` re-derives every level from scratch and re-scores the board
//!   with the reference's own scorer, so the internal state is pinned too, not just the
//!   output.
//! * **Both repair strategies.** `Scratch::bfs_repair` picks between rebuilding a component
//!   and walking its levels up. Two incremental engines run side by side, one on each, so
//!   neither is left as untested dead code and they are checked against each other as well.
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

use alpha_lines_game::incremental::HW;
use alpha_lines_game::rng::Rng;
use alpha_lines_game::{BatchedLinesGame, IncrementalGame};

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
    let mut bfs = IncrementalGame::new(n, seed);
    let mut walk = IncrementalGame::new(n, seed);
    bfs.set_bfs_repair(true);
    walk.set_bfs_repair(false);

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
        bfs.action_step(&idx0, &idx1).unwrap();
        walk.action_step(&idx0, &idx1).unwrap();

        // The masks are the incremental engine's own bitboard expansion, not the reference
        // kernel, so they need comparing like everything else.
        assert_eq!(bfs.get_legal_masks(), refg.get_legal_masks(), "masks at step {steps}");

        for (name, inc) in [("bfs", &bfs), ("walk", &walk)] {
            assert_eq!(inc.boards, refg.boards, "{name}: boards diverged at step {steps}");
            assert_eq!(inc.scores_f32(), refg.scores, "{name}: scores diverged at step {steps}");
            assert_eq!(inc.move_counts, refg.move_counts, "{name}: move counts at step {steps}");
            assert_eq!(inc.finished, refg.finished, "{name}: finished flags at step {steps}");
        }
        // the two strategies must agree on the internal state too, not only the output
        assert_eq!(bfs.levels, walk.levels, "the repair strategies disagree at step {steps}");
        bfs.check_invariants().unwrap_or_else(|e| panic!("step {steps}: {e}"));
        walk.check_invariants().unwrap_or_else(|e| panic!("step {steps}: {e}"));

        steps += 1;
        assert!(steps < 200, "rollout did not terminate");
    }

    // the terminal outcome is derived state and gets its own comparison, once, at the end
    assert_eq!(bfs.to_reference().get_terminal_outcomes(), refg.get_terminal_outcomes());

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
        assert_eq!(inc.scores_f32(), refg.scores, "scores diverged at step {steps}");
        inc.check_invariants().unwrap_or_else(|e| panic!("step {steps}: {e}"));
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
/// Where a call returns a different representation rather than different data — `legal_bits`
/// packs into 80 bits what `get_legal_masks` spreads over 160 floats — it is checked for
/// logical equivalence against the board itself instead of for byte equality.
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
        assert_eq!(inc.scores_f32(), refg.scores, "scores_f32 at step {steps}");

        // --- masks: same four vectors, byte for byte ---
        assert_eq!(inc.get_legal_masks(), refg.get_legal_masks(), "get_legal_masks {steps}");
        assert_eq!(inc.raw_masks(), refg.raw_masks(), "raw_masks at step {steps}");

        // --- legal_masks_into must fill dirty buffers with exactly that ---
        let (want0, want1, wantc0, wantc1) = refg.get_legal_masks();
        let mut m0 = vec![7.0f32; n * HW];
        let mut m1 = vec![7.0f32; n * HW];
        let mut c0 = vec![7.0f32; n];
        let mut c1 = vec![7.0f32; n];
        inc.legal_masks_into(&mut m0, &mut m1, &mut c0, &mut c1);
        assert_eq!((m0, m1, c0, c1), (want0.clone(), want1.clone(), wantc0, wantc1),
                   "legal_masks_into at step {steps}");

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
        let view = inc.to_reference();
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
        let rebuilt = IncrementalGame::from_state(
            inc.boards.clone(), inc.move_counts.clone(), inc.finished.clone(), seed,
        );
        assert_eq!(rebuilt.scores, inc.scores, "from_state scores at step {steps}");
        assert_eq!(rebuilt.levels, inc.levels, "from_state levels at step {steps}");
        rebuilt.check_invariants().unwrap();

        if refg.finished.iter().all(|&f| f) {
            break;
        }
        refg.distribution_step(&d0, &d1);
        inc.distribution_step(&d0, &d1);
        steps += 1;
        assert!(steps < 200, "rollout did not terminate");
    }

    // --- terminal-only calls ---
    let view = inc.to_reference();
    assert_eq!(view.get_terminal_outcomes(), refg.get_terminal_outcomes());

    // --- selecting a subset of games must detach the same way from both ---
    let picked: Vec<usize> = (0..n).step_by(3).collect();
    let a = refg.clone_states_to_batch(&picked);
    let b = inc.to_reference().clone_states_to_batch(&picked);
    assert_eq!((a.boards, a.scores, a.move_counts, a.finished),
               (b.boards, b.scores, b.move_counts, b.finished), "clone_states_to_batch");

    // --- the printed form has to round-trip identically through both ---
    let text = refg.format_state(Some(&picked[..4]), 0);
    let from_ref = BatchedLinesGame::import_prints(&text, 0, seed).unwrap();
    let reimported = IncrementalGame::from_state(
        from_ref.boards.clone(), from_ref.move_counts.clone(), from_ref.finished.clone(), seed,
    );
    assert_eq!(reimported.boards, from_ref.boards, "import_prints boards");
    assert_eq!(reimported.scores_f32(), from_ref.scores, "import_prints scores");
    reimported.check_invariants().unwrap();

    // --- an illegal action must be rejected the same way, with the same message ---
    let mut a = BatchedLinesGame::new(n, seed);
    let mut b = IncrementalGame::new(n, seed);
    let bad = vec![1i64; n]; // (0,1) is not a playable square
    assert_eq!(a.action_step(&bad, &bad).unwrap_err(), b.action_step(&bad, &bad).unwrap_err());
    assert_eq!(a.boards, b.boards, "a rejected action_step must leave the batch untouched");

    println!("{n} games, {steps} steps, every public call compared at every step");
}
