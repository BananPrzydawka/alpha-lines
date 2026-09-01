//! Play a large batch of games and require the engine to match the oracle bit for bit, at
//! every move, in every observable field.
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

mod oracle;

use alpha_lines_game::incremental::{
    rebuild_levels, Rng, Scratch, HEIGHT, HW, INF, MAX_LEVEL, PLAYABLE_SQUARE, PLAYER_0_MARK,
    PLAYER_1_MARK, WIDTH,
};
use alpha_lines_game::IncrementalGame;
use oracle::{apply_and_score_kernel, legal_masks_kernel, sample_move_kernel, score_player};

/// The oracle as a batch of games: four plain arrays and the kernels that move them. This is
/// all `BatchedLinesGame` ever was for the purposes of checking the engine, minus the
/// encoding and rendering the engine does not implement.
struct Oracle {
    n: usize,
    boards: Vec<i8>,
    scores: Vec<f32>,
    move_counts: Vec<i32>,
    finished: Vec<bool>,
    /// the oracle's own RNG, not the engine's — they must be separate to be evidence
    rng: oracle::Rng,
}

impl Oracle {
    fn new(n: usize, seed: u64) -> Self {
        let mut boards = vec![0i8; n * HW];
        for g in 0..n {
            for r in 0..HEIGHT {
                for c in 0..WIDTH {
                    if (r + c) % 2 == 0 {
                        boards[g * HW + r * WIDTH + c] = PLAYABLE_SQUARE;
                    }
                }
            }
        }
        Oracle {
            n,
            boards,
            scores: vec![0.0; n * 2],
            move_counts: vec![0; n],
            finished: vec![false; n],
            rng: oracle::Rng::new(seed),
        }
    }

    fn masks(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        legal_masks_kernel(&self.boards, self.n, &self.move_counts, WIDTH / 2, HEIGHT, WIDTH)
    }

    fn active(&self) -> Vec<bool> {
        self.finished.iter().map(|&f| !f).collect()
    }

    fn apply(&mut self, idx_0: &[i64], idx_1: &[i64]) {
        let w = WIDTH as i64;
        let active = self.active();
        let (r0, c0): (Vec<i64>, Vec<i64>) =
            idx_0.iter().map(|&i| (i / w, i % w)).unzip();
        let (r1, c1): (Vec<i64>, Vec<i64>) =
            idx_1.iter().map(|&i| (i / w, i % w)).unzip();
        apply_and_score_kernel(
            &mut self.boards, &mut self.move_counts, &mut self.finished, &mut self.scores,
            &r0, &c0, &r1, &c1, &active, self.n, HEIGHT, WIDTH,
        );
    }

    /// One move per active game, sampled the way the original kernels sample.
    fn distribution_step(&mut self, d0: &[f32], d1: &[f32]) {
        let active = self.active();
        if !active.iter().any(|&a| a) {
            return;
        }
        let (m0, m1, _, _) = self.masks();
        let (r0, c0) =
            sample_move_kernel(d0, &m0, &active, self.n, HEIGHT, WIDTH, &mut self.rng);
        let (r1, c1) =
            sample_move_kernel(d1, &m1, &active, self.n, HEIGHT, WIDTH, &mut self.rng);
        apply_and_score_kernel(
            &mut self.boards, &mut self.move_counts, &mut self.finished, &mut self.scores,
            &r0, &c0, &r1, &c1, &active, self.n, HEIGHT, WIDTH,
        );
    }
}

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

/// The engine keeps integer scores because every score is a sum of run lengths; the oracle
/// keeps f32 because numpy did. The comparison needs one of them converted.
fn scores_as_f32(inc: &IncrementalGame) -> Vec<f32> {
    inc.scores.iter().map(|&s| s as f32).collect()
}

/// The dense masks the oracle produces, rebuilt from the engine's 80-bit legality words.
/// Not something the engine offers — this is the test doing the unpacking, so the two
/// representations can be compared at all.
fn masks_from_bits(inc: &IncrementalGame) -> (Vec<f32>, Vec<f32>) {
    let half = WIDTH / 2;
    let mut m0 = vec![0.0f32; inc.n * HW];
    let mut m1 = vec![0.0f32; inc.n * HW];
    for g in 0..inc.n {
        let bits = inc.legal_bits(g);
        let first = inc.move_counts[g] == 0;
        for i in 0..HW {
            let k = i >> 1;
            if i % 2 != (i / WIDTH) % 2 || bits[k >> 6] >> (k & 63) & 1 == 0 {
                continue;
            }
            let c = i % WIDTH;
            if !(first && c >= half) {
                m0[g * HW + i] = 1.0;
            }
            if !(first && c < half) {
                m1[g * HW + i] = 1.0;
            }
        }
    }
    (m0, m1)
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
fn the_engine_matches_the_oracle_bit_for_bit() {
    let n: usize = std::env::var("PARITY_GAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);
    let seed = 0xa1f4_1e5u64;

    let mut refg = Oracle::new(n, seed);
    let mut inc = IncrementalGame::new(n, seed);

    let mut rng = Rng::new(seed ^ 0x9e37_79b9);
    // [random, forced] — the odd-numbered games and the even-numbered ones
    let mut moves = [0usize; 2];
    let mut collisions = [0usize; 2];
    let mut steps = 0usize;

    while !refg.finished.iter().all(|&f| f) {
        let (m0, m1, _, _) = refg.masks();
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

        refg.apply(&idx0, &idx1);
        inc.action_step(&idx0, &idx1).unwrap();

        assert_eq!(inc.boards, refg.boards, "boards diverged at step {steps}");
        assert_eq!(scores_as_f32(&inc), refg.scores, "scores diverged at step {steps}");
        assert_eq!(inc.move_counts, refg.move_counts, "move counts at step {steps}");
        assert_eq!(inc.finished, refg.finished, "finished flags at step {steps}");
        let (m0, m1, _, _) = refg.masks();
        assert_eq!(masks_from_bits(&inc), (m0, m1), "legality at step {steps}");
        check_levels_and_scores(&inc, &format!("step {steps}"));
        check_legality(&inc, &format!("step {steps}"));

        steps += 1;
        assert!(steps < 200, "rollout did not terminate");
    }

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
fn the_bitboard_sampler_picks_the_same_moves_as_the_oracle_sampler() {
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

    let mut refg = Oracle::new(n, seed);
    let mut inc = IncrementalGame::new(n, seed);
    let mut steps = 0usize;

    while !refg.finished.iter().all(|&f| f) {
        refg.distribution_step(&d0, &d1);
        inc.distribution_step(&d0, &d1);
        assert_eq!(inc.boards, refg.boards, "sampler diverged at step {steps}");
        assert_eq!(scores_as_f32(&inc), refg.scores, "scores diverged at step {steps}");
        check_levels_and_scores(&inc, &format!("step {steps}"));
        check_legality(&inc, &format!("step {steps}"));
        steps += 1;
        assert!(steps < 200, "rollout did not terminate");
    }
    assert_eq!(steps, 44, "a full rollout should take 44 steps, took {steps}");
}

/// Every externally visible call, checked against the oracle on many game states.
///
/// The two tests above cover the engine's *behaviour* — that playing a game produces the
/// same board and the same score. This one covers its *surface*: for each public call, on
/// each of a rollout's states, does the engine hand back what the oracle says it should.
///
/// `legal_bits` is the one call returning a different representation rather than different
/// data: it packs into 80 bits what the oracle spreads over 160 floats. So it is checked for
/// logical equivalence against the board itself, and the oracle's dense masks are separately
/// checked to be exactly that set intersected with the first-move half rule.
#[test]
fn every_public_call_agrees_with_the_oracle() {
    use alpha_lines_game::incremental::LEGAL_WORDS;

    let n: usize = 512;
    let seed = 0xc0de_5eedu64;
    let half = WIDTH / 2;

    let mut rng = Rng::new(seed);
    let d0: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
    let d1: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();

    let mut refg = Oracle::new(n, seed);
    let mut inc = IncrementalGame::new(n, seed);
    let mut steps = 0usize;

    loop {
        assert_eq!(inc.boards, refg.boards, "boards at step {steps}");
        assert_eq!(scores_as_f32(&inc), refg.scores, "scores at step {steps}");
        assert_eq!(inc.move_counts, refg.move_counts, "move_counts at step {steps}");
        assert_eq!(inc.finished, refg.finished, "finished at step {steps}");
        assert_eq!(inc.scores.len(), 2 * n, "one integer score per player per game");

        // legal_bits, cell by cell, against the board and against the oracle's masks
        let (want0, want1, wc0, wc1) = refg.masks();
        for g in 0..n {
            let bits = inc.legal_bits(g);
            assert_eq!(bits.len(), LEGAL_WORDS);
            let first = refg.move_counts[g] == 0;
            let (mut n0, mut n1) = (0.0f32, 0.0f32);
            for i in 0..HW {
                let k = i >> 1;
                let set = i % 2 == (i / WIDTH) % 2 && bits[k >> 6] >> (k & 63) & 1 == 1;
                let playable = refg.boards[g * HW + i] == PLAYABLE_SQUARE;
                assert_eq!(set, playable, "legal_bits g{g} cell {i} at step {steps}");
                let c = i % WIDTH;
                let want_0 = playable && !(first && c >= half);
                let want_1 = playable && !(first && c < half);
                assert_eq!(want0[g * HW + i] == 1.0, want_0, "mask0 g{g} cell {i}");
                assert_eq!(want1[g * HW + i] == 1.0, want_1, "mask1 g{g} cell {i}");
                n0 += f32::from(want_0);
                n1 += f32::from(want_1);
            }
            assert_eq!((wc0[g], wc1[g]), (n0, n1), "legal counts g{g} at step {steps}");
        }

        // from_state is handed the board and nothing else, so everything else has to come
        // back from it — including the two flags it is no longer told
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
        check_legality(&rebuilt, "from_state");

        if refg.finished.iter().all(|&f| f) {
            break;
        }
        refg.distribution_step(&d0, &d1);
        inc.distribution_step(&d0, &d1);
        steps += 1;
        assert!(steps < 200, "rollout did not terminate");
    }

    // an illegal action must be rejected, name the games it was wrong about, and leave the
    // batch exactly as it was
    let mut a = IncrementalGame::new(n, seed);
    let before = a.boards.clone();
    let bad = vec![1i64; n]; // (0,1) is not a playable square
    let err = a.action_step(&bad, &bad).unwrap_err();
    assert!(err.contains("Invalid move"), "unexpected rejection message: {err}");
    assert!(err.contains(&format!("{}", n - 1)), "the message should name every bad game");
    assert_eq!(a.boards, before, "a rejected action_step must leave the batch untouched");

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

/// The engine and the oracle define the square encoding and the RNG separately — on purpose,
/// so that agreeing with each other means something. Separate definitions of the same
/// constant are exactly the kind of thing that drifts, so this pins them together. The
/// constants are a compile-time check and cost nothing to keep.
#[test]
fn the_engine_and_the_oracle_still_agree_on_the_encoding_and_the_rng() {
    use alpha_lines_game::incremental as inc;

    const _: () = assert!(inc::NON_PLAYABLE_SQUARE == oracle::NON_PLAYABLE_SQUARE);
    const _: () = assert!(inc::PLAYABLE_SQUARE == oracle::PLAYABLE_SQUARE);
    const _: () = assert!(inc::REMOVED_SQUARE == oracle::REMOVED_SQUARE);
    const _: () = assert!(inc::PLAYER_0_MARK == oracle::PLAYER_0_MARK);
    const _: () = assert!(inc::PLAYER_1_MARK == oracle::PLAYER_1_MARK);

    // and the two RNGs must be the same generator, or the samplers would diverge
    let (mut a, mut b) = (inc::Rng::new(0xabc), oracle::Rng::new(0xabc));
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
