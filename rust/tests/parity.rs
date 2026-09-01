//! Play a lot of games and require the engine to match the oracle bit for bit, at every
//! move, in every observable field.
//!
//! The oracle (`tests/oracle`) is a straight port of `main/game_kernels.py` and is itself
//! verified byte-for-byte against the Python by `xcheck/xcheck.py`. So parity here is parity
//! with the Python, transitively — which is why these few tests can stand in for a pile of
//! unit tests: anything the engine gets wrong about a board, a score, a move count or a
//! terminal flag shows up as a mismatch within a move or two of happening.
//!
//! The oracle is still batch-shaped, because the numba kernels it ports were. The engine is
//! not: it plays one board at a time, so every test here holds a `Vec<Game>` against one
//! oracle batch and steps them in step. That mismatch is doing useful work — the games share
//! a single [`Scratch`], and hundreds of them interleaved through one workspace is exactly
//! what would expose scratch state leaking from one board into the next.
//!
//! Four things a bare parity check would *not* catch, and how they are covered:
//!
//! * **Levels.** They are internal to the engine, so the oracle has nothing to compare
//!   against and a corrupt level can sit there until it eventually poisons a score.
//!   `check_levels_and_scores` re-derives every level from scratch and re-scores with the
//!   oracle's own scorer, so the internal state is pinned too, not just the output.
//! * **The masks.** The oracle builds them by scanning 160 floats; the engine packs the same
//!   information into 80 bits. Every square of every game is compared both ways, with the
//!   test unpacking the bits from the documented layout rather than asking the engine to.
//! * **The rare paths.** Collisions are what drive the slow removal path, but only if there
//!   is structure for them to cut: forcing one on every move flattens the board and the slow
//!   path then runs *zero* times, measured. Pure random play is in fact the best exerciser of
//!   it, and already collides on ~7% of moves. So most games are free play, and only the
//!   even-numbered ones are nudged — one forced collision every sixth step, which triples the
//!   collision rate while leaving the blobs six moves to grow back. Both halves are counted
//!   separately and both are asserted, so neither can quietly stop testing anything.
//! * **Forking.** A search clones positions constantly, and a clone that shares something it
//!   should not stays invisible until two branches interfere. `a_fork_is_independent...`
//!   plays a parent and a clone apart and requires neither to feel the other.
//!
//! Meant to be run as `cargo test --release`; a debug build is ~50x slower, but it is worth
//! running there too — that is where `action_step`'s legality `debug_assert`s are live.
//! `PARITY_GAMES` overrides the game count.

mod oracle;

use alpha_lines_game::game::{
    legal_cell, rebuild_levels, Rng, Scratch, HEIGHT, HW, INF, LEGAL_WORDS, MAX_LEVEL,
    PLAYABLE_SQUARE, PLAYER_0_MARK, PLAYER_1_MARK, WIDTH,
};
use alpha_lines_game::Game;
use oracle::{apply_and_score_kernel, legal_masks_kernel, sample_move_kernel, score_player};

/// The oracle as a batch of games: four plain arrays and the kernels that move them, in the
/// shape the numba code had. Minus the encoding and rendering, which the engine does not
/// implement and so has nothing to be checked against.
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


// ------------------------------------------------- checks the engine does not carry
//
// The engine ships no self-verification: re-deriving levels and re-scoring a board is test
// code, and test code belongs here. Both checkers work off the public surface — `cells`,
// `levels`, `scores`, `legal_moves` — plus the oracle's own scorer, which is what makes them
// evidence rather than the engine agreeing with itself.

/// Every level must equal what a from-scratch BFS would produce, and every running score
/// must equal what the oracle's scorer says about the same board.
fn check_levels_and_scores(games: &[Game], where_: &str) {
    let mut scratch = Scratch::new();
    let mut fresh = vec![0u8; HW];
    for (g, game) in games.iter().enumerate() {
        rebuild_levels(&game.cells, &mut fresh, &mut scratch);
        assert_eq!(
            &game.levels[..],
            &fresh[..],
            "{where_}: game {g} levels drifted from a from-scratch rebuild"
        );
        for (k, mark) in [PLAYER_0_MARK, PLAYER_1_MARK].into_iter().enumerate() {
            let want = score_player(&game.cells, 0, mark, HEIGHT, WIDTH) as i32;
            assert_eq!(
                game.scores[k], want,
                "{where_}: game {g} player {k} running score is wrong"
            );
        }
    }
}

/// Every legality bit must match the board, and `finished` must match whether any bit is
/// left. This also pins the bit packing itself, since the two sides are built by different
/// code: the mask is maintained one cleared bit at a time as moves are played, and the
/// board is what the moves actually wrote.
fn check_legality(games: &[Game], where_: &str) {
    for (g, game) in games.iter().enumerate() {
        // the playable set, whoever is to move: the opening halves are disjoint and cover
        // everything, and after the opening both players see the same set
        let (a, b) = (game.legal_moves(0), game.legal_moves(1));
        let bits = [a[0] | b[0], a[1] | b[1]];
        let mut any = false;
        for i in 0..HW {
            let k = i >> 1;
            let playable = game.cells[i] == PLAYABLE_SQUARE;
            let parity = i % 2 == (i / WIDTH) % 2;
            let set = parity && bits[k >> 6] >> (k & 63) & 1 == 1;
            assert_eq!(set, playable, "{where_}: game {g} cell {i} legality bit");
            any |= playable;
        }
        assert_eq!(game.finished, !any, "{where_}: game {g} finished flag");
    }
}

/// Does `player`'s mask hold the bit for board index `i`?
///
/// The engine offers no such call on purpose — the mask is already bits, and a caller reads
/// them. So the test does the reading, which is what makes this evidence: the packing is
/// re-derived here from the documented layout rather than asked for.
fn legal_at(game: &Game, i: usize, player: usize) -> bool {
    if i >= HW || legal_cell(i >> 1) != i {
        return false; // not a playable-parity square, so it holds no bit of its own
    }
    let k = i >> 1;
    game.legal_moves(player)[k >> 6] >> (k & 63) & 1 == 1
}

/// A uniformly random legal move: `popcount`, one bounded draw, one select over 80 bits.
/// Also not the engine's job — a driver that wants uniform play has the mask and can do this.
fn uniform_move(game: &Game, player: usize, rng: &mut Rng) -> usize {
    let w: [u64; LEGAL_WORDS] = game.legal_moves(player);
    let count = w[0].count_ones() + w[1].count_ones();
    assert!(count > 0, "no legal move for player {player}");
    let mut k = rng.randint(count as u64) as u32;
    for (wi, &word) in w.iter().enumerate() {
        let c = word.count_ones();
        if k < c {
            let mut x = word;
            for _ in 0..k {
                x &= x - 1; // clear the lowest set bit
            }
            return legal_cell((wi << 6) + x.trailing_zeros() as usize);
        }
        k -= c;
    }
    unreachable!("select past the end of the legal set")
}

/// The boards, laid out the way the oracle lays out its batch, so the two can be compared in
/// one assert instead of a loop that stops at the first mismatch.
fn boards_of(games: &[Game]) -> Vec<i8> {
    games.iter().flat_map(|g| g.cells).collect()
}

/// The engine keeps integer scores because every score is a sum of run lengths; the oracle
/// keeps f32 because numpy did. The comparison needs one of them converted.
fn scores_as_f32(games: &[Game]) -> Vec<f32> {
    games
        .iter()
        .flat_map(|g| [g.scores[0] as f32, g.scores[1] as f32])
        .collect()
}

/// The dense masks the oracle produces, rebuilt from the engine's 80-bit legality words.
/// Not something the engine offers — this is the test doing the unpacking, so the two
/// representations can be compared at all.
fn masks_from_bits(games: &[Game]) -> (Vec<f32>, Vec<f32>) {
    let n = games.len();
    let mut m0 = vec![0.0f32; n * HW];
    let mut m1 = vec![0.0f32; n * HW];
    for (g, game) in games.iter().enumerate() {
        for i in 0..HW {
            m0[g * HW + i] = f32::from(legal_at(game, i, 0));
            m1[g * HW + i] = f32::from(legal_at(game, i, 1));
        }
    }
    (m0, m1)
}

/// One uniformly random legal index per game; 0 for finished games, which is never applied.
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

/// One move for every unfinished game, each game sampling its own moves.
///
/// The draw order is not incidental. The oracle samples player 0 for *every* game out of one
/// RNG, then player 1 for every game, so to compare move against move this has to consume the
/// same stream in the same order — hence two passes, even though a single game would
/// naturally draw both of its moves together. A single differing draw sends the two sides
/// down permanently different games, which is what makes divergence loud rather than subtle.
fn sampled_step(games: &mut [Game], d0: &[f32], d1: &[f32], rng: &mut Rng, s: &mut Scratch) {
    let n = games.len();
    let mut i0 = vec![0usize; n];
    let mut i1 = vec![0usize; n];
    for (g, game) in games.iter().enumerate() {
        if !game.finished {
            i0[g] = game.sample_move(&d0[g * HW..][..HW], 0, rng);
        }
    }
    for (g, game) in games.iter().enumerate() {
        if !game.finished {
            i1[g] = game.sample_move(&d1[g * HW..][..HW], 1, rng);
        }
    }
    for (g, game) in games.iter_mut().enumerate() {
        if !game.finished {
            game.action_step(i0[g], i1[g], s);
        }
    }
}

fn game_count(default: usize) -> usize {
    std::env::var("PARITY_GAMES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[test]
fn the_engine_matches_the_oracle_bit_for_bit() {
    let n = game_count(10_000);
    let seed = 0xa1f4_1e5u64;

    let mut refg = Oracle::new(n, seed);
    let mut games: Vec<Game> = (0..n).map(|_| Game::new()).collect();
    let mut scratch = Scratch::new();

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

        // the move the oracle is about to play has to be one the engine agrees is legal
        for (g, game) in games.iter().enumerate() {
            if game.finished {
                continue;
            }
            assert!(legal_at(game, idx0[g] as usize, 0), "g{g} p0 move at step {steps}");
            assert!(legal_at(game, idx1[g] as usize, 1), "g{g} p1 move at step {steps}");
        }

        refg.apply(&idx0, &idx1);
        for (g, game) in games.iter_mut().enumerate() {
            if !game.finished {
                game.action_step(idx0[g] as usize, idx1[g] as usize, &mut scratch);
            }
        }

        assert_eq!(boards_of(&games), refg.boards, "boards diverged at step {steps}");
        assert_eq!(scores_as_f32(&games), refg.scores, "scores diverged at step {steps}");
        assert_eq!(
            games.iter().map(|g| g.move_count as i32).collect::<Vec<_>>(),
            refg.move_counts,
            "move counts at step {steps}"
        );
        assert_eq!(
            games.iter().map(|g| g.finished).collect::<Vec<_>>(),
            refg.finished,
            "finished flags at step {steps}"
        );
        let (m0, m1, _, _) = refg.masks();
        assert_eq!(masks_from_bits(&games), (m0, m1), "legality at step {steps}");
        check_levels_and_scores(&games, &format!("step {steps}"));
        check_legality(&games, &format!("step {steps}"));

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
/// That one feeds both sides an explicit move index so it can control what gets played; it
/// therefore never runs `sample_move` at all. Here both sides pick their own moves from the
/// same distributions and the same RNG seed, so a single differing draw sends the two games
/// down permanently different paths.
///
/// The engine walks set bits in an 80-bit board; the oracle walks 160 squares against a
/// materialized f32 mask. Same order, same accumulation, same RNG consumption — this is what
/// pins that down. The masks agree exactly because a mask entry is 1.0 or 0.0, so
/// `dist * mask` is either `dist` unchanged or an exact zero, and adding exact zeros cannot
/// move an f64 sum.
#[test]
fn the_weighted_sampler_picks_the_same_moves_as_the_oracle_sampler() {
    // Fixed, and deliberately not honouring `PARITY_GAMES`: the step count asserted at the
    // end is the length of the longest game in *this* population, so it is only a constant
    // for a fixed number of games.
    let n: usize = 2_000;
    let seed = 0x5a3_1e5u64;

    // Both distributions are strictly positive on purpose. If every *legal* square carried
    // exactly zero weight, both samplers would fall back — but to different things. The
    // oracle takes a uniform draw over all 160 squares, which can land on a square that is
    // not playable at all; it absorbs that because it rescores the board from scratch.
    // `Game::sample_move` deliberately falls back to a uniform draw over the *legal* squares
    // instead, which is the sane behaviour and not bit-compatible with the quirk. A softmax
    // policy never produces the situation.
    let mut rng = Rng::new(seed);
    let d0: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
    let d1: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();

    let mut refg = Oracle::new(n, seed);
    let mut games: Vec<Game> = (0..n).map(|_| Game::new()).collect();
    let mut scratch = Scratch::new();
    // the engine's own RNG, seeded identically but stepped separately — the two must stay in
    // lockstep by consuming the same draws, not by sharing a generator
    let mut engine_rng = Rng::new(seed);
    let mut steps = 0usize;

    while !refg.finished.iter().all(|&f| f) {
        refg.distribution_step(&d0, &d1);
        sampled_step(&mut games, &d0, &d1, &mut engine_rng, &mut scratch);
        assert_eq!(boards_of(&games), refg.boards, "sampler diverged at step {steps}");
        assert_eq!(scores_as_f32(&games), refg.scores, "scores diverged at step {steps}");
        check_levels_and_scores(&games, &format!("step {steps}"));
        check_legality(&games, &format!("step {steps}"));
        steps += 1;
        assert!(steps < 200, "rollout did not terminate");
    }
    assert_eq!(steps, 44, "{n} games should take 44 steps to all finish, took {steps}");
}

/// Every externally visible call, checked against the oracle on many game states.
///
/// The two tests above cover the engine's *behaviour* — that playing a game produces the same
/// board and the same score. This one covers its *surface*: for each public call, on each of
/// a rollout's states, does the engine hand back what the oracle says it should.
///
/// `legal_moves` is the one call returning a different representation rather than different
/// data: it packs into 80 bits what the oracle spreads over 160 floats. So it is checked for
/// logical equivalence against the board itself, and the oracle's dense masks are separately
/// checked to be exactly that set intersected with the first-move half rule.
#[test]
fn every_public_call_agrees_with_the_oracle() {
    let n: usize = 512;
    let seed = 0xc0de_5eedu64;
    let half = WIDTH / 2;

    let mut rng = Rng::new(seed);
    let d0: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
    let d1: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
    let uniform = vec![1.0f32; HW];

    let mut refg = Oracle::new(n, seed);
    let mut games: Vec<Game> = (0..n).map(|_| Game::new()).collect();
    let mut scratch = Scratch::new();
    let mut engine_rng = Rng::new(seed);
    let mut draw_rng = Rng::new(seed ^ 0xdd);
    let mut steps = 0usize;

    loop {
        assert_eq!(boards_of(&games), refg.boards, "boards at step {steps}");
        assert_eq!(scores_as_f32(&games), refg.scores, "scores at step {steps}");

        let (want0, want1, wc0, wc1) = refg.masks();
        for (g, game) in games.iter().enumerate() {
            assert_eq!(game.move_count as i32, refg.move_counts[g], "move_count g{g}");
            assert_eq!(game.finished, refg.finished[g], "finished g{g}");

            // legal_moves and legal_count, cell by cell, against the board and against the
            // oracle's masks
            let (a, b) = (game.legal_moves(0), game.legal_moves(1));
            let bits = [a[0] | b[0], a[1] | b[1]];
            let first = refg.move_counts[g] == 0;
            let (mut n0, mut n1) = (0.0f32, 0.0f32);
            for i in 0..HW {
                let k = i >> 1;
                let set = i % 2 == (i / WIDTH) % 2 && bits[k >> 6] >> (k & 63) & 1 == 1;
                let playable = refg.boards[g * HW + i] == PLAYABLE_SQUARE;
                assert_eq!(set, playable, "legal_moves g{g} cell {i} at step {steps}");
                let c = i % WIDTH;
                let want_0 = playable && !(first && c >= half);
                let want_1 = playable && !(first && c < half);
                assert_eq!(want0[g * HW + i] == 1.0, want_0, "mask0 g{g} cell {i}");
                assert_eq!(want1[g * HW + i] == 1.0, want_1, "mask1 g{g} cell {i}");
                assert_eq!(legal_at(game, i, 0), want_0, "legal_moves g{g} p0 cell {i}");
                assert_eq!(legal_at(game, i, 1), want_1, "legal_moves g{g} p1 cell {i}");
                n0 += f32::from(want_0);
                n1 += f32::from(want_1);
            }
            assert_eq!((wc0[g], wc1[g]), (n0, n1), "legal counts g{g} at step {steps}");
            assert_eq!(game.legal_count(0) as f32, n0, "legal_count g{g} p0");
            assert_eq!(game.legal_count(1) as f32, n1, "legal_count g{g} p1");

            // both samplers have to land inside the legal set, and inside this player's half
            // on the opening move
            if !game.finished {
                for player in [0usize, 1usize] {
                    let a = uniform_move(game, player, &mut draw_rng);
                    let b = game.sample_move(&uniform, player, &mut draw_rng);
                    assert!(legal_at(game, a, player), "uniform pick g{g} p{player} illegal");
                    assert!(legal_at(game, b, player), "sample_move g{g} p{player} illegal");
                }
            }
        }

        // from_cells is handed the board and nothing else, so everything else has to come
        // back from it — including the two flags it is no longer told. `move_count` is the
        // one thing that cannot: the board records whether any move was made, not how many,
        // which is all the opening-move rule ever asks.
        for (g, game) in games.iter().enumerate() {
            let mut adopted = Game::from_cells(game.cells, &mut scratch);
            assert_eq!(
                adopted.move_count,
                u32::from(game.move_count > 0),
                "from_cells g{g} first-move flag at step {steps}"
            );
            adopted.move_count = game.move_count;
            assert_eq!(adopted, *game, "from_cells g{g} differs at step {steps}");
        }

        if refg.finished.iter().all(|&f| f) {
            break;
        }
        refg.distribution_step(&d0, &d1);
        sampled_step(&mut games, &d0, &d1, &mut engine_rng, &mut scratch);
        steps += 1;
        assert!(steps < 200, "rollout did not terminate");
    }

    println!("{n} games, {steps} steps, every public call compared at every step");
}

/// Forking a position must produce a game that neither feels nor is felt by its parent.
///
/// This is the operation a tree search leans on hardest, and the way it breaks is quiet: a
/// clone that shared something would look right until two branches were played apart. The
/// derived `Clone` cannot drop a field, but the [`Scratch`] the two games take turns
/// borrowing could carry state from one into the other — so the parent and the fork are
/// played apart through *one* workspace, in alternating order, and both are held to the
/// oracle's scorer the whole way.
#[test]
fn a_fork_is_independent_of_the_game_it_came_from() {
    let n = 256;
    let seed = 0xf0_1ced_u64;
    let mut rng = Rng::new(seed);
    let mut scratch = Scratch::new();

    let mut parents: Vec<Game> = (0..n).map(|_| Game::new()).collect();
    for _ in 0..12 {
        for game in parents.iter_mut() {
            let (i0, i1) = (uniform_move(game, 0, &mut rng), uniform_move(game, 1, &mut rng));
            game.action_step(i0, i1, &mut scratch);
        }
    }

    // a fork is a memcpy, and equal to what it came from
    let mut forks: Vec<Game> = parents.clone();
    assert_eq!(forks, parents, "a fresh fork differs from its parent");
    // a third copy, taken at the same moment and then never touched again, plus the boards
    // it held at that moment recorded outside any Game
    let frozen = parents.clone();
    let fork_point: Vec<[i8; HW]> = parents.iter().map(|g| g.cells).collect();

    // play them apart, alternating so the shared scratch is handed back and forth mid-game
    let mut steps = 0;
    while parents.iter().any(|g| !g.finished) || forks.iter().any(|g| !g.finished) {
        for g in 0..n {
            for side in [&mut parents, &mut forks] {
                let game = &mut side[g];
                if game.finished {
                    continue;
                }
                let (i0, i1) = (uniform_move(game, 0, &mut rng), uniform_move(game, 1, &mut rng));
                game.action_step(i0, i1, &mut scratch);
            }
        }
        check_levels_and_scores(&parents, &format!("parent, step {steps}"));
        check_levels_and_scores(&forks, &format!("fork, step {steps}"));
        check_legality(&parents, &format!("parent, step {steps}"));
        check_legality(&forks, &format!("fork, step {steps}"));
        steps += 1;
        assert!(steps < 200, "rollout did not terminate");
    }
    assert!(steps > 10, "the games finished suspiciously fast");

    // they were played apart, so they must have ended apart — otherwise this test would pass
    // just as happily if `action_step` did nothing
    assert!(
        (0..n).any(|g| parents[g].cells != forks[g].cells),
        "a parent and its fork played independently ended up identical"
    );
    // and the copy nobody played must still hold the position it was forked from — checked
    // against boards recorded outside any Game, so nothing that happened afterwards, in
    // either branch or in the workspace they shared, wrote through into it
    for g in 0..n {
        assert_eq!(frozen[g].cells, fork_point[g], "an untouched fork was written into");
        assert_ne!(frozen[g].cells, parents[g].cells, "the parent never left the fork point");
    }
}

/// The engine and the oracle define the square encoding and the RNG separately — on purpose,
/// so that agreeing with each other means something. Separate definitions of the same
/// constant are exactly the kind of thing that drifts, so this pins them together. The
/// constants are a compile-time check and cost nothing to keep.
#[test]
fn the_engine_and_the_oracle_still_agree_on_the_encoding_and_the_rng() {
    use alpha_lines_game::game as eng;

    const _: () = assert!(eng::NON_PLAYABLE_SQUARE == oracle::NON_PLAYABLE_SQUARE);
    const _: () = assert!(eng::PLAYABLE_SQUARE == oracle::PLAYABLE_SQUARE);
    const _: () = assert!(eng::REMOVED_SQUARE == oracle::REMOVED_SQUARE);
    const _: () = assert!(eng::PLAYER_0_MARK == oracle::PLAYER_0_MARK);
    const _: () = assert!(eng::PLAYER_1_MARK == oracle::PLAYER_1_MARK);

    // and the two RNGs must be the same generator, or the samplers would diverge
    let (mut a, mut b) = (eng::Rng::new(0xabc), oracle::Rng::new(0xabc));
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
    let mut scratch = Scratch::new();

    let mut highest = 0u8;
    for _ in 0..n {
        let mut game = Game::new();
        while !game.finished {
            let (i0, i1) = (uniform_move(&game, 0, &mut rng), uniform_move(&game, 1, &mut rng));
            game.action_step(i0, i1, &mut scratch);
            for &l in &game.levels {
                if l != INF && l > highest {
                    highest = l;
                }
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
