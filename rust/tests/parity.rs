//! The engine against the oracle, bit for bit, at every move, in every observable field.
//!
//! The oracle (`tests/oracle`) is a port of `main/game_kernels.py`, itself verified against
//! the Python by `xcheck/xcheck.py`, so parity here is parity with the Python transitively.
//!
//! It is batch-shaped because the numba kernels were; the engine is not, so the tests hold a
//! `Vec<Game>` against one oracle batch. The games share one `Scratch`, which is what would
//! expose workspace state leaking from one board into the next.
//!
//! Checks the oracle cannot make on its own: levels are re-derived from scratch every step,
//! the 80-bit masks are unpacked here from the documented layout rather than asked for, and
//! collisions are nudged up because they are what drives the slow removal path.
//!
//! `cargo test --release`; also worth running in debug, where `action_step`'s legality
//! `debug_assert`s are live. `PARITY_GAMES` overrides the game count.

mod oracle;

use alpha_lines_game::game::{
    legal_cell, rebuild_levels, Rng, Scratch, HEIGHT, HW, INF, LEGAL_WORDS, MAX_LEVEL,
    PLAYABLE_SQUARE, PLAYER_0_MARK, PLAYER_1_MARK, WIDTH,
};
use alpha_lines_game::Game;
use oracle::{apply_and_score_kernel, legal_masks_kernel, sample_move_kernel, score_player};

/// The oracle as a batch of games: four arrays and the kernels that move them.
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


/// Every level must equal a from-scratch BFS, and every running score the oracle's scorer.
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

/// Every mask bit must match the board, and `finished` whether any bit is left.
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

/// Does `player`'s mask hold the bit for board index `i`? Unpacked here from the documented
/// layout, not asked of the engine, which is what makes agreement evidence.
fn legal_at(game: &Game, i: usize, player: usize) -> bool {
    if i >= HW || legal_cell(i >> 1) != i {
        return false; // not a playable-parity square, so it holds no bit of its own
    }
    let k = i >> 1;
    game.legal_moves(player)[k >> 6] >> (k & 63) & 1 == 1
}

/// A uniformly random legal move: `popcount`, one bounded draw, one select over 80 bits.
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

/// The boards laid out the way the oracle lays out its batch.
fn boards_of(games: &[Game]) -> Vec<i8> {
    games.iter().flat_map(|g| g.cells).collect()
}

/// The engine scores in integers, the oracle in f32 because numpy did.
fn scores_as_f32(games: &[Game]) -> Vec<f32> {
    games
        .iter()
        .flat_map(|g| [g.scores[0] as f32, g.scores[1] as f32])
        .collect()
}

/// The oracle's dense masks, rebuilt from the engine's 80-bit ones.
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

/// One uniformly random legal index per game, from the oracle's mask.
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

        // every sixth step, steer the even-numbered games onto their opponent's square where
        // it is legal for both; the odd-numbered ones are left as ordinary random play
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

    // if either half stops reaching the slow paths, fail loudly rather than test nothing
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

/// `distribution_step` against the oracle's sampler, both picking their own moves from the
/// same distribution and the same RNG seed, so one differing draw diverges permanently.
///
/// The oracle is run one game at a time. It samples player 0 for every game in its batch,
/// then player 1 for every game; a single `Game` draws both of its own moves together, so
/// only a batch of one consumes the RNG in the same order.
///
/// The distributions are strictly positive on purpose: on an all-zero one the two fall back
/// to different things, the oracle drawing uniformly over all 160 squares — a numba quirk a
/// softmax policy never triggers.
#[test]
fn distribution_step_picks_the_same_moves_as_the_oracle_sampler() {
    let n = game_count(2_000);
    let seed = 0x5a3_1e5u64;
    let mut scratch = Scratch::new();
    let mut longest = 0usize;

    for g in 0..n {
        let s = seed.wrapping_add(g as u64);
        let mut rng = Rng::new(s);
        let d0: Vec<f32> = (0..HW).map(|_| rng.random() as f32).collect();
        let d1: Vec<f32> = (0..HW).map(|_| rng.random() as f32).collect();

        let mut refg = Oracle::new(1, s);
        let mut game = Game::new();
        let mut engine_rng = Rng::new(s);
        let mut steps = 0usize;

        while !refg.finished[0] {
            refg.distribution_step(&d0, &d1);
            game.distribution_step(&d0, &d1, &mut engine_rng, &mut scratch);
            assert_eq!(&game.cells[..], &refg.boards[..], "g{g} board at step {steps}");
            assert_eq!(
                [game.scores[0] as f32, game.scores[1] as f32],
                [refg.scores[0], refg.scores[1]],
                "g{g} scores at step {steps}"
            );
            assert_eq!(game.finished, refg.finished[0], "g{g} finished at step {steps}");
            steps += 1;
            assert!(steps < 200, "rollout did not terminate");
        }
        longest = longest.max(steps);
        check_levels_and_scores(std::slice::from_ref(&game), &format!("game {g}"));
        check_legality(std::slice::from_ref(&game), &format!("game {g}"));
    }
    assert!(longest >= 40, "longest game was only {longest} steps");
}

/// Every public call against the oracle, on every state of a rollout. The tests above cover
/// behaviour; this one covers the surface.
#[test]
fn every_public_call_agrees_with_the_oracle() {
    let n: usize = 512;
    let seed = 0xc0de_5eedu64;
    let half = WIDTH / 2;

    let mut refg = Oracle::new(n, seed);
    let mut games: Vec<Game> = (0..n).map(|_| Game::new()).collect();
    let mut scratch = Scratch::new();
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

            // a move drawn from the mask has to be inside it
            if !game.finished {
                for player in [0usize, 1usize] {
                    let m = uniform_move(game, player, &mut draw_rng);
                    assert!(legal_at(game, m, player), "drawn move g{g} p{player} illegal");
                }
            }
        }

        // from_cells gets the board and nothing else, so everything else has to come back
        // from it — except move_count, which the board records only as 0-or-1
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
        // explicit moves, so this test does not depend on RNG draw order
        let idx0 = pick(&want0, &refg.finished, n, &mut draw_rng);
        let idx1 = pick(&want1, &refg.finished, n, &mut draw_rng);
        refg.apply(&idx0, &idx1);
        for (g, game) in games.iter_mut().enumerate() {
            if !game.finished {
                game.action_step(idx0[g] as usize, idx1[g] as usize, &mut scratch);
            }
        }
        steps += 1;
        assert!(steps < 200, "rollout did not terminate");
    }

    println!("{n} games, {steps} steps, every public call compared at every step");
}

/// A fork must neither feel nor be felt by its parent. The derived `Clone` cannot drop a
/// field, but the shared [`Scratch`] could carry state from one into the other — so the two
/// are played apart through one workspace, alternating, and both held to the oracle's scorer.
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
    // a third copy never touched again, plus its boards recorded outside any Game
    let frozen = parents.clone();
    let fork_point: Vec<[i8; HW]> = parents.iter().map(|g| g.cells).collect();

    // alternating, so the shared scratch is handed back and forth mid-game
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

    // played apart, so they must have ended apart
    assert!(
        (0..n).any(|g| parents[g].cells != forks[g].cells),
        "a parent and its fork played independently ended up identical"
    );
    // and the untouched copy must still hold the position it was forked from
    for g in 0..n {
        assert_eq!(frozen[g].cells, fork_point[g], "an untouched fork was written into");
        assert_ne!(frozen[g].cells, parents[g].cells, "the parent never left the fork point");
    }
}

/// The engine and the oracle define the encoding and the RNG separately, on purpose. This
/// pins the two definitions together so they cannot drift.
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

/// `MAX_LEVEL` is a bound derived on paper; this checks it against a lot of real games. If
/// the argument were wrong, cells would be declared unreachable that are not.
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
    // and the run has to reach deep levels, or the check above proves nothing
    assert!(highest > 5, "highest level was only {highest}; this run is not testing the bound");
    println!("highest real level over {n} games: {highest} (cap {MAX_LEVEL})");
}
