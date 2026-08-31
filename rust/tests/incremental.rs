//! Tests for the level-based incremental scorer.
//!
//! Two kinds. First, targeted cases for the situations the design turns on: a severed
//! cycle, a line anchored at both ends cut in the middle, a dead blob revived, collisions.
//! Second, differential rollouts against the reference implementation, which is itself
//! verified byte-for-byte against `main/game.py`.

use alpha_lines_game::config::{HEIGHT, WIDTH};
use alpha_lines_game::game_kernels::*;
use alpha_lines_game::incremental::{
    apply_move, check_invariants, insert, remove, Scratch, HW, INF, MAX_LEVEL,
};
use alpha_lines_game::rng::Rng;
use alpha_lines_game::{BatchedLinesGame, IncrementalGame};

/// A single board with its levels and running score, for the low-level tests.
struct Board {
    cells: Vec<i8>,
    level: Vec<u8>,
    score: Vec<i32>,
    scratch: Scratch,
}

impl Board {
    fn new() -> Self {
        let mut cells = vec![NON_PLAYABLE_SQUARE; HW];
        for r in 0..HEIGHT {
            for c in 0..WIDTH {
                if (r + c) % 2 == 0 {
                    cells[r * WIDTH + c] = PLAYABLE_SQUARE;
                }
            }
        }
        Board { cells, level: vec![INF; HW], score: vec![0; 2], scratch: Scratch::new() }
    }

    fn put(&mut self, r: usize, c: usize, p: i8) -> &mut Self {
        insert(&mut self.cells, &mut self.level, &mut self.score, r * WIDTH + c, p, &mut self.scratch);
        self.check();
        self
    }

    fn take(&mut self, r: usize, c: usize) -> &mut Self {
        remove(&mut self.cells, &mut self.level, &mut self.score, r * WIDTH + c, &mut self.scratch);
        self.check();
        self
    }

    fn collide(&mut self, r: usize, c: usize) -> &mut Self {
        apply_move(&mut self.cells, &mut self.level, &mut self.score, (r, c), (r, c), &mut self.scratch);
        self.check();
        self
    }

    fn check(&self) {
        check_invariants(&self.cells, &self.level, &self.score).unwrap();
    }

    fn level_at(&self, r: usize, c: usize) -> u8 {
        self.level[r * WIDTH + c]
    }

    fn score_of(&self, p: i8) -> i32 {
        self.score[if p == PLAYER_0_MARK { 0 } else { 1 }]
    }
}

const P0: i8 = PLAYER_0_MARK;
const P1: i8 = PLAYER_1_MARK;

// ------------------------------------------------------------------------ level basics

#[test]
fn a_border_mark_is_level_zero_and_an_isolated_interior_mark_is_unreachable() {
    let mut b = Board::new();
    b.put(0, 4, P0);
    assert_eq!(b.level_at(0, 4), 0);
    b.put(4, 4, P0);
    assert_eq!(b.level_at(4, 4), INF, "an interior mark with no route to the border is dead");
    assert_eq!(b.score_of(P0), 0);
}

#[test]
fn levels_count_steps_to_the_nearest_border() {
    let mut b = Board::new();
    for k in 0..5 {
        b.put(k, k, P0);
    }
    for k in 0..5 {
        assert_eq!(b.level_at(k, k), k as u8, "cell ({k},{k})");
    }
}

#[test]
fn a_new_anchor_pulls_the_levels_down_behind_it() {
    // A chain anchored only at (0,0), so levels count up along it. Anchoring the far end
    // gives every cell in the tail a shorter route, and those levels have to fall.
    // Nothing scores differently — this is the pure-bookkeeping case.
    let mut b = Board::new();
    for k in 0..HEIGHT - 1 {
        b.put(k, k, P0);
    }
    for k in 0..HEIGHT - 1 {
        assert_eq!(b.level_at(k, k), k as u8);
    }
    assert_eq!(b.score_of(P0), (HEIGHT - 1) as i32);

    b.put(HEIGHT - 1, HEIGHT - 1, P0); // (9,9) is on the bottom border
    assert_eq!(b.level_at(9, 9), 0);
    assert_eq!(b.level_at(8, 8), 1, "the new anchor beats the old route of length 8");
    assert_eq!(b.level_at(7, 7), 2);
    assert_eq!(b.level_at(6, 6), 3);
    assert_eq!(b.level_at(5, 5), 4);
    assert_eq!(b.level_at(4, 4), 4, "the old route was already this good, so it stands");
    assert_eq!(b.score_of(P0), HEIGHT as i32, "the run just got one cell longer");
}

#[test]
fn levels_never_exceed_the_cap() {
    let mut b = Board::new();
    for k in 0..8 {
        b.put(k, k, P0);
    }
    for i in 0..HW {
        if b.cells[i] == P0 {
            assert!(b.level[i] <= MAX_LEVEL || b.level[i] == INF);
        }
    }
}

// ------------------------------------------------------------------------ scoring rules

#[test]
fn a_lone_mark_scores_nothing_and_a_pair_scores_two() {
    let mut b = Board::new();
    b.put(0, 0, P0);
    assert_eq!(b.score_of(P0), 0);
    b.put(1, 1, P0);
    assert_eq!(b.score_of(P0), 2, "a reachable run of 2 scores its full length");
}

#[test]
fn filling_a_gap_between_two_lone_marks_scores_three() {
    let mut b = Board::new();
    b.put(0, 0, P0); // border anchor
    b.put(2, 2, P0);
    assert_eq!(b.score_of(P0), 0, "not adjacent yet");
    b.put(1, 1, P0);
    assert_eq!(b.score_of(P0), 3);
}

#[test]
fn a_cell_in_two_qualifying_runs_is_counted_twice() {
    // an X centred at (1,1): both diagonals through it qualify, and that double count is
    // the whole point of the scoring rule
    let mut b = Board::new();
    b.put(0, 0, P0).put(1, 1, P0).put(2, 2, P0).put(0, 2, P0).put(2, 0, P0);
    assert_eq!(b.score_of(P0), 6);
}

#[test]
fn an_unreachable_run_scores_nothing_until_it_is_connected() {
    let mut b = Board::new();
    b.put(4, 4, P0).put(5, 5, P0).put(6, 6, P0);
    assert_eq!(b.score_of(P0), 0, "an interior blob has no route to the border");
    for i in [4, 5, 6] {
        assert_eq!(b.level_at(i, i), INF);
    }
}

// ------------------------------------------------------------- the cases the design turns on

#[test]
fn reviving_a_dead_blob_scores_the_whole_thing_at_once() {
    let mut b = Board::new();
    // build the chain inwards-out, so every cell is dead as it is placed
    for k in (1..7).rev() {
        b.put(k, k, P0);
        assert_eq!(b.score_of(P0), 0, "still no route to the border");
    }
    // the border cell wakes all six at once
    b.put(0, 0, P0);
    assert_eq!(b.score_of(P0), 7, "a run of 7, now reachable");
    for k in 0..7 {
        assert_eq!(b.level_at(k, k), k as u8);
    }
}

#[test]
fn cutting_a_line_anchored_at_both_ends_keeps_both_halves_alive() {
    // the case a plain reachable bit cannot answer without a full flood fill
    let mut b = Board::new();
    for k in 0..HEIGHT {
        b.put(k, k, P0); // (0,0) and (9,9) are both on the border
    }
    assert_eq!(b.score_of(P0), HEIGHT as i32);
    assert_eq!(b.level_at(4, 4), 4);
    assert_eq!(b.level_at(5, 5), 4, "level is distance to the *nearest* border");

    b.take(4, 4);
    // two runs of 4 and 5, both still anchored
    assert_eq!(b.score_of(P0), 9);
    for k in 0..HEIGHT {
        if k != 4 {
            assert_ne!(b.level_at(k, k), INF, "({k},{k}) still has a route");
        }
    }
}

#[test]
fn cutting_a_line_anchored_at_one_end_kills_the_far_half() {
    let mut b = Board::new();
    for k in 0..6 {
        b.put(k, k, P0); // only (0,0) is on the border
    }
    assert_eq!(b.score_of(P0), 6);

    b.take(2, 2);
    assert_eq!(b.score_of(P0), 2, "only the anchored run of 2 still counts");
    assert_eq!(b.level_at(0, 0), 0);
    assert_eq!(b.level_at(1, 1), 1);
    for k in 3..6 {
        assert_eq!(b.level_at(k, k), INF, "({k},{k}) was cut off");
    }
}

#[test]
fn a_severed_cycle_climbs_to_unreachable_without_any_cycle_detection() {
    // the diamond from the design discussion: A on the border, B and D adjacent to A,
    // C adjacent to both B and D. Removing A leaves B, C, D justifying each other.
    let (a, bb, c, d) = ((0, 4), (1, 5), (2, 4), (1, 3));
    let mut b = Board::new();
    b.put(a.0, a.1, P0).put(bb.0, bb.1, P0).put(c.0, c.1, P0).put(d.0, d.1, P0);

    assert_eq!(b.level_at(a.0, a.1), 0);
    assert_eq!(b.level_at(bb.0, bb.1), 1);
    assert_eq!(b.level_at(d.0, d.1), 1);
    assert_eq!(b.level_at(c.0, c.1), 2);
    assert_eq!(b.score_of(P0), 8, "four runs of 2, two per diagonal family");

    b.take(a.0, a.1);
    assert_eq!(b.score_of(P0), 0, "nothing in the loop reaches the border any more");
    for (r, cc) in [bb, c, d] {
        assert_eq!(b.level_at(r, cc), INF, "({r},{cc}) should have climbed to INF");
    }
}

#[test]
fn removing_a_cell_nobody_routed_through_takes_the_local_path() {
    // (3,3) hangs off the side of an anchored chain; nothing routes through it
    let mut b = Board::new();
    b.put(0, 0, P0).put(1, 1, P0).put(2, 2, P0).put(3, 3, P0);
    assert_eq!(b.score_of(P0), 4);
    b.take(3, 3);
    assert_eq!(b.score_of(P0), 3);
    assert_eq!(b.level_at(2, 2), 2, "the survivors keep their levels");
}

// --------------------------------------------------------------------------- collisions

#[test]
fn a_collision_clears_the_centre_and_its_four_diagonal_neighbours() {
    let mut b = Board::new();
    b.put(3, 3, P0).put(5, 5, P1);
    b.collide(4, 4);
    assert_eq!(b.cells[4 * WIDTH + 4], REMOVED_SQUARE);
    for (r, c) in [(3, 3), (3, 5), (5, 3), (5, 5)] {
        assert_eq!(b.cells[r * WIDTH + c], REMOVED_SQUARE, "({r},{c})");
    }
}

#[test]
fn a_collision_can_cut_both_players_at_once() {
    let mut b = Board::new();
    // player 0 runs down the main diagonal, player 1 down an anti-diagonal
    b.put(0, 0, P0).put(1, 1, P0).put(2, 2, P0).put(3, 3, P0);
    b.put(0, 8, P1).put(1, 7, P1).put(2, 6, P1).put(3, 5, P1);
    assert_eq!(b.score_of(P0), 4);
    assert_eq!(b.score_of(P1), 4);

    // (4,4) is empty and playable; its diagonal neighbours include (3,3) and (3,5),
    // one cell from each player
    b.collide(4, 4);
    assert_eq!(b.score_of(P0), 3, "player 0 lost the tail of its run");
    assert_eq!(b.score_of(P1), 3, "player 1 lost the tail of its run");
}

#[test]
fn a_corner_collision_stays_in_bounds() {
    let mut b = Board::new();
    b.put(1, 1, P0);
    b.collide(0, 0);
    assert_eq!(b.cells[0], REMOVED_SQUARE);
    assert_eq!(b.cells[1 * WIDTH + 1], REMOVED_SQUARE);
}

// -------------------------------------------------------------- differential vs reference

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

/// Drives both implementations through the same random legal moves and compares state
/// after every move. Returns how many collisions occurred, so a caller can assert the
/// removal paths were actually exercised.
fn differential_rollout(n: usize, seed: u64, check_every_step: bool) -> usize {
    let mut refg = BatchedLinesGame::new(n, seed);
    let mut inc = IncrementalGame::new(n, seed);
    let mut rng = Rng::new(seed ^ 0x9e37_79b9);
    let mut collisions = 0usize;
    let mut steps = 0usize;

    while !refg.finished.iter().all(|&f| f) {
        let (m0, m1, _, _) = refg.get_legal_masks();
        let idx0 = pick(&m0, &refg.finished, n, &mut rng);
        let idx1 = pick(&m1, &refg.finished, n, &mut rng);
        for g in 0..n {
            if !refg.finished[g] && idx0[g] == idx1[g] {
                collisions += 1;
            }
        }

        refg.action_step(&idx0, &idx1).unwrap();
        inc.action_step(&idx0, &idx1).unwrap();

        assert_eq!(inc.boards, refg.boards, "seed {seed} step {steps}: boards diverged");
        assert_eq!(inc.scores_f32(), refg.scores, "seed {seed} step {steps}: scores diverged");
        assert_eq!(inc.move_counts, refg.move_counts, "seed {seed} step {steps}");
        assert_eq!(inc.finished, refg.finished, "seed {seed} step {steps}");
        if check_every_step {
            inc.check_invariants()
                .unwrap_or_else(|e| panic!("seed {seed} step {steps}: {e}"));
        }

        steps += 1;
        assert!(steps < 200, "rollout did not terminate");
    }
    collisions
}

#[test]
fn matches_the_reference_over_random_rollouts() {
    let mut collisions = 0;
    for seed in 0..12u64 {
        collisions += differential_rollout(128, seed, false);
    }
    assert!(collisions > 100, "expected the collision path to be exercised, saw {collisions}");
}

#[test]
fn holds_its_invariants_after_every_move_of_a_full_rollout() {
    let mut collisions = 0;
    for seed in 100..103u64 {
        collisions += differential_rollout(24, seed, true);
    }
    assert!(collisions > 0, "expected at least one collision, saw {collisions}");
}

/// Collisions are rare under random play, so this forces them: both players are handed the
/// same distribution, which makes them pick the same square far more often.
#[test]
fn matches_the_reference_under_heavy_collision_pressure() {
    for seed in 0..6u64 {
        let n = 64;
        let mut refg = BatchedLinesGame::new(n, seed);
        let mut inc = IncrementalGame::new(n, seed);
        let mut rng = Rng::new(seed ^ 0xfeed);
        let mut collisions = 0usize;
        let mut steps = 0usize;

        while !refg.finished.iter().all(|&f| f) {
            let (m0, m1, _, _) = refg.get_legal_masks();
            let a = pick(&m0, &refg.finished, n, &mut rng);
            let b = pick(&m1, &refg.finished, n, &mut rng);
            // Every other game is steered into a collision, but only where the square is
            // legal for both players — on the opening move the halves are disjoint, so a
            // collision is impossible and the independently sampled pair stands.
            let mut idx0 = a.clone();
            let mut idx1 = b;
            for g in (0..n).step_by(2) {
                if !refg.finished[g] && m1[g * HW + a[g] as usize] == 1.0 {
                    idx1[g] = a[g];
                }
            }
            for g in 0..n {
                if !refg.finished[g] && idx0[g] == idx1[g] {
                    collisions += 1;
                }
            }

            refg.action_step(&idx0, &idx1).unwrap();
            inc.action_step(&idx0, &idx1).unwrap();
            assert_eq!(inc.boards, refg.boards, "seed {seed} step {steps}");
            assert_eq!(inc.scores_f32(), refg.scores, "seed {seed} step {steps}");
            inc.check_invariants().unwrap();

            steps += 1;
            assert!(steps < 200);
        }
        assert!(collisions > 50, "seed {seed}: only {collisions} collisions");
    }
}

#[test]
fn distribution_step_matches_a_from_scratch_rescore() {
    let n = 96;
    let mut inc = IncrementalGame::new(n, 7);
    let mut rng = Rng::new(11);
    let dist: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
    let mut steps = 0;
    while !inc.finished.iter().all(|&f| f) {
        inc.distribution_step(&dist, &dist);
        steps += 1;
        assert!(steps < 200);
    }
    inc.check_invariants().unwrap();
    for g in 0..n {
        let cells = &inc.boards[g * HW..(g + 1) * HW];
        assert_eq!(inc.scores[g * 2], score_player(cells, 0, P0, HEIGHT, WIDTH) as i32);
        assert_eq!(inc.scores[g * 2 + 1], score_player(cells, 0, P1, HEIGHT, WIDTH) as i32);
    }
}

#[test]
fn from_state_reconstructs_levels_and_scores() {
    // play a while, then rebuild from the raw board alone and check nothing was lost
    let n = 32;
    let mut inc = IncrementalGame::new(n, 3);
    let mut rng = Rng::new(5);
    let dist: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
    for _ in 0..15 {
        inc.distribution_step(&dist, &dist);
    }
    let rebuilt = IncrementalGame::from_state(
        inc.boards.clone(),
        inc.move_counts.clone(),
        inc.finished.clone(),
        0,
    );
    assert_eq!(rebuilt.levels, inc.levels);
    assert_eq!(rebuilt.scores, inc.scores);
    rebuilt.check_invariants().unwrap();
}

#[test]
fn the_live_list_sampler_picks_the_same_moves_as_the_reference_sampler() {
    // The engine's sampler no longer reads a mask, but it must still walk the candidate
    // squares in row-major order and consume the RNG identically, so a seeded rollout has
    // to reproduce the reference implementation move for move.
    for seed in 0..6u64 {
        let n = 96;
        let mut refg = BatchedLinesGame::new(n, seed);
        let mut inc = IncrementalGame::new(n, seed);
        let mut rng = Rng::new(seed ^ 0x5171);
        let d0: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
        let d1: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();

        let mut steps = 0;
        while !refg.finished.iter().all(|&f| f) {
            refg.distribution_step(&d0, &d1);
            inc.distribution_step(&d0, &d1);
            assert_eq!(inc.boards, refg.boards, "seed {seed} step {steps}: different move chosen");
            assert_eq!(inc.scores_f32(), refg.scores, "seed {seed} step {steps}");
            assert_eq!(inc.finished, refg.finished, "seed {seed} step {steps}");
            steps += 1;
            assert!(steps < 200);
        }
        inc.check_invariants().unwrap();
    }
}

#[test]
fn the_live_list_survives_collisions_and_stays_row_major() {
    let n = 48;
    let mut inc = IncrementalGame::new(n, 21);
    let mut rng = Rng::new(99);
    // one shared distribution makes both players collide far more often
    let dist: Vec<f32> = (0..n * HW).map(|_| rng.random() as f32).collect();
    let mut steps = 0;
    while !inc.finished.iter().all(|&f| f) {
        inc.distribution_step(&dist, &dist);
        // check_invariants asserts the live list equals the playable squares, in order
        inc.check_invariants().unwrap();
        steps += 1;
        assert!(steps < 200);
    }
}
