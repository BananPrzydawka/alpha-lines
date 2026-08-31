//! Unit tests for the kernel layer: the scoring rule, the legal-move masks, the sampler,
//! and move application. These pin down behaviour that the Python cross-check can only
//! observe indirectly (or not at all, in the sampler's case).

use alpha_lines_game::config::{HEIGHT, WIDTH};
use alpha_lines_game::game_kernels::*;
use alpha_lines_game::rng::Rng;

const HW: usize = HEIGHT * WIDTH;

/// Builds a one-game board from a picture, `.` = playable, ` ` = non-playable,
/// `#` = removed, `A` = player 0, `B` = player 1. Rows are given top to bottom.
fn board_from(rows: &[&str]) -> Vec<i8> {
    assert_eq!(rows.len(), HEIGHT);
    let mut b = vec![NON_PLAYABLE_SQUARE; HW];
    for (r, row) in rows.iter().enumerate() {
        let chars: Vec<char> = row.chars().collect();
        assert_eq!(chars.len(), WIDTH, "row {r} has wrong width");
        for (c, ch) in chars.iter().enumerate() {
            b[r * WIDTH + c] = match ch {
                ' ' => NON_PLAYABLE_SQUARE,
                '.' => PLAYABLE_SQUARE,
                '#' => REMOVED_SQUARE,
                'A' => PLAYER_0_MARK,
                'B' => PLAYER_1_MARK,
                _ => panic!("bad char {ch:?}"),
            };
        }
    }
    b
}

fn empty_rows() -> Vec<String> {
    (0..HEIGHT)
        .map(|r| (0..WIDTH).map(|c| if (r + c) % 2 == 0 { '.' } else { ' ' }).collect())
        .collect()
}

fn with_marks(marks: &[(usize, usize, i8)]) -> Vec<i8> {
    let rows = empty_rows();
    let refs: Vec<&str> = rows.iter().map(|s| s.as_str()).collect();
    let mut b = board_from(&refs);
    for &(r, c, v) in marks {
        b[r * WIDTH + c] = v;
    }
    b
}

// ------------------------------------------------------------------------------- scoring

#[test]
fn empty_board_scores_zero() {
    let b = with_marks(&[]);
    assert_eq!(score_player(&b, 0, PLAYER_0_MARK, HEIGHT, WIDTH), 0.0);
    assert_eq!(score_player(&b, 0, PLAYER_1_MARK, HEIGHT, WIDTH), 0.0);
}

#[test]
fn single_stone_scores_zero() {
    // runs shorter than 2 never score, even on the border
    let b = with_marks(&[(0, 0, PLAYER_0_MARK)]);
    assert_eq!(score_player(&b, 0, PLAYER_0_MARK, HEIGHT, WIDTH), 0.0);
}

#[test]
fn border_touching_run_of_two_scores_its_length() {
    // (0,0) and (1,1) are on the same main diagonal; (0,0) is on the border
    let b = with_marks(&[(0, 0, PLAYER_0_MARK), (1, 1, PLAYER_0_MARK)]);
    assert_eq!(score_player(&b, 0, PLAYER_0_MARK, HEIGHT, WIDTH), 2.0);
}

#[test]
fn interior_run_not_connected_to_border_scores_zero() {
    // (4,4)-(5,5) is a diagonal run of 2 that touches no edge and no edge-connected stone
    let b = with_marks(&[(4, 4, PLAYER_0_MARK), (5, 5, PLAYER_0_MARK)]);
    assert_eq!(score_player(&b, 0, PLAYER_0_MARK, HEIGHT, WIDTH), 0.0);
}

#[test]
fn interior_run_scores_once_reachable_through_a_chain() {
    // a chain from the left border to the interior run makes the run count
    let mut marks = vec![(4, 4, PLAYER_0_MARK), (5, 5, PLAYER_0_MARK)];
    for (r, c) in [(4, 0), (4, 2), (5, 1), (5, 3)] {
        marks.push((r, c, PLAYER_0_MARK));
    }
    let b = with_marks(&marks);
    let s = score_player(&b, 0, PLAYER_0_MARK, HEIGHT, WIDTH);
    assert!(s > 0.0, "expected the border-connected chain to make runs count, got {s}");
}

#[test]
fn reachability_is_eight_connected_not_four() {
    // (0,0) on the border, (1,1) diagonally adjacent: 8-connectivity reaches it,
    // 4-connectivity would not (the orthogonal neighbours are non-playable)
    let b = with_marks(&[(0, 0, PLAYER_0_MARK), (1, 1, PLAYER_0_MARK), (2, 2, PLAYER_0_MARK)]);
    assert_eq!(score_player(&b, 0, PLAYER_0_MARK, HEIGHT, WIDTH), 3.0);
}

#[test]
fn both_diagonal_directions_are_counted() {
    // an X centred at (1,1): main diagonal (0,0)-(1,1)-(2,2) and anti-diagonal (0,2)-(1,1)-(2,0)
    let b = with_marks(&[
        (0, 0, PLAYER_0_MARK),
        (1, 1, PLAYER_0_MARK),
        (2, 2, PLAYER_0_MARK),
        (0, 2, PLAYER_0_MARK),
        (2, 0, PLAYER_0_MARK),
    ]);
    assert_eq!(score_player(&b, 0, PLAYER_0_MARK, HEIGHT, WIDTH), 6.0);
}

#[test]
fn players_are_scored_independently() {
    let b = with_marks(&[
        (0, 0, PLAYER_0_MARK),
        (1, 1, PLAYER_0_MARK),
        (0, 4, PLAYER_1_MARK),
        (1, 5, PLAYER_1_MARK),
        (2, 6, PLAYER_1_MARK),
    ]);
    assert_eq!(score_player(&b, 0, PLAYER_0_MARK, HEIGHT, WIDTH), 2.0);
    assert_eq!(score_player(&b, 0, PLAYER_1_MARK, HEIGHT, WIDTH), 3.0);
}

#[test]
fn score_batch_matches_per_game_scoring() {
    let mut boards = Vec::new();
    let a = with_marks(&[(0, 0, PLAYER_0_MARK), (1, 1, PLAYER_0_MARK)]);
    let b = with_marks(&[(9, 1, PLAYER_0_MARK), (8, 2, PLAYER_0_MARK), (7, 3, PLAYER_0_MARK)]);
    boards.extend_from_slice(&a);
    boards.extend_from_slice(&b);
    let batch = score_batch(&boards, 2, PLAYER_0_MARK, HEIGHT, WIDTH);
    assert_eq!(batch[0], score_player(&boards, 0, PLAYER_0_MARK, HEIGHT, WIDTH));
    assert_eq!(batch[1], score_player(&boards, 1, PLAYER_0_MARK, HEIGHT, WIDTH));
    assert_eq!(batch[0], 2.0);
    assert_eq!(batch[1], 3.0);
}

// --------------------------------------------------------------------------- legal masks

#[test]
fn first_move_splits_the_board_between_players() {
    let boards = with_marks(&[]);
    let (m0, m1, c0, c1) = legal_masks_kernel(&boards, 1, &[0], WIDTH / 2, HEIGHT, WIDTH);
    for r in 0..HEIGHT {
        for c in 0..WIDTH {
            let playable = (r + c) % 2 == 0;
            let i = r * WIDTH + c;
            assert_eq!(m0[i], (playable && c < WIDTH / 2) as u8 as f32, "p0 at {r},{c}");
            assert_eq!(m1[i], (playable && c >= WIDTH / 2) as u8 as f32, "p1 at {r},{c}");
        }
    }
    assert_eq!(c0[0], 40.0);
    assert_eq!(c1[0], 40.0);
    assert_eq!(c0[0] + c1[0], 80.0);
}

#[test]
fn after_the_first_move_both_players_see_every_playable_square() {
    let boards = with_marks(&[]);
    let (m0, m1, c0, c1) = legal_masks_kernel(&boards, 1, &[1], WIDTH / 2, HEIGHT, WIDTH);
    assert_eq!(m0, m1);
    assert_eq!(c0[0], 80.0);
    assert_eq!(c1[0], 80.0);
}

#[test]
fn occupied_and_removed_squares_are_illegal() {
    let boards = with_marks(&[(0, 0, PLAYER_0_MARK), (2, 2, PLAYER_1_MARK), (4, 4, REMOVED_SQUARE)]);
    let (m0, _, c0, _) = legal_masks_kernel(&boards, 1, &[1], WIDTH / 2, HEIGHT, WIDTH);
    assert_eq!(m0[0], 0.0);
    assert_eq!(m0[2 * WIDTH + 2], 0.0);
    assert_eq!(m0[4 * WIDTH + 4], 0.0);
    assert_eq!(c0[0], 77.0);
}

#[test]
fn masks_are_computed_per_game() {
    let mut boards = with_marks(&[]);
    boards.extend_from_slice(&with_marks(&[]));
    let (_, _, c0, c1) = legal_masks_kernel(&boards, 2, &[0, 3], WIDTH / 2, HEIGHT, WIDTH);
    assert_eq!((c0[0], c1[0]), (40.0, 40.0)); // still on its first move
    assert_eq!((c0[1], c1[1]), (80.0, 80.0)); // past it
}

// ------------------------------------------------------------------------------- sampler

#[test]
fn sampler_only_ever_returns_legal_squares() {
    let boards = with_marks(&[(0, 0, PLAYER_0_MARK), (2, 2, REMOVED_SQUARE)]);
    let (mask, _, _, _) = legal_masks_kernel(&boards, 1, &[1], WIDTH / 2, HEIGHT, WIDTH);
    let dist = vec![1.0f32; HW];
    let mut rng = Rng::new(7);
    for _ in 0..20_000 {
        let (r, c) = sample_move_kernel(&dist, &mask, &[true], 1, HEIGHT, WIDTH, &mut rng);
        let i = r[0] as usize * WIDTH + c[0] as usize;
        assert_eq!(mask[i], 1.0, "sampled illegal square ({}, {})", r[0], c[0]);
    }
}

#[test]
fn sampler_ignores_inactive_games_and_leaves_the_placeholder() {
    let boards = with_marks(&[]);
    let (mask, _, _, _) = legal_masks_kernel(&boards, 1, &[1], WIDTH / 2, HEIGHT, WIDTH);
    let dist = vec![1.0f32; HW];
    let mut rng = Rng::new(7);
    let (r, c) = sample_move_kernel(&dist, &mask, &[false], 1, HEIGHT, WIDTH, &mut rng);
    assert_eq!((r[0], c[0]), (0, 0));
}

#[test]
fn sampler_is_deterministic_for_a_one_hot_distribution() {
    let boards = with_marks(&[]);
    let (mask, _, _, _) = legal_masks_kernel(&boards, 1, &[1], WIDTH / 2, HEIGHT, WIDTH);
    let mut dist = vec![0.0f32; HW];
    dist[6 * WIDTH + 4] = 1.0; // (6,4) is playable
    let mut rng = Rng::new(11);
    for _ in 0..1000 {
        let (r, c) = sample_move_kernel(&dist, &mask, &[true], 1, HEIGHT, WIDTH, &mut rng);
        assert_eq!((r[0], c[0]), (6, 4));
    }
}

#[test]
fn sampler_frequencies_track_the_distribution() {
    // two legal squares with weights 3:1
    let boards = with_marks(&[]);
    let (mut mask, _, _, _) = legal_masks_kernel(&boards, 1, &[1], WIDTH / 2, HEIGHT, WIDTH);
    for m in mask.iter_mut() {
        *m = 0.0;
    }
    let a = 0usize; // (0,0)
    let b = 2 * WIDTH + 2; // (2,2)
    mask[a] = 1.0;
    mask[b] = 1.0;
    let mut dist = vec![0.0f32; HW];
    dist[a] = 3.0;
    dist[b] = 1.0;

    let mut rng = Rng::new(2024);
    let trials = 200_000;
    let mut hits_a = 0;
    for _ in 0..trials {
        let (r, c) = sample_move_kernel(&dist, &mask, &[true], 1, HEIGHT, WIDTH, &mut rng);
        if (r[0], c[0]) == (0, 0) {
            hits_a += 1;
        }
    }
    let p = hits_a as f64 / trials as f64;
    assert!((p - 0.75).abs() < 0.01, "expected ~0.75, got {p}");
}

#[test]
fn sampler_falls_back_to_an_unmasked_uniform_draw_when_all_weight_is_zero() {
    // faithful to the Python kernel: the degenerate branch ignores the mask entirely
    let mask = vec![1.0f32; HW];
    let dist = vec![0.0f32; HW];
    let mut rng = Rng::new(3);
    let mut seen = std::collections::HashSet::new();
    for _ in 0..5000 {
        let (r, c) = sample_move_kernel(&dist, &mask, &[true], 1, HEIGHT, WIDTH, &mut rng);
        assert!((r[0] as usize) < HEIGHT && (c[0] as usize) < WIDTH);
        seen.insert((r[0], c[0]));
    }
    assert!(seen.len() > HW / 2, "fallback should cover the whole board, saw {}", seen.len());
}

// ---------------------------------------------------------------------- move application

#[test]
fn distinct_moves_place_one_mark_each() {
    let mut boards = with_marks(&[]);
    let mut mc = vec![0i32];
    let mut fin = vec![false];
    let mut scores = vec![0.0f32; 2];
    apply_and_score_kernel(
        &mut boards, &mut mc, &mut fin, &mut scores,
        &[0], &[0], &[2], &[2], &[true], 1, HEIGHT, WIDTH,
    );
    assert_eq!(boards[0], PLAYER_0_MARK);
    assert_eq!(boards[2 * WIDTH + 2], PLAYER_1_MARK);
    assert_eq!(mc[0], 1);
    assert!(!fin[0]);
}

#[test]
fn a_collision_removes_the_square_and_its_playable_neighbours() {
    let mut boards = with_marks(&[]);
    let mut mc = vec![0i32];
    let mut fin = vec![false];
    let mut scores = vec![0.0f32; 2];
    // both players pick (4, 4)
    apply_and_score_kernel(
        &mut boards, &mut mc, &mut fin, &mut scores,
        &[4], &[4], &[4], &[4], &[true], 1, HEIGHT, WIDTH,
    );
    assert_eq!(boards[4 * WIDTH + 4], REMOVED_SQUARE);
    // the four diagonal neighbours are playable-parity, so they go too
    for (r, c) in [(3, 3), (3, 5), (5, 3), (5, 5)] {
        assert_eq!(boards[r * WIDTH + c], REMOVED_SQUARE, "({r},{c}) should be removed");
    }
    // the orthogonal neighbours are the wrong parity and stay non-playable
    for (r, c) in [(3, 4), (5, 4), (4, 3), (4, 5)] {
        assert_eq!(boards[r * WIDTH + c], NON_PLAYABLE_SQUARE, "({r},{c}) should be untouched");
    }
    assert_eq!(mc[0], 1);
}

#[test]
fn a_corner_collision_stays_in_bounds() {
    let mut boards = with_marks(&[]);
    let mut mc = vec![0i32];
    let mut fin = vec![false];
    let mut scores = vec![0.0f32; 2];
    apply_and_score_kernel(
        &mut boards, &mut mc, &mut fin, &mut scores,
        &[0], &[0], &[0], &[0], &[true], 1, HEIGHT, WIDTH,
    );
    assert_eq!(boards[0], REMOVED_SQUARE);
    assert_eq!(boards[1 * WIDTH + 1], REMOVED_SQUARE);
}

#[test]
fn inactive_games_are_untouched() {
    let mut boards = with_marks(&[]);
    let before = boards.clone();
    let mut mc = vec![7i32];
    let mut fin = vec![true];
    let mut scores = vec![0.0f32; 2];
    apply_and_score_kernel(
        &mut boards, &mut mc, &mut fin, &mut scores,
        &[0], &[0], &[2], &[2], &[false], 1, HEIGHT, WIDTH,
    );
    assert_eq!(boards, before);
    assert_eq!(mc[0], 7);
}

#[test]
fn finished_flips_when_the_last_playable_square_goes() {
    let mut boards = vec![NON_PLAYABLE_SQUARE; HW];
    boards[0] = PLAYABLE_SQUARE;
    boards[2 * WIDTH + 2] = PLAYABLE_SQUARE;
    let mut mc = vec![1i32];
    let mut fin = vec![false];
    let mut scores = vec![0.0f32; 2];
    apply_and_score_kernel(
        &mut boards, &mut mc, &mut fin, &mut scores,
        &[0], &[0], &[2], &[2], &[true], 1, HEIGHT, WIDTH,
    );
    assert!(fin[0], "board has no playable squares left");
}

#[test]
fn scores_are_refreshed_by_move_application() {
    let mut boards = with_marks(&[(0, 0, PLAYER_0_MARK)]);
    let mut mc = vec![1i32];
    let mut fin = vec![false];
    let mut scores = vec![0.0f32; 2];
    // player 0 extends the border-connected diagonal to length 2
    apply_and_score_kernel(
        &mut boards, &mut mc, &mut fin, &mut scores,
        &[1], &[1], &[8], &[8], &[true], 1, HEIGHT, WIDTH,
    );
    assert_eq!(scores[0], 2.0);
    assert_eq!(scores[1], 0.0);
}

#[test]
fn a_collision_blast_overwrites_marks_already_on_the_board() {
    // faithful to the Python kernel: the removal writes REMOVED_SQUARE over the
    // neighbourhood unconditionally, stones included
    let mut boards = with_marks(&[(3, 3, PLAYER_0_MARK), (5, 5, PLAYER_1_MARK)]);
    let mut mc = vec![1i32];
    let mut fin = vec![false];
    let mut scores = vec![0.0f32; 2];
    apply_and_score_kernel(
        &mut boards, &mut mc, &mut fin, &mut scores,
        &[4], &[4], &[4], &[4], &[true], 1, HEIGHT, WIDTH,
    );
    assert_eq!(boards[3 * WIDTH + 3], REMOVED_SQUARE);
    assert_eq!(boards[5 * WIDTH + 5], REMOVED_SQUARE);
}
