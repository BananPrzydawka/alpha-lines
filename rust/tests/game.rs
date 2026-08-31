//! Tests for the `BatchedLinesGame` surface: construction, the two step methods, cloning,
//! encoding, terminal outcomes, rendering, and the print/import round-trip. Plus a
//! randomized rollout that asserts the state invariants after every move.

use alpha_lines_game::config::{HEIGHT, WIDTH};
use alpha_lines_game::game_kernels::*;
use alpha_lines_game::BatchedLinesGame;

const HW: usize = HEIGHT * WIDTH;

fn idx(r: usize, c: usize) -> i64 {
    (r * WIDTH + c) as i64
}

// -------------------------------------------------------------------------- construction

#[test]
fn a_fresh_batch_is_the_checkerboard_starting_position() {
    let g = BatchedLinesGame::new(3, 0);
    assert_eq!(g.n, 3);
    assert_eq!(g.half_width, WIDTH / 2);
    for game in 0..3 {
        for r in 0..HEIGHT {
            for c in 0..WIDTH {
                let want = if (r + c) % 2 == 0 { PLAYABLE_SQUARE } else { NON_PLAYABLE_SQUARE };
                assert_eq!(g.boards[game * HW + r * WIDTH + c], want, "game {game} at {r},{c}");
            }
        }
    }
    assert!(g.scores.iter().all(|&s| s == 0.0));
    assert!(g.move_counts.iter().all(|&m| m == 0));
    assert!(g.finished.iter().all(|&f| !f));
}

// ---------------------------------------------------------------------------- action_step

#[test]
fn action_step_places_both_marks() {
    let mut g = BatchedLinesGame::new(1, 0);
    g.action_step(&[idx(0, 0)], &[idx(0, 8)]).unwrap();
    assert_eq!(g.boards[0], PLAYER_0_MARK);
    assert_eq!(g.boards[8], PLAYER_1_MARK);
    assert_eq!(g.move_counts[0], 1);
}

#[test]
fn action_step_rejects_an_illegal_move_with_the_python_message() {
    let mut g = BatchedLinesGame::new(2, 0);
    // player 0's first move must be in the left half; (0, 8) is not
    let err = g.action_step(&[idx(0, 0), idx(0, 8)], &[idx(0, 8), idx(0, 10)]).unwrap_err();
    assert_eq!(
        err,
        "Invalid move detected in batch at game indices: [1]. \
         Execution aborted; no games updated."
    );
}

#[test]
fn a_rejected_action_step_leaves_every_game_untouched() {
    let mut g = BatchedLinesGame::new(2, 0);
    let boards = g.boards.clone();
    let move_counts = g.move_counts.clone();
    // game 0's move is fine, game 1's is not; neither may be applied
    assert!(g.action_step(&[idx(0, 0), idx(0, 8)], &[idx(0, 8), idx(0, 10)]).is_err());
    assert_eq!(g.boards, boards);
    assert_eq!(g.move_counts, move_counts);
}

#[test]
fn the_first_move_is_restricted_to_opposite_halves() {
    let mut g = BatchedLinesGame::new(1, 0);
    // player 1 may not open in the left half
    assert!(g.action_step(&[idx(0, 0)], &[idx(0, 2)]).is_err());
    // player 0 may not open in the right half
    assert!(g.action_step(&[idx(0, 10)], &[idx(0, 8)]).is_err());
    // the legal pairing works, and afterwards the restriction is gone
    g.action_step(&[idx(0, 0)], &[idx(0, 8)]).unwrap();
    g.action_step(&[idx(0, 10)], &[idx(0, 2)]).unwrap();
    assert_eq!(g.move_counts[0], 2);
}

#[test]
fn action_step_on_an_all_finished_batch_is_a_no_op() {
    let mut g = BatchedLinesGame::new(1, 0);
    g.finished[0] = true;
    let boards = g.boards.clone();
    g.action_step(&[idx(0, 0)], &[idx(0, 8)]).unwrap();
    assert_eq!(g.boards, boards);
    assert_eq!(g.move_counts[0], 0);
}

#[test]
fn a_collision_via_action_step_removes_the_neighbourhood() {
    let mut g = BatchedLinesGame::new(1, 0);
    g.action_step(&[idx(0, 0)], &[idx(0, 8)]).unwrap();
    g.action_step(&[idx(4, 4)], &[idx(4, 4)]).unwrap();
    assert_eq!(g.boards[4 * WIDTH + 4], REMOVED_SQUARE);
    for (r, c) in [(3, 3), (3, 5), (5, 3), (5, 5)] {
        assert_eq!(g.boards[r * WIDTH + c], REMOVED_SQUARE);
    }
}

// ----------------------------------------------------------------------- distribution_step

#[test]
fn distribution_step_only_plays_legal_squares() {
    let mut g = BatchedLinesGame::new(64, 99);
    let dist = vec![1.0f32; 64 * HW];
    let mut steps = 0;
    while !g.finished.iter().all(|&f| f) {
        let before = g.boards.clone();
        let finished_before = g.finished.clone();
        let (m0, m1, _, _) = g.get_legal_masks();
        g.distribution_step(&dist, &dist);

        for game in 0..g.n {
            if finished_before[game] {
                continue;
            }
            for i in 0..HW {
                let (b, a) = (before[game * HW + i], g.boards[game * HW + i]);
                if b == a {
                    continue;
                }
                match a {
                    // a mark can only be placed on a square the mask allowed
                    PLAYER_0_MARK => assert_eq!(m0[game * HW + i], 1.0, "game {game} illegal p0 move"),
                    PLAYER_1_MARK => assert_eq!(m1[game * HW + i], 1.0, "game {game} illegal p1 move"),
                    // the collision blast is the one path that overwrites a non-playable
                    // square, and it only ever writes REMOVED_SQUARE
                    REMOVED_SQUARE => {}
                    other => panic!("game {game} square {i} became {other}"),
                }
            }
        }
        steps += 1;
        assert!(steps < 200);
    }
}

// ------------------------------------------------------------------------------- cloning

#[test]
fn clone_states_to_batch_copies_the_selected_games_and_detaches_them() {
    let mut g = BatchedLinesGame::new(4, 0);
    g.action_step(&[idx(0, 0); 4], &[idx(0, 8); 4]).unwrap();
    g.action_step(&[idx(2, 2); 4], &[idx(2, 10); 4]).unwrap();

    let mut clone = g.clone_states_to_batch(&[2, 0, 2]);
    assert_eq!(clone.n, 3);
    for (t, &s) in [2usize, 0, 2].iter().enumerate() {
        assert_eq!(clone.boards[t * HW..(t + 1) * HW], g.boards[s * HW..(s + 1) * HW]);
        assert_eq!(clone.scores[t * 2], g.scores[s * 2]);
        assert_eq!(clone.scores[t * 2 + 1], g.scores[s * 2 + 1]);
        assert_eq!(clone.move_counts[t], g.move_counts[s]);
        assert_eq!(clone.finished[t], g.finished[s]);
    }

    // advancing the clone must not touch the parent
    let parent_boards = g.boards.clone();
    clone.action_step(&[idx(4, 4); 3], &[idx(4, 12); 3]).unwrap();
    assert_eq!(g.boards, parent_boards);
}

// ------------------------------------------------------------------------------ encoding

#[test]
fn encoding_channels_zero_through_four_are_a_one_hot_of_the_board() {
    let mut g = BatchedLinesGame::new(1, 0);
    g.action_step(&[idx(0, 0)], &[idx(0, 8)]).unwrap();
    g.action_step(&[idx(4, 4)], &[idx(4, 4)]).unwrap(); // collision -> removed squares
    let e = g.get_encoded_states(0);
    for i in 0..HW {
        let v = g.boards[i];
        for ch in 0..5usize {
            let want = if ch as i8 == v { 1.0 } else { 0.0 };
            // channel 1 is overwritten by the playable mask, which agrees here
            assert_eq!(e[ch * HW + i], want, "channel {ch} at {i} for board value {v}");
        }
    }
}

#[test]
fn encoding_channel_one_hides_the_half_the_player_may_not_open_in() {
    let g = BatchedLinesGame::new(1, 0);
    let e0 = g.get_encoded_states(0);
    let e1 = g.get_encoded_states(1);
    for r in 0..HEIGHT {
        for c in 0..WIDTH {
            let i = r * WIDTH + c;
            let playable = (r + c) % 2 == 0;
            assert_eq!(e0[HW + i], (playable && c < WIDTH / 2) as u8 as f32);
            assert_eq!(e1[HW + i], (playable && c >= WIDTH / 2) as u8 as f32);
        }
    }
}

#[test]
fn encoding_swaps_the_player_planes_for_player_one() {
    let mut g = BatchedLinesGame::new(1, 0);
    g.action_step(&[idx(0, 0)], &[idx(0, 8)]).unwrap();
    let e0 = g.get_encoded_states(0);
    let e1 = g.get_encoded_states(1);
    assert_eq!(e0[3 * HW..4 * HW], e1[4 * HW..5 * HW]);
    assert_eq!(e0[4 * HW..5 * HW], e1[3 * HW..4 * HW]);
}

#[test]
fn encoding_channels_five_and_six_are_own_then_opponent_score() {
    let mut g = BatchedLinesGame::new(1, 0);
    g.action_step(&[idx(0, 0)], &[idx(0, 8)]).unwrap();
    g.action_step(&[idx(1, 1)], &[idx(3, 11)]).unwrap();
    let norm = (WIDTH * HEIGHT) as f32 / 2.0;
    let (s0, s1) = (g.scores[0], g.scores[1]);
    assert!(s0 > 0.0, "expected player 0 to have scored");

    let e0 = g.get_encoded_states(0);
    assert!(e0[5 * HW..6 * HW].iter().all(|&x| x == s0 / norm));
    assert!(e0[6 * HW..7 * HW].iter().all(|&x| x == s1 / norm));

    let e1 = g.get_encoded_states(1);
    assert!(e1[5 * HW..6 * HW].iter().all(|&x| x == s1 / norm));
    assert!(e1[6 * HW..7 * HW].iter().all(|&x| x == s0 / norm));
}

// ---------------------------------------------------------------------- terminal outcomes

#[test]
fn terminal_outcomes_encode_win_draw_loss() {
    let mut g = BatchedLinesGame::new(3, 0);
    g.finished = vec![true; 3];
    g.scores = vec![5.0, 3.0, /**/ 4.0, 4.0, /**/ 1.0, 9.0];
    let (p0, p1) = g.get_terminal_outcomes();
    assert_eq!(p0, vec![0, 1, 2]);
    assert_eq!(p1, vec![2, 1, 0]);
}

#[test]
#[should_panic(expected = "some parallel games are still active")]
fn terminal_outcomes_refuse_an_unfinished_batch() {
    let g = BatchedLinesGame::new(2, 0);
    let _ = g.get_terminal_outcomes();
}

// ------------------------------------------------------------------- rendering and import

#[test]
fn rendering_uses_the_expected_frame_and_symbols() {
    let mut g = BatchedLinesGame::new(1, 0);
    g.action_step(&[idx(0, 0)], &[idx(0, 8)]).unwrap();
    let text = g.format_state(None, 0);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], "Game Index: 0 | Scores: [0, 0]");
    assert_eq!(lines[1], format!("+{}+", "-".repeat(WIDTH * 2)));
    // player 0 at (0,0) renders "||", player 1 at (0,8) renders "oo"
    assert_eq!(lines[2], "|||  --  --  --  oo  --  --  --  |");
    assert_eq!(lines.len(), 2 + HEIGHT + 1);
    assert_eq!(lines[lines.len() - 1], lines[1]);
}

#[test]
fn rendering_selects_the_requested_games() {
    let g = BatchedLinesGame::new(4, 0);
    let text = g.format_state(Some(&[3, 1]), 0);
    let heads: Vec<&str> = text.lines().filter(|l| l.starts_with("Game Index")).collect();
    assert_eq!(heads, vec!["Game Index: 3 | Scores: [0, 0]", "Game Index: 1 | Scores: [0, 0]"]);
}

#[test]
fn import_prints_round_trips_a_played_out_batch() {
    let mut g = BatchedLinesGame::new(16, 5);
    let dist = vec![1.0f32; 16 * HW];
    for _ in 0..12 {
        g.distribution_step(&dist, &dist);
    }
    for player in [0usize, 1] {
        let text = g.format_state(None, player);
        let back = BatchedLinesGame::import_prints(&text, player, 0).unwrap();
        assert_eq!(back.n, g.n);
        assert_eq!(back.boards, g.boards);
        assert_eq!(back.scores, g.scores);
        assert_eq!(back.finished, g.finished);
        // import_prints cannot recover the real move count; it only distinguishes
        // "untouched" from "played", exactly as the Python version does
        assert!(back.move_counts.iter().all(|&m| m == 1));
        assert_eq!(back.format_state(None, player), text);
    }
}

#[test]
fn import_prints_restores_squares_the_first_move_rule_blanked_out() {
    // on move 0 the right half renders as blanks for player 0; the parity rule puts them back
    let g = BatchedLinesGame::new(2, 0);
    let text = g.format_state(None, 0);
    assert!(text.lines().nth(2).unwrap().contains("        "), "right half should render blank");
    let back = BatchedLinesGame::import_prints(&text, 0, 0).unwrap();
    assert_eq!(back.boards, g.boards);
    assert!(back.move_counts.iter().all(|&m| m == 0));
}

#[test]
fn import_prints_rejects_a_board_whose_printed_score_disagrees() {
    let mut g = BatchedLinesGame::new(1, 0);
    g.action_step(&[idx(0, 0)], &[idx(0, 8)]).unwrap();
    g.action_step(&[idx(1, 1)], &[idx(3, 11)]).unwrap();
    let text = g.format_state(None, 0).replace("Scores: [2, 0]", "Scores: [7, 0]");
    let err = BatchedLinesGame::import_prints(&text, 0, 0).unwrap_err();
    assert!(err.contains("score mismatch"), "unexpected error: {err}");
}

#[test]
fn import_prints_rejects_empty_input() {
    assert!(BatchedLinesGame::import_prints("", 0, 0).is_err());
}

// ------------------------------------------------------------------ end-to-end invariants

#[test]
fn a_full_random_rollout_keeps_every_state_invariant() {
    let mut g = BatchedLinesGame::new(128, 4242);
    let mut rng = alpha_lines_game::rng::Rng::new(7);
    let dist: Vec<f32> = (0..128 * HW).map(|_| rng.random() as f32).collect();

    let mut steps = 0;
    while !g.finished.iter().all(|&f| f) {
        let prev_counts = g.move_counts.clone();
        let prev_finished = g.finished.clone();
        g.distribution_step(&dist, &dist);
        steps += 1;
        assert!(steps < 200, "rollout did not terminate");

        let recomputed_0 = score_batch(&g.boards, g.n, PLAYER_0_MARK, HEIGHT, WIDTH);
        let recomputed_1 = score_batch(&g.boards, g.n, PLAYER_1_MARK, HEIGHT, WIDTH);

        for game in 0..g.n {
            // wrong-parity squares can never become playable or occupied
            for r in 0..HEIGHT {
                for c in 0..WIDTH {
                    if (r + c) % 2 != 0 {
                        assert_eq!(g.boards[game * HW + r * WIDTH + c], NON_PLAYABLE_SQUARE);
                    }
                }
            }
            assert!(g.boards[game * HW..(game + 1) * HW].iter().all(|&v| (0..5).contains(&v)));

            // cached scores agree with a from-scratch rescore
            assert_eq!(g.scores[game * 2], recomputed_0[game], "game {game} p0 score drifted");
            assert_eq!(g.scores[game * 2 + 1], recomputed_1[game], "game {game} p1 score drifted");

            // finished is exactly "no playable square left", and is sticky
            let any_playable =
                g.boards[game * HW..(game + 1) * HW].iter().any(|&v| v == PLAYABLE_SQUARE);
            assert_eq!(g.finished[game], !any_playable, "game {game} finished flag is wrong");
            if prev_finished[game] {
                assert!(g.finished[game], "game {game} un-finished itself");
            }

            // active games advance by exactly one move, finished games not at all
            let want = prev_counts[game] + if prev_finished[game] { 0 } else { 1 };
            assert_eq!(g.move_counts[game], want, "game {game} move count");
        }

        // masks never mark a non-playable square
        let (m0, m1, c0, c1) = g.get_legal_masks();
        for game in 0..g.n {
            let mut n0 = 0.0f32;
            let mut n1 = 0.0f32;
            for i in 0..HW {
                let playable = g.boards[game * HW + i] == PLAYABLE_SQUARE;
                assert!(playable || m0[game * HW + i] == 0.0);
                assert!(playable || m1[game * HW + i] == 0.0);
                n0 += m0[game * HW + i];
                n1 += m1[game * HW + i];
            }
            assert_eq!(c0[game], n0);
            assert_eq!(c1[game], n1);
        }
    }

    // every game ends, and the outcome codes are consistent with the final scores
    let (p0, p1) = g.get_terminal_outcomes();
    for game in 0..g.n {
        let (s0, s1) = (g.scores[game * 2], g.scores[game * 2 + 1]);
        assert_eq!(p0[game], if s0 > s1 { 0 } else if s0 == s1 { 1 } else { 2 });
        assert_eq!(p1[game], 2 - p0[game]);
    }
}
