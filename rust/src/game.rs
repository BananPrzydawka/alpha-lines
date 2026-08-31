//! Port of `main/game.py` (`batched_lines_game`).
//!
//! Faithful 1:1: same state, same method semantics, same output formatting, same quirks
//! (including `print_state` rendering first-move-restricted squares as blanks, and
//! `import_prints` collapsing `move_counts` to 0/1).
//!
//! The one structural difference is the RNG: numba's kernels draw from a hidden global
//! `np.random` state, so the sampler needs an explicit source here. It lives on the struct.

use crate::config::{HEIGHT, WIDTH};
use crate::game_kernels::{
    apply_and_score_kernel, legal_masks_kernel, sample_move_kernel, score_batch,
    NON_PLAYABLE_SQUARE, PLAYABLE_SQUARE, PLAYER_0_MARK, PLAYER_1_MARK, REMOVED_SQUARE,
};
use crate::rng::Rng;

/// Multiple games in parallel. `Vec`-backed state; the kernels handle the whole step
/// pipeline (masks, sampling, move application, collision, scoring, finished check) in one
/// call.
#[derive(Clone, Debug)]
pub struct BatchedLinesGame {
    pub n: usize,
    /// (N, H, W) row-major.
    pub boards: Vec<i8>,
    /// (N, 2) row-major.
    pub scores: Vec<f32>,
    pub move_counts: Vec<i32>,
    pub finished: Vec<bool>,
    pub half_width: usize,
    pub rng: Rng,
}

impl BatchedLinesGame {
    pub const NON_PLAYABLE_SQUARE: i8 = NON_PLAYABLE_SQUARE;
    pub const PLAYABLE_SQUARE: i8 = PLAYABLE_SQUARE;
    pub const REMOVED_SQUARE: i8 = REMOVED_SQUARE;
    pub const PLAYER_0_MARK: i8 = PLAYER_0_MARK;
    pub const PLAYER_1_MARK: i8 = PLAYER_1_MARK;

    /// `__init__`. The seed has no Python counterpart; see the module note on the RNG.
    pub fn new(num_games: usize, seed: u64) -> Self {
        let hw = HEIGHT * WIDTH;
        let mut boards = vec![NON_PLAYABLE_SQUARE; num_games * hw];
        for g in 0..num_games {
            for r in 0..HEIGHT {
                for c in 0..WIDTH {
                    if (r + c) % 2 == 0 {
                        boards[g * hw + r * WIDTH + c] = PLAYABLE_SQUARE;
                    }
                }
            }
        }
        BatchedLinesGame {
            n: num_games,
            boards,
            scores: vec![0.0; num_games * 2],
            move_counts: vec![0; num_games],
            finished: vec![false; num_games],
            half_width: WIDTH / 2,
            rng: Rng::new(seed),
        }
    }

    /// `~self.finished`
    fn active(&self) -> Vec<bool> {
        self.finished.iter().map(|f| !f).collect()
    }

    /// `_raw_masks`: the (N, H, W)-shaped masks, before the flatten `get_legal_masks` does.
    pub fn raw_masks(&self) -> (Vec<f32>, Vec<f32>) {
        let (m0, m1, _, _) = legal_masks_kernel(
            &self.boards,
            self.n,
            &self.move_counts,
            self.half_width,
            HEIGHT,
            WIDTH,
        );
        (m0, m1)
    }

    /// `distribution_step`: one move for all active games, sampled from `dist_p0`/`dist_p1`,
    /// each (N, H, W) row-major.
    pub fn distribution_step(&mut self, dist_p0: &[f32], dist_p1: &[f32]) {
        let active = self.active();
        if !active.iter().any(|&a| a) {
            return;
        }
        let (mask_0, mask_1) = self.raw_masks();
        let (r0, c0) = sample_move_kernel(
            dist_p0, &mask_0, &active, self.n, HEIGHT, WIDTH, &mut self.rng,
        );
        let (r1, c1) = sample_move_kernel(
            dist_p1, &mask_1, &active, self.n, HEIGHT, WIDTH, &mut self.rng,
        );
        apply_and_score_kernel(
            &mut self.boards,
            &mut self.move_counts,
            &mut self.finished,
            &mut self.scores,
            &r0,
            &c0,
            &r1,
            &c1,
            &active,
            self.n,
            HEIGHT,
            WIDTH,
        );
    }

    /// `action_step`: advances games using explicit 1D action indices after validating move
    /// legality. Mirrors the Python `ValueError` as an `Err`.
    pub fn action_step(&mut self, idx_0: &[i64], idx_1: &[i64]) -> Result<(), String> {
        let active = self.active();
        if !active.iter().any(|&a| a) {
            return Ok(());
        }

        let hw = HEIGHT * WIDTH;
        let (mask_0, mask_1) = self.raw_masks();

        let mut invalid_indices: Vec<usize> = Vec::new();
        for g in 0..self.n {
            let a0 = idx_0[g] as usize;
            let a1 = idx_1[g] as usize;
            if active[g] && (mask_0[g * hw + a0] != 1.0 || mask_1[g * hw + a1] != 1.0) {
                invalid_indices.push(g);
            }
        }
        if !invalid_indices.is_empty() {
            return Err(format!(
                "Invalid move detected in batch at game indices: {:?}. \
                 Execution aborted; no games updated.",
                invalid_indices
            ));
        }

        let w = WIDTH as i64;
        let r0: Vec<i64> = idx_0.iter().map(|&i| i / w).collect();
        let c0: Vec<i64> = idx_0.iter().map(|&i| i % w).collect();
        let r1: Vec<i64> = idx_1.iter().map(|&i| i / w).collect();
        let c1: Vec<i64> = idx_1.iter().map(|&i| i % w).collect();

        apply_and_score_kernel(
            &mut self.boards,
            &mut self.move_counts,
            &mut self.finished,
            &mut self.scores,
            &r0,
            &c0,
            &r1,
            &c1,
            &active,
            self.n,
            HEIGHT,
            WIDTH,
        );
        Ok(())
    }

    /// `clone_states_to_batch`: a new instance holding cloned states at the given indices.
    pub fn clone_states_to_batch(&mut self, batch_indices: &[usize]) -> BatchedLinesGame {
        let hw = HEIGHT * WIDTH;
        // The Python version constructs a fresh instance; here that also means a fresh RNG,
        // which is drawn from the parent so clones stay deterministic under a seeded parent.
        let mut target = BatchedLinesGame::new(batch_indices.len(), self.rng.next_u64());
        for (t, &s) in batch_indices.iter().enumerate() {
            target.boards[t * hw..(t + 1) * hw].copy_from_slice(&self.boards[s * hw..(s + 1) * hw]);
            target.scores[t * 2] = self.scores[s * 2];
            target.scores[t * 2 + 1] = self.scores[s * 2 + 1];
            target.move_counts[t] = self.move_counts[s];
            target.finished[t] = self.finished[s];
        }
        target
    }

    /// `get_encoded_states`: (N, 7, H, W) f32, row-major.
    pub fn get_encoded_states(&self, player: usize) -> Vec<f32> {
        let n = self.n;
        let hw = HEIGHT * WIDTH;
        let plane = hw;
        let per_game = 7 * hw;
        let mut encoding = vec![0.0f32; n * per_game];

        for g in 0..n {
            for i in 0..hw {
                let v = self.boards[g * hw + i];
                if (0..5).contains(&v) {
                    encoding[g * per_game + v as usize * plane + i] = 1.0;
                }
            }
        }

        if player == 1 {
            for g in 0..n {
                for i in 0..hw {
                    let a = g * per_game + 3 * plane + i;
                    let b = g * per_game + 4 * plane + i;
                    encoding.swap(a, b);
                }
            }
        }

        // channel 1 is overwritten by the playable mask, which additionally honors the
        // first-move half-board restriction
        for g in 0..n {
            let first = self.move_counts[g] == 0;
            for r in 0..HEIGHT {
                for c in 0..WIDTH {
                    let i = r * WIDTH + c;
                    let mut playable =
                        if self.boards[g * hw + i] == PLAYABLE_SQUARE { 1.0f32 } else { 0.0f32 };
                    if first {
                        if player == 0 && c >= self.half_width {
                            playable = 0.0;
                        } else if player == 1 && c < self.half_width {
                            playable = 0.0;
                        }
                    }
                    encoding[g * per_game + plane + i] = playable;
                }
            }
        }

        let norm = (WIDTH * HEIGHT) as f32 / 2.0;
        let (own, opp) = if player == 0 { (0usize, 1usize) } else { (1usize, 0usize) };
        for g in 0..n {
            let v_own = self.scores[g * 2 + own] / norm;
            let v_opp = self.scores[g * 2 + opp] / norm;
            for i in 0..hw {
                encoding[g * per_game + 5 * plane + i] = v_own;
                encoding[g * per_game + 6 * plane + i] = v_opp;
            }
        }

        encoding
    }

    /// `get_terminal_outcomes`: win/draw/loss codes (0, 1, 2) for both players.
    /// Panics if any game is still active, matching the Python assert.
    pub fn get_terminal_outcomes(&self) -> (Vec<i64>, Vec<i64>) {
        assert!(
            self.finished.iter().all(|&f| f),
            "Cannot compute terminal outcomes: some parallel games are still active."
        );
        let mut val_p0 = vec![0i64; self.n];
        let mut val_p1 = vec![0i64; self.n];
        for g in 0..self.n {
            let s0 = self.scores[g * 2];
            let s1 = self.scores[g * 2 + 1];
            val_p0[g] = if s0 > s1 { 0 } else if s0 == s1 { 1 } else { 2 };
            val_p1[g] = if s1 > s0 { 0 } else if s1 == s0 { 1 } else { 2 };
        }
        (val_p0, val_p1)
    }

    /// `get_legal_masks`: `(mask_p0, mask_p1, count_p0, count_p1)` with the masks flattened
    /// to (N, H*W) — which, row-major, is the same buffer the kernel already produced.
    pub fn get_legal_masks(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        legal_masks_kernel(
            &self.boards,
            self.n,
            &self.move_counts,
            self.half_width,
            HEIGHT,
            WIDTH,
        )
    }

    /// `print_state`, as a string. `indices == None` means every game.
    pub fn format_state(&self, indices: Option<&[usize]>, player: usize) -> String {
        let all: Vec<usize>;
        let indices = match indices {
            Some(ix) => ix,
            None => {
                all = (0..self.n).collect();
                &all
            }
        };

        let encoding = self.get_encoded_states(player);
        let h = HEIGHT;
        let w = WIDTH;
        let hw = h * w;
        let per_game = 7 * hw;
        let norm = (w * h) as f64 / 2.0;
        let symbols = ["  ", "--", "##", "||", "oo"];

        let mut out = String::new();
        for &idx in indices {
            let base = idx * per_game;
            let s_own = python_round(encoding[base + 5 * hw] as f64 * norm) as i64;
            let s_opp = python_round(encoding[base + 6 * hw] as f64 * norm) as i64;
            let (s0, s1) = if player == 0 { (s_own, s_opp) } else { (s_opp, s_own) };

            out.push_str(&format!("Game Index: {} | Scores: [{}, {}]\n", idx, s0, s1));
            let border: String = format!("+{}+", "-".repeat(w * 2));
            out.push_str(&border);
            out.push('\n');
            for r in 0..h {
                out.push('|');
                for c in 0..w {
                    // np.argmax over the first 5 channels: first maximum wins
                    let mut best = 0usize;
                    let mut best_v = encoding[base + r * w + c];
                    for v in 1..5usize {
                        let x = encoding[base + v * hw + r * w + c];
                        if x > best_v {
                            best_v = x;
                            best = v;
                        }
                    }
                    out.push_str(symbols[best]);
                }
                out.push_str("|\n");
            }
            out.push_str(&border);
            out.push('\n');
        }
        out
    }

    /// `print_state`.
    pub fn print_state(&self, indices: Option<&[usize]>, player: usize) {
        print!("{}", self.format_state(indices, player));
    }

    /// `import_prints`: reconstruct a batched instance from one or more `print_state` blocks.
    pub fn import_prints(text: &str, player: usize, seed: u64) -> Result<Self, String> {
        let (own_mark, opp_mark) = if player == 1 {
            (PLAYER_1_MARK, PLAYER_0_MARK)
        } else {
            (PLAYER_0_MARK, PLAYER_1_MARK)
        };
        let cell = |s: &str| -> Option<i8> {
            match s {
                "  " => Some(NON_PLAYABLE_SQUARE),
                "--" => Some(PLAYABLE_SQUARE),
                "##" => Some(REMOVED_SQUARE),
                "||" => Some(own_mark),
                "oo" => Some(opp_mark),
                _ => None,
            }
        };

        let mut blocks: Vec<(Option<(i64, i64)>, Vec<String>)> = Vec::new();
        let mut current_scores: Option<(i64, i64)> = None;
        let mut current_board_lines: Vec<String> = Vec::new();

        for line in text.lines() {
            let stripped = line.trim();
            if stripped.is_empty() {
                continue;
            }
            if stripped.starts_with("Game Index:") {
                if !current_board_lines.is_empty() {
                    blocks.push((current_scores, std::mem::take(&mut current_board_lines)));
                }
                current_scores = parse_scores(stripped);
            } else if stripped.starts_with('+') && stripped.chars().all(|ch| ch == '+' || ch == '-')
            {
                // border
            } else {
                current_board_lines.push(line.to_string());
            }
        }
        if !current_board_lines.is_empty() {
            blocks.push((current_scores, current_board_lines));
        }

        if blocks.is_empty() {
            return Err("No board data found in input.".to_string());
        }

        let hw = HEIGHT * WIDTH;
        let mut game = BatchedLinesGame::new(blocks.len(), seed);

        for (g, (_expected_scores, board_lines)) in blocks.iter().enumerate() {
            if board_lines.len() != HEIGHT {
                return Err(format!(
                    "Game {}: row count mismatch: parsed {}, expected {}",
                    g,
                    board_lines.len(),
                    HEIGHT
                ));
            }
            let w = (board_lines[0].chars().count() - 2) / 2;
            if w != WIDTH {
                return Err(format!(
                    "Game {}: column count mismatch: parsed {}, expected {}",
                    g, w, WIDTH
                ));
            }

            for (r, line) in board_lines.iter().enumerate() {
                let chars: Vec<char> = line.chars().collect();
                let inner: String = chars[1..chars.len() - 1].iter().collect();
                let inner: Vec<char> = inner.chars().collect();
                for c in 0..w {
                    let pair: String = inner[c * 2..(c + 1) * 2].iter().collect();
                    let mut val = cell(&pair)
                        .ok_or_else(|| format!("Game {}: unknown cell {:?}", g, pair))?;
                    if val == NON_PLAYABLE_SQUARE && (r + c) % 2 == 0 {
                        val = PLAYABLE_SQUARE;
                    }
                    game.boards[g * hw + r * WIDTH + c] = val;
                }
            }
        }

        let s0 = score_batch(&game.boards, game.n, PLAYER_0_MARK, HEIGHT, WIDTH);
        let s1 = score_batch(&game.boards, game.n, PLAYER_1_MARK, HEIGHT, WIDTH);
        for g in 0..game.n {
            game.scores[g * 2] = s0[g];
            game.scores[g * 2 + 1] = s1[g];
        }

        for (g, (expected_scores, _)) in blocks.iter().enumerate() {
            let mut non_initial = false;
            let mut any_playable = false;
            for i in 0..hw {
                let v = game.boards[g * hw + i];
                if v != PLAYABLE_SQUARE && v != NON_PLAYABLE_SQUARE {
                    non_initial = true;
                }
                if v == PLAYABLE_SQUARE {
                    any_playable = true;
                }
            }
            game.move_counts[g] = if non_initial { 1 } else { 0 };
            game.finished[g] = !any_playable;

            if let Some((e0, e1)) = *expected_scores {
                let actual = (game.scores[g * 2] as i64, game.scores[g * 2 + 1] as i64);
                if actual != (e0, e1) {
                    return Err(format!(
                        "Game {}: score mismatch: printed ({}, {}), computed ({}, {})",
                        g, e0, e1, actual.0, actual.1
                    ));
                }
            }
        }

        Ok(game)
    }
}

/// `re.search(r"Scores:\s*\[(\d+),\s*(\d+)\]", stripped)`, hand-rolled.
fn parse_scores(stripped: &str) -> Option<(i64, i64)> {
    let start = stripped.find("Scores:")? + "Scores:".len();
    let bytes: Vec<char> = stripped[start..].chars().collect();
    let mut i = 0;
    while i < bytes.len() && bytes[i].is_whitespace() {
        i += 1;
    }
    if i >= bytes.len() || bytes[i] != '[' {
        return None;
    }
    i += 1;
    let a_start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == a_start {
        return None;
    }
    let a: i64 = bytes[a_start..i].iter().collect::<String>().parse().ok()?;
    if i >= bytes.len() || bytes[i] != ',' {
        return None;
    }
    i += 1;
    while i < bytes.len() && bytes[i].is_whitespace() {
        i += 1;
    }
    let b_start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == b_start {
        return None;
    }
    let b: i64 = bytes[b_start..i].iter().collect::<String>().parse().ok()?;
    if i >= bytes.len() || bytes[i] != ']' {
        return None;
    }
    Some((a, b))
}

/// Python's `round()`: half-to-even. `f64::round` is half-away-from-zero, which would
/// disagree on exact .5 values.
fn python_round(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}
