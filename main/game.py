import numpy as np
import re

from config import height, width
from game_kernels import (
    NON_PLAYABLE_SQUARE, PLAYABLE_SQUARE, REMOVED_SQUARE, PLAYER_0_MARK, PLAYER_1_MARK,
    legal_masks_kernel, sample_move_kernel, apply_and_score_kernel, score_batch,
)


class batched_lines_game:
    """multiple games in parallel. Numpy-backed state; numba-compiled kernels handle the
    whole step pipeline (masks, sampling, move application, collision, scoring, finished
    check) in one call, avoiding per-op dispatch overhead on these small boards."""

    device = "cpu"  # kept for interface parity; state is numpy, not torch

    NON_PLAYABLE_SQUARE = NON_PLAYABLE_SQUARE
    PLAYABLE_SQUARE = PLAYABLE_SQUARE
    REMOVED_SQUARE = REMOVED_SQUARE
    PLAYER_0_MARK = PLAYER_0_MARK
    PLAYER_1_MARK = PLAYER_1_MARK

    _playable_mask_cache = None

    @classmethod
    def _playable_mask(cls):
        if cls._playable_mask_cache is None:
            r = np.arange(height)[:, None]
            c = np.arange(width)[None, :]
            cls._playable_mask_cache = (r + c) % 2 == 0
        return cls._playable_mask_cache

    def __init__(self, num_games=1):
        self.n = num_games
        self.boards = np.full((num_games, height, width), self.NON_PLAYABLE_SQUARE, dtype=np.int8)
        self.boards[:, self._playable_mask()] = self.PLAYABLE_SQUARE
        self.scores = np.zeros((num_games, 2), dtype=np.float32)
        self.move_counts = np.zeros(num_games, dtype=np.int32)
        self.finished = np.zeros(num_games, dtype=np.bool_)
        self.half_width = width // 2

    def _raw_masks(self):
        mask_0, mask_1, _, _ = legal_masks_kernel(self.boards, self.move_counts, self.half_width, height, width)
        return mask_0, mask_1

    def distribution_step(self, dist_p0, dist_p1):
        """One move for all active games, sampled from dist_p0/p1: (N, H, W)."""
        active = ~self.finished
        if not active.any():
            return
        mask_0, mask_1 = self._raw_masks()
        r0, c0 = sample_move_kernel(np.asarray(dist_p0), mask_0, active, height, width)
        r1, c1 = sample_move_kernel(np.asarray(dist_p1), mask_1, active, height, width)
        apply_and_score_kernel(self.boards, self.move_counts, self.finished, self.scores,
                                r0, c0, r1, c1, active, height, width)

    def action_step(self, idx_0, idx_1):
        """Advances games using explicit 1D action indices after validating move legality."""
        active = ~self.finished
        if not active.any():
            return

        idx_0 = np.asarray(idx_0)
        idx_1 = np.asarray(idx_1)
        mask_0, mask_1 = (m.reshape(self.n, -1) for m in self._raw_masks())
        b = np.arange(self.n)
        invalid = active & ((mask_0[b, idx_0] != 1.0) | (mask_1[b, idx_1] != 1.0))

        if invalid.any():
            raise ValueError(
                f"Invalid move detected in batch at game indices: {invalid.nonzero()[0].tolist()}. "
                f"Execution aborted; no games updated."
            )

        r0, c0 = (idx_0 // width).astype(np.int64), (idx_0 % width).astype(np.int64)
        r1, c1 = (idx_1 // width).astype(np.int64), (idx_1 % width).astype(np.int64)
        apply_and_score_kernel(self.boards, self.move_counts, self.finished, self.scores,
                                r0, c0, r1, c1, active, height, width)

    def clone_states_to_batch(self, batch_indices) -> "batched_lines_game":
        """Creates and returns a new instance containing cloned states at the given indices."""
        idx = np.asarray(batch_indices)
        target = batched_lines_game(num_games=len(idx))
        target.boards[:] = self.boards[idx]
        target.scores[:] = self.scores[idx]
        target.move_counts[:] = self.move_counts[idx]
        target.finished[:] = self.finished[idx]
        return target

    def get_encoded_states(self, player):
        """Get encoded states. Returns (N, 7, H, W) numpy float32 array.
        Wrap with torch.from_numpy(...) at the call site if feeding a torch model (zero-copy)."""
        n = self.n
        encoding = np.zeros((n, 7, height, width), dtype=np.float32)
        for v in range(5):
            encoding[:, v][self.boards == v] = 1.0

        if player == 1:
            encoding[:, 3], encoding[:, 4] = encoding[:, 4].copy(), encoding[:, 3].copy()

        playable = (self.boards == self.PLAYABLE_SQUARE).astype(np.float32)
        first = self.move_counts == 0
        if first.any():
            if player == 0:
                playable[first, :, self.half_width:] = 0
            elif player == 1:
                playable[first, :, :self.half_width] = 0
        encoding[:, 1] = playable

        norm = width * height / 2
        own, opp = (0, 1) if player == 0 else (1, 0)
        encoding[:, 5] = (self.scores[:, own] / norm)[:, None, None]
        encoding[:, 6] = (self.scores[:, opp] / norm)[:, None, None]

        return encoding

    def get_terminal_outcomes(self):
        """Returns win/draw/loss codes (0, 1, 2) for both players.
        Strictly enforces that all parallel games in the batch are complete.
        """
        assert self.finished.all(), "Cannot compute terminal outcomes: some parallel games are still active."

        s0 = self.scores[:, 0]
        s1 = self.scores[:, 1]

        val_p0 = np.where(s0 > s1, 0, np.where(s0 == s1, 1, 2))
        val_p1 = np.where(s1 > s0, 0, np.where(s1 == s0, 1, 2))
        return val_p0, val_p1

    def get_legal_masks(self):
        """Returns (mask_p0, mask_p1, count_p0, count_p1).
        mask_p0, mask_p1: (N, H*W) float; count_p0, count_p1: (N,) float.
        """
        mask_0, mask_1, count_0, count_1 = legal_masks_kernel(self.boards, self.move_counts, self.half_width, height, width)
        return mask_0.reshape(self.n, -1), mask_1.reshape(self.n, -1), count_0, count_1

    def print_state(self, index=None, player=0):
        """Prints game board configuration and tracking metrics."""
        if index is None:
            indices = list(range(self.n))
        elif isinstance(index, int):
            indices = [index]
        else:
            indices = [int(i) for i in index]

        encoding = self.get_encoded_states(player)
        h, w = encoding.shape[2], encoding.shape[3]
        norm = w * h / 2
        symbols = {0: "  ", 1: "--", 2: "##", 3: "||", 4: "oo"}

        for idx in indices:
            game_enc = encoding[idx]
            s_own = int(round(game_enc[5, 0, 0].item() * norm))
            s_opp = int(round(game_enc[6, 0, 0].item() * norm))
            s0, s1 = (s_own, s_opp) if player == 0 else (s_opp, s_own)

            print(f"Game Index: {idx} | Scores: [{s0}, {s1}]")
            board_state = np.argmax(game_enc[:5], axis=0)
            border = "+" + "-" * (w * 2) + "+"
            print(border)
            for r in range(h):
                line = "|" + "".join(symbols[board_state[r, c].item()] for c in range(w)) + "|"
                print(line)
            print(border)

    @classmethod
    def import_prints(cls, text: str, player: int = 0) -> "batched_lines_game":
        """Reconstruct a batched instance from one or more print_state blocks."""
        own_mark, opp_mark = (cls.PLAYER_1_MARK, cls.PLAYER_0_MARK) if player == 1 else (cls.PLAYER_0_MARK, cls.PLAYER_1_MARK)
        cell_map = {
            "  ": cls.NON_PLAYABLE_SQUARE, "--": cls.PLAYABLE_SQUARE,
            "##": cls.REMOVED_SQUARE, "||": own_mark, "oo": opp_mark,
        }

        blocks = []
        current_scores = None
        current_board_lines = []

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("Game Index:"):
                if current_board_lines:
                    blocks.append((current_scores, current_board_lines))
                    current_board_lines = []
                current_scores = None
                m = re.search(r"Scores:\s*\[(\d+),\s*(\d+)\]", stripped)
                if m:
                    current_scores = (int(m.group(1)), int(m.group(2)))
            elif stripped.startswith("+") and all(ch in "+-" for ch in stripped):
                pass
            else:
                current_board_lines.append(line)

        if current_board_lines:
            blocks.append((current_scores, current_board_lines))

        assert blocks, "No board data found in input."

        game = cls(num_games=len(blocks))

        for g, (expected_scores, board_lines) in enumerate(blocks):
            assert len(board_lines) == height, \
                f"Game {g}: row count mismatch: parsed {len(board_lines)}, expected {height}"
            w = (len(board_lines[0]) - 2) // 2
            assert w == width, \
                f"Game {g}: column count mismatch: parsed {w}, expected {width}"

            for r, line in enumerate(board_lines):
                inner = line[1:-1]
                for c in range(w):
                    val = cell_map[inner[c * 2:(c + 1) * 2]]
                    if val == cls.NON_PLAYABLE_SQUARE and (r + c) % 2 == 0:
                        val = cls.PLAYABLE_SQUARE
                    game.boards[g, r, c] = val

        game.scores[:, 0] = score_batch(game.boards, cls.PLAYER_0_MARK, height, width)
        game.scores[:, 1] = score_batch(game.boards, cls.PLAYER_1_MARK, height, width)

        for g, (expected_scores, _) in enumerate(blocks):
            non_initial = (
                (game.boards[g] != cls.PLAYABLE_SQUARE)
                & (game.boards[g] != cls.NON_PLAYABLE_SQUARE)
            )
            game.move_counts[g] = 1 if non_initial.any() else 0
            game.finished[g] = not (game.boards[g] == cls.PLAYABLE_SQUARE).any()

            if expected_scores is not None:
                actual = (int(game.scores[g, 0]), int(game.scores[g, 1]))
                assert actual == expected_scores, \
                    f"Game {g}: score mismatch: printed {expected_scores}, computed {actual}"

        return game