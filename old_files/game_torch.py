import torch
import re

from config import height, width


class batched_lines_game:
    """multiple games in parallel. only use batched. faster for more than 41 games at once"""
 
    # game seems to be pretty much always more preformant on cpu, due to small tensor sizes and large cpu-gpu communication overhead
    device = "cpu"
 
    # Game state constants
    NON_PLAYABLE_SQUARE = 0
    PLAYABLE_SQUARE = 1
    REMOVED_SQUARE = 2
    PLAYER_0_MARK = 3
    PLAYER_1_MARK = 4
 
    # Device-specific precomputation registry
    _precomputed_cache = {}
 
    @classmethod
    def _get_precomputations(cls):
        """Computes or retrieves masks and shearing indexing matrices for a specific device."""
        if cls.device in cls._precomputed_cache:
            return cls._precomputed_cache[cls.device]
 
        # Precompute playable mask
        row_idx = torch.arange(height).unsqueeze(1)
        col_idx = torch.arange(width).unsqueeze(0)
        playable_mask = ((row_idx + col_idx) % 2 == 0).to(cls.device)
 
        # Precompute edge mask for flood fill
        edge_mask = torch.zeros(height, width, dtype=torch.bool, device=cls.device)
        edge_mask[0, :] = True
        edge_mask[-1, :] = True
        edge_mask[:, 0] = True
        edge_mask[:, -1] = True
 
        # Shearing precomputation for batched line scoring
        sheared_width = height + width - 1
 
        nesw_col_idx = torch.full((height, sheared_width), -1, dtype=torch.long)
        for r in range(height):
            for c in range(width):
                nesw_col_idx[r, c + r] = c
 
        nwse_col_idx = torch.full((height, sheared_width), -1, dtype=torch.long)
        for r in range(height):
            for c in range(width):
                nwse_col_idx[r, c - r + height - 1] = c
 
        nesw_valid = (nesw_col_idx >= 0).to(cls.device)
        nwse_valid = (nwse_col_idx >= 0).to(cls.device)
        nesw_col_safe = nesw_col_idx.clamp(min=0).to(cls.device)
        nwse_col_safe = nwse_col_idx.clamp(min=0).to(cls.device)
 
        nesw_score_cols = []
        for j in range(sheared_width):
            count = sum(1 for r in range(height) if 0 <= j - r < width and (r + j - r) % 2 == 0)
            if count >= 2:
                nesw_score_cols.append(j)
 
        nwse_score_cols = []
        for j in range(sheared_width):
            count = sum(1 for r in range(height) if 0 <= j - (height - 1) + r < width and (r + j - (height - 1) + r) % 2 == 0)
            if count >= 2:
                nwse_score_cols.append(j)
 
        nesw_score_cols_t = torch.tensor(nesw_score_cols, dtype=torch.long, device=cls.device)
        nwse_score_cols_t = torch.tensor(nwse_score_cols, dtype=torch.long, device=cls.device)
 
        cls._precomputed_cache[cls.device] = {
            "playable_mask": playable_mask,
            "edge_mask": edge_mask,
            "nesw_col_safe": nesw_col_safe,
            "nwse_col_safe": nwse_col_safe,
            "nesw_valid": nesw_valid,
            "nwse_valid": nwse_valid,
            "nesw_score_cols_t": nesw_score_cols_t,
            "nwse_score_cols_t": nwse_score_cols_t,
        }
        return cls._precomputed_cache[cls.device]
 
    def __init__(self, num_games=1):
        self.n = num_games
 
        # Pull precomputations from cache without re-allocation
        cache = self._get_precomputations()
        self._playable_mask = cache["playable_mask"]
        self._edge_mask = cache["edge_mask"]
        self._nesw_col_safe = cache["nesw_col_safe"]
        self._nwse_col_safe = cache["nwse_col_safe"]
        self._nesw_valid = cache["nesw_valid"]
        self._nwse_valid = cache["nwse_valid"]
        self._nesw_score_cols_t = cache["nesw_score_cols_t"]
        self._nwse_score_cols_t = cache["nwse_score_cols_t"]
 
        self.boards = torch.full(
            (num_games, height, width),
            self.NON_PLAYABLE_SQUARE,
            dtype=torch.int8,
            device=self.device,
        )
        self.boards[:, self._playable_mask] = self.PLAYABLE_SQUARE
 
        self.scores = torch.zeros(num_games, 2, dtype=torch.float32, device=self.device)
        self.move_counts = torch.zeros(num_games, dtype=torch.int32, device=self.device)
        self.finished = torch.zeros(num_games, dtype=torch.bool, device=self.device)
        self.half_width = width // 2
 
    def _raw_masks(self):
        """(N, H, W) float legal-move masks for both players, honoring the first-move half-board rule."""
        playable = (self.boards == self.PLAYABLE_SQUARE).float()
        mask_0, mask_1 = playable.clone(), playable.clone()
        first = self.move_counts == 0
        if first.any():
            mask_0[first, :, self.half_width:] = 0
            mask_1[first, :, :self.half_width] = 0
        return mask_0, mask_1
 
    def _sample_move(self, dist, mask, active):
        """Samples one flat move index per game from dist, restricted to legal squares."""
        flat = (dist * mask).view(self.n, -1)
        flat[~active] = 0
        totals = flat.sum(1, keepdim=True)
        prob = flat / totals.clamp(min=1e-8)
        no_moves = ~active & (totals.squeeze(1) < 1e-7)
        if no_moves.any():
            prob[no_moves] = 1.0 / prob.shape[1]
        return torch.multinomial(prob, 1).squeeze(1)
 
    def _apply_moves(self, active, idx_0, idx_1):
        """Marks a move (or resolves a collision) for each active game and refreshes derived state."""
        r0, c0 = idx_0 // width, idx_0 % width
        r1, c1 = idx_1 // width, idx_1 % width
        collision = (r0 == r1) & (c0 == c1)
 
        nc = active & ~collision
        if nc.any():
            i = nc.nonzero(as_tuple=True)[0]
            self.boards[i, r0[i], c0[i]] = self.PLAYER_0_MARK
            self.boards[i, r1[i], c1[i]] = self.PLAYER_1_MARK
 
        ca = active & collision
        if ca.any():
            i = ca.nonzero(as_tuple=True)[0]
            cr, cc = r0[i], c0[i]
            self.boards[i, cr, cc] = self.REMOVED_SQUARE
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc2 = cr + dr, cc + dc
                    valid = (nr >= 0) & (nr < height) & (nc2 >= 0) & (nc2 < width)
                    m = valid & ((nr + nc2) % 2 == 0)
                    if m.any():
                        self.boards[i[m], nr[m], nc2[m]] = self.REMOVED_SQUARE
 
        self.move_counts[active] += 1
        self._update_scores()
        has_playable = (self.boards == self.PLAYABLE_SQUARE).view(self.n, -1).any(dim=1)
        self.finished |= ~has_playable
 
    def _update_scores(self):
        """Vectorized line score using board shearing, for both players. Edge flood fill
        (8-connected, from the border inward) is inlined since it's only ever used here."""
        for p, pmark in enumerate([self.PLAYER_0_MARK, self.PLAYER_1_MARK]):
            player_mask = (self.boards == pmark)
            reached = player_mask & self._edge_mask.unsqueeze(0)
            while True:
                dilated = reached.clone()
                dilated[:, 1:, :] |= reached[:, :-1, :]
                dilated[:, :-1, :] |= reached[:, 1:, :]
                dilated[:, :, 1:] |= reached[:, :, :-1]
                dilated[:, :, :-1] |= reached[:, :, 1:]
                dilated[:, 1:, 1:] |= reached[:, :-1, :-1]
                dilated[:, 1:, :-1] |= reached[:, :-1, 1:]
                dilated[:, :-1, 1:] |= reached[:, 1:, :-1]
                dilated[:, :-1, :-1] |= reached[:, 1:, 1:]
                new = dilated & player_mask
                if (new == reached).all():
                    break
                reached = new
            reached, player_mask = reached.float(), player_mask.float()
 
            scores = torch.zeros(self.n, dtype=torch.float32, device=self.device)
 
            for col_idx_safe, valid_mask, score_cols in [
                (self._nesw_col_safe, self._nesw_valid, self._nesw_score_cols_t),
                (self._nwse_col_safe, self._nwse_valid, self._nwse_score_cols_t),
            ]:
                idx = col_idx_safe.unsqueeze(0).expand(self.n, -1, -1)
                vm = valid_mask.unsqueeze(0).float()
                sheared_marks = torch.gather(player_mask, 2, idx) * vm
                sheared_reached = torch.gather(reached, 2, idx) * vm
 
                sheared_marks = sheared_marks[:, :, score_cols]
                sheared_reached = sheared_reached[:, :, score_cols]
 
                cumlen = torch.zeros_like(sheared_marks)
                cumreach = torch.zeros_like(sheared_reached)
 
                cumlen[:, 0, :] = sheared_marks[:, 0, :]
                cumreach[:, 0, :] = sheared_reached[:, 0, :]
 
                for i in range(1, height):
                    cumlen[:, i, :] = (cumlen[:, i - 1, :] + 1) * sheared_marks[:, i, :]
                    cumreach[:, i, :] = torch.max(
                        cumreach[:, i - 1, :] * sheared_marks[:, i, :],
                        sheared_reached[:, i, :]
                    )
 
                is_end = torch.zeros_like(sheared_marks, dtype=torch.bool)
                is_end[:, :-1, :] = (sheared_marks[:, :-1, :] > 0) & (sheared_marks[:, 1:, :] == 0)
                is_end[:, -1, :] = sheared_marks[:, -1, :] > 0
 
                valid = is_end & (cumlen >= 2) & (cumreach > 0)
                scores += (cumlen * valid.float()).sum(dim=(1, 2))
 
            self.scores[:, p] = scores

    def distribution_step(self, dist_p0, dist_p1):
        """One move for all active games, sampled from dist_p0/p1: (N, H, W) on self.device."""
        active = ~self.finished
        if not active.any():
            return
        mask_0, mask_1 = self._raw_masks()
        idx_0 = self._sample_move(dist_p0, mask_0, active)
        idx_1 = self._sample_move(dist_p1, mask_1, active)
        self._apply_moves(active, idx_0, idx_1)
 
    def action_step(self, idx_0, idx_1):
        """Advances games using explicit 1D action indices after validating move legality."""
        active = ~self.finished
        if not active.any():
            return
 
        mask_0, mask_1 = (m.view(self.n, -1) for m in self._raw_masks())
        b = torch.arange(self.n, device=self.device)
        invalid = active & ((mask_0[b, idx_0] != 1.0) | (mask_1[b, idx_1] != 1.0))
 
        if invalid.any():
            raise ValueError(
                f"Invalid move detected in batch at game indices: {invalid.nonzero(as_tuple=True)[0].tolist()}. "
                f"Execution aborted; no games updated."
            )
 
        self._apply_moves(active, idx_0, idx_1)
 
    def clone_states_to_batch(self, batch_indices) -> "batched_lines_game":
        """Creates and returns a new instance containing cloned states at the given indices."""
        target = batched_lines_game(num_games=len(batch_indices))
        target.boards[:] = self.boards[batch_indices].clone()
        target.scores[:] = self.scores[batch_indices].clone()
        target.move_counts[:] = self.move_counts[batch_indices].clone()
        target.finished[:] = self.finished[batch_indices].clone()
        return target
 
    def get_encoded_states(self, player):
        """Get encoded states. Returns (N, 7, H, W)."""
        encoding = torch.zeros(self.n, 7, height, width, device=self.device)
        encoding.scatter_(1, self.boards.long().unsqueeze(1), 1.0)
 
        if player == 1:
            encoding[:, 3], encoding[:, 4] = encoding[:, 4].clone(), encoding[:, 3].clone()
 
        playable = (self.boards == self.PLAYABLE_SQUARE).float()
        first = self.move_counts == 0
        if first.any():
            if player == 0:
                playable[first, :, self.half_width:] = 0
            elif player == 1:
                playable[first, :, :self.half_width] = 0
        encoding[:, 1] = playable
 
        norm = width * height / 2
        own, opp = (0, 1) if player == 0 else (1, 0)
        encoding[:, 5] = (self.scores[:, own] / norm).view(-1, 1, 1)
        encoding[:, 6] = (self.scores[:, opp] / norm).view(-1, 1, 1)
 
        return encoding
 
    def get_terminal_outcomes(self):
        """Returns win/draw/loss codes (0, 1, 2) for both players.
        Strictly enforces that all parallel games in the batch are complete.
        """
        assert self.finished.all(), "Cannot compute terminal outcomes: some parallel games are still active."
 
        s0 = self.scores[:, 0].long()
        s1 = self.scores[:, 1].long()
 
        val_p0 = torch.where(s0 > s1, 0, torch.where(s0 == s1, 1, 2))
        val_p1 = torch.where(s1 > s0, 0, torch.where(s1 == s0, 1, 2))
        return val_p0, val_p1
 
    def get_legal_masks(self):
        """Returns (mask_p0, mask_p1, count_p0, count_p1).
        mask_p0, mask_p1: (N, H*W) float; count_p0, count_p1: (N,) float.
        """
        mask_0, mask_1 = (m.view(self.n, -1) for m in self._raw_masks())
        return mask_0, mask_1, mask_0.sum(dim=1), mask_1.sum(dim=1)
 
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
            board_state = torch.argmax(game_enc[:5], dim=0)
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
 
        game._update_scores()
 
        for g, (expected_scores, _) in enumerate(blocks):
            non_initial = (
                (game.boards[g] != cls.PLAYABLE_SQUARE)
                & (game.boards[g] != cls.NON_PLAYABLE_SQUARE)
            )
            game.move_counts[g] = 1 if non_initial.any() else 0
            game.finished[g] = not (game.boards[g] == cls.PLAYABLE_SQUARE).any()
 
            if expected_scores is not None:
                actual = (int(game.scores[g, 0].item()), int(game.scores[g, 1].item()))
                assert actual == expected_scores, \
                    f"Game {g}: score mismatch: printed {expected_scores}, computed {actual}"
 
        return game
 