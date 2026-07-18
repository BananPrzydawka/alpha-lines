import torch
import random

from config import height, width

non_playable_square = 0
playable_square = 1
removed_square = 2
player_0_mark = 3
player_1_mark = 4

# Precompute playable mask
_row_idx = torch.arange(height).unsqueeze(1)
_col_idx = torch.arange(width).unsqueeze(0)
_playable_mask = ((_row_idx + _col_idx) % 2 == 0)

# Precompute edge mask for flood fill
_edge_mask = torch.zeros(height, width, dtype=torch.bool)
_edge_mask[0, :] = True
_edge_mask[-1, :] = True
_edge_mask[:, 0] = True
_edge_mask[:, -1] = True

# --- Shearing precomputation for batched line scoring ---
# Shear the board so diagonals become columns, then use vertical cumsum-with-reset.
# NE-SW diagonals (r+c=const): shift row r right by r
# NW-SE diagonals (r-c=const): shift row r right by (H-1-r)
_sheared_width = height + width - 1

_nesw_col_idx = torch.full((height, _sheared_width), -1, dtype=torch.long)
for r in range(height):
    for c in range(width):
        _nesw_col_idx[r, c + r] = c

_nwse_col_idx = torch.full((height, _sheared_width), -1, dtype=torch.long)
for r in range(height):
    for c in range(width):
        _nwse_col_idx[r, c - r + height - 1] = c

_nesw_valid = (_nesw_col_idx >= 0)
_nwse_valid = (_nwse_col_idx >= 0)
_nesw_col_safe = _nesw_col_idx.clamp(min=0)
_nwse_col_safe = _nwse_col_idx.clamp(min=0)

# Only sheared columns with >= 2 playable cells can produce scoring runs
_nesw_score_cols = []
for j in range(_sheared_width):
    count = sum(1 for r in range(height) if 0 <= j - r < width and (r + j - r) % 2 == 0)
    if count >= 2:
        _nesw_score_cols.append(j)

_nwse_score_cols = []
for j in range(_sheared_width):
    count = sum(1 for r in range(height) if 0 <= j - (height - 1) + r < width and (r + j - (height - 1) + r) % 2 == 0)
    if count >= 2:
        _nwse_score_cols.append(j)

_nesw_score_cols_t = torch.tensor(_nesw_score_cols, dtype=torch.long)
_nwse_score_cols_t = torch.tensor(_nwse_score_cols, dtype=torch.long)

# Precompute diagonal coordinates for Python single-game scoring
_diags1_coords = []
for s in range(0, height + width, 2):
    cells = []
    for r in range(height):
        c = s - r
        if 0 <= c < width:
            cells.append((r, c))
    if len(cells) >= 2:
        _diags1_coords.append(cells)

_diags2_coords = []
for d in range(-width, height + 1):
    if d % 2 != 0:
        continue
    cells = []
    for r in range(height):
        c = r - d
        if 0 <= c < width:
            cells.append((r, c))
    if len(cells) >= 2:
        _diags2_coords.append(cells)


def _calc_line_scores_python(board, player_mark):
    """Pure Python line score — fast for single small boards."""
    player_squares = set()
    for r in range(height):
        for c in range(width):
            if board[r][c] == player_mark:
                player_squares.add((r, c))

    if not player_squares:
        return 0

    def can_reach_edge(start):
        if start[0] in (0, height - 1) or start[1] in (0, width - 1):
            return True
        queue = [start]
        visited = {start}
        while queue:
            r, c = queue.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in player_squares and (nr, nc) not in visited:
                        if nr in (0, height - 1) or nc in (0, width - 1):
                            return True
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return False

    score = 0
    for cells in [_diags1_coords, _diags2_coords]:
        for diag_cells in cells:
            current = []
            for r, c in diag_cells:
                if (r, c) in player_squares:
                    current.append((r, c))
                else:
                    if len(current) >= 2:
                        if can_reach_edge(current[0]):
                            score += len(current)
                    current = []
            if len(current) >= 2:
                if can_reach_edge(current[0]):
                    score += len(current)
    return score


class lines_game:
    """Single game instance using pure Python for speed on small boards."""

    def __init__(self):
        self.board = [[non_playable_square] * width for _ in range(height)]
        self.scores = [0, 0]
        self.move_count = 0
        self.finished = False

        for r in range(height):
            for c in range(width):
                if (r + c) % 2 == 0:
                    self.board[r][c] = playable_square

    def make_move(self, move_player_0, move_player_1):
        if self.finished:
            return

        mask_0 = self.get_legal_move_mask(0)
        mask_1 = self.get_legal_move_mask(1)

        if mask_0[move_player_0[0]][move_player_0[1]] != 1 or mask_1[move_player_1[0]][move_player_1[1]] != 1:
            return

        if move_player_0 == move_player_1:
            r, c = move_player_0
            self.board[r][c] = removed_square
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr + nc) % 2 == 0:
                        self.board[nr][nc] = removed_square
        else:
            self.board[move_player_0[0]][move_player_0[1]] = player_0_mark
            self.board[move_player_1[0]][move_player_1[1]] = player_1_mark

        self.move_count += 1
        self.scores = self._calculate_scores()
        self.check_game_end()

    def get_legal_move_mask(self, player):
        mask = [[0.0] * width for _ in range(height)]
        for r in range(height):
            for c in range(width):
                if self.board[r][c] == playable_square:
                    mask[r][c] = 1.0

        if self.move_count == 0:
            half = width // 2
            if player == 0:
                for r in range(height):
                    for c in range(half, width):
                        mask[r][c] = 0.0
            elif player == 1:
                for r in range(height):
                    for c in range(0, half):
                        mask[r][c] = 0.0
        return mask

    def make_move_from_distributions(self, dist_p0, dist_p1):
        mask_0 = self.get_legal_move_mask(0)
        mask_1 = self.get_legal_move_mask(1)

        flat_0 = [0.0] * (height * width)
        flat_1 = [0.0] * (height * width)
        idx = 0
        for r in range(height):
            row0 = mask_0[r]
            row1 = mask_1[r]
            for c in range(width):
                flat_0[idx] = row0[c]
                flat_1[idx] = row1[c]
                idx += 1

        total_0 = sum(flat_0)
        total_1 = sum(flat_1)
        if total_0 > 0 and total_1 > 0:
            idx_0 = random.choices(range(len(flat_0)), weights=flat_0, k=1)[0]
            idx_1 = random.choices(range(len(flat_1)), weights=flat_1, k=1)[0]
            self.make_move((idx_0 // width, idx_0 % width), (idx_1 // width, idx_1 % width))

    def get_encoded_state(self, player):
        bt = torch.tensor(self.board, dtype=torch.int8)
        encoding = torch.zeros((7, height, width))
        encoding.scatter_(0, bt.long().unsqueeze(0), 1.0)

        playable_t = (bt == playable_square).float()
        if self.move_count == 0:
            half = width // 2
            if player == 0:
                playable_t[:, half:] = 0
            elif player == 1:
                playable_t[:, :half] = 0
        encoding[1] = playable_t

        norm = width * height / 2
        if player == 0:
            encoding[5] = self.scores[0] / norm
            encoding[6] = self.scores[1] / norm
        elif player == 1:
            encoding[6] = self.scores[0] / norm
            encoding[5] = self.scores[1] / norm
        return encoding

    def _calculate_scores(self):
        s0 = _calc_line_scores_python(self.board, player_0_mark)
        s1 = _calc_line_scores_python(self.board, player_1_mark)
        return [s0, s1]

    def check_game_end(self):
        for r in range(height):
            for c in range(width):
                if self.board[r][c] == playable_square:
                    return
        self.finished = True

    def print_board(self):
        print(self.scores)
        border = "+" + "-" * (width * 2) + "+"
        print(border)
        for col in range(height):
            line = "|"
            for row in range(width):
                v = self.board[col][row]
                if v == non_playable_square:
                    line += "  "
                elif v == playable_square:
                    line += "--"
                elif v == removed_square:
                    line += "##"
                elif v == player_0_mark:
                    line += "||"
                elif v == player_1_mark:
                    line += "oo"
            line += "|"
            print(line)
        print(border)


class BatchedLinesGame:
    """N games in parallel with fully vectorized tensor operations."""

    def __init__(self, num_games, device='cpu'):
        self.n = num_games
        self.device = device
        self.boards = torch.full(
            (num_games, height, width), non_playable_square,
            dtype=torch.int8, device=device
        )
        self.boards[:, _playable_mask] = playable_square

        self.scores = torch.zeros(num_games, 2, dtype=torch.float32, device=device)
        self.move_counts = torch.zeros(num_games, dtype=torch.int32, device=device)
        self.finished = torch.zeros(num_games, dtype=torch.bool, device=device)
        self.half_width = width // 2

    def step(self, dist_p0, dist_p1):
        """One move for all active games. dist_p0/p1: (N, H, W) on self.device."""
        active = ~self.finished
        if not active.any():
            return

        # Legal move masks
        playable = (self.boards == playable_square)
        masks_0 = playable.float()
        masks_1 = playable.float()
        first = self.move_counts == 0
        if first.any():
            masks_0[first, :, self.half_width:] = 0
            masks_1[first, :, :self.half_width] = 0

        # Mask + normalize
        md_0 = dist_p0 * masks_0
        md_1 = dist_p1 * masks_1
        md_0[~active] = 0
        md_1[~active] = 0
        flat_0 = md_0.view(self.n, -1)
        flat_1 = md_1.view(self.n, -1)
        sums_0 = flat_0.sum(1, keepdim=True).clamp(min=1e-8)
        sums_1 = flat_1.sum(1, keepdim=True).clamp(min=1e-8)
        prob_0 = flat_0 / sums_0
        prob_1 = flat_1 / sums_1
        finished_no_moves_0 = ~active & (sums_0.squeeze(1) < 1e-7)
        if finished_no_moves_0.any():
            prob_0[finished_no_moves_0] = 1.0 / prob_0.shape[1]
        finished_no_moves_1 = ~active & (sums_1.squeeze(1) < 1e-7)
        if finished_no_moves_1.any():
            prob_1[finished_no_moves_1] = 1.0 / prob_1.shape[1]

        # Sample moves
        idx_0 = torch.multinomial(prob_0, 1).squeeze(1)
        idx_1 = torch.multinomial(prob_1, 1).squeeze(1)
        r0, c0 = idx_0 // width, idx_0 % width
        r1, c1 = idx_1 // width, idx_1 % width

        # Apply moves
        collision = (r0 == r1) & (c0 == c1)

        nc = active & ~collision
        if nc.any():
            i = nc.nonzero(as_tuple=True)[0]
            self.boards[i, r0[i], c0[i]] = player_0_mark
            self.boards[i, r1[i], c1[i]] = player_1_mark

        ca = active & collision
        if ca.any():
            i = ca.nonzero(as_tuple=True)[0]
            cr, cc = r0[i], c0[i]
            self.boards[i, cr, cc] = removed_square
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc2 = cr + dr, cc + dc
                    valid = (nr >= 0) & (nr < height) & (nc2 >= 0) & (nc2 < width)
                    playable_diag = ((nr + nc2) % 2 == 0)
                    m = valid & playable_diag
                    if m.any():
                        self.boards[i[m], nr[m], nc2[m]] = removed_square

        self.move_counts[active] += 1
        self._update_scores()
        has_playable = (self.boards == playable_square).view(self.n, -1).any(dim=1)
        self.finished = self.finished | ~has_playable

    def _update_scores(self):
        for p, pmark in enumerate([player_0_mark, player_1_mark]):
            self.scores[:, p] = self._batched_line_score(pmark)

    def _batched_line_score(self, player_mark):
        """Vectorized line score using board shearing.

        Shear the board so diagonals become columns. Contiguous marks along
        a diagonal become contiguous vertical runs. Use cumsum-with-reset
        along rows to find run lengths, score runs that reach the edge.
        """
        player_mask = (self.boards == player_mark).float()
        reached = self._batched_flood_fill(player_mask.bool()).float()

        scores = torch.zeros(self.n, dtype=torch.float32, device=self.device)

        for col_idx_safe, valid_mask, score_cols in [
            (_nesw_col_safe, _nesw_valid, _nesw_score_cols_t),
            (_nwse_col_safe, _nwse_valid, _nwse_score_cols_t),
        ]:
            # Gather into sheared layout: (N, H, sw)
            idx = col_idx_safe.unsqueeze(0).expand(self.n, -1, -1)
            vm = valid_mask.unsqueeze(0).float()
            sheared_marks = torch.gather(player_mask, 2, idx) * vm
            sheared_reached = torch.gather(reached, 2, idx) * vm

            # Only columns that can have runs >= 2
            sheared_marks = sheared_marks[:, :, score_cols]
            sheared_reached = sheared_reached[:, :, score_cols]

            # Cumsum-with-reset along rows (dim 1)
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

            # Run ends: mark[r] > 0 and mark[r+1] == 0 (or last row)
            is_end = torch.zeros_like(sheared_marks, dtype=torch.bool)
            is_end[:, :-1, :] = (sheared_marks[:, :-1, :] > 0) & (sheared_marks[:, 1:, :] == 0)
            is_end[:, -1, :] = sheared_marks[:, -1, :] > 0

            # Score valid runs (length >= 2, reaches edge)
            valid = is_end & (cumlen >= 2) & (cumreach > 0)
            scores += (cumlen * valid.float()).sum(dim=(1, 2))

        return scores

    def _batched_flood_fill(self, player_mask):
        """8-connected flood fill from edges, batched across all games."""
        reached = player_mask & _edge_mask.unsqueeze(0)

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

        return reached

    def get_encoded_states(self, player):
        """Get encoded states. Returns (N, 7, H, W)."""
        encoding = torch.zeros(self.n, 7, height, width, device=self.device)
        encoding.scatter_(1, self.boards.long().unsqueeze(1), 1.0)

        playable = (self.boards == playable_square).float()
        first = self.move_counts == 0
        if first.any():
            if player == 0:
                playable[first, :, self.half_width:] = 0
            elif player == 1:
                playable[first, :, :self.half_width] = 0
        encoding[:, 1] = playable

        norm = width * height / 2
        if player == 0:
            encoding[:, 5] = self.scores[:, 0] / norm
            encoding[:, 6] = self.scores[:, 1] / norm
        elif player == 1:
            encoding[:, 6] = self.scores[:, 0] / norm
            encoding[:, 5] = self.scores[:, 1] / norm

        return encoding
