import numpy as np
from numba import njit, prange
from config import game_kernels_parralel

NON_PLAYABLE_SQUARE = 0
PLAYABLE_SQUARE = 1
REMOVED_SQUARE = 2
PLAYER_0_MARK = 3
PLAYER_1_MARK = 4


@njit(cache=True, parallel=game_kernels_parralel)
def _score_player(boards, g, player_mark, height, width):
    """8-connected flood fill from the border + diagonal run-length scan, for one game/player."""
    player_mask = np.empty((height, width), dtype=np.bool_)
    for r in range(height):
        for c in range(width):
            player_mask[r, c] = boards[g, r, c] == player_mark

    reached = np.zeros((height, width), dtype=np.bool_)
    stack_r = np.empty(height * width, dtype=np.int64)
    stack_c = np.empty(height * width, dtype=np.int64)
    sp = 0
    for r in range(height):
        for c in range(width):
            if player_mask[r, c] and (r == 0 or r == height - 1 or c == 0 or c == width - 1):
                reached[r, c] = True
                stack_r[sp] = r
                stack_c[sp] = c
                sp += 1
    while sp > 0:
        sp -= 1
        r = stack_r[sp]
        c = stack_c[sp]
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and player_mask[nr, nc] and not reached[nr, nc]:
                    reached[nr, nc] = True
                    stack_r[sp] = nr
                    stack_c[sp] = nc
                    sp += 1

    score = 0.0
    # anti-diagonals: r + c = d
    for d in range(height + width - 1):
        cur_len, cur_reach = 0, False
        r0 = max(0, d - (width - 1))
        r1 = min(height - 1, d)
        for r in range(r0, r1 + 1):
            c = d - r
            if player_mask[r, c]:
                cur_len += 1
                cur_reach = cur_reach or reached[r, c]
            else:
                if cur_len >= 2 and cur_reach:
                    score += cur_len
                cur_len, cur_reach = 0, False
        if cur_len >= 2 and cur_reach:
            score += cur_len

    # main diagonals: c - r = e
    for e in range(-(height - 1), width):
        cur_len, cur_reach = 0, False
        r0 = max(0, -e)
        r1 = min(height - 1, width - 1 - e)
        for r in range(r0, r1 + 1):
            c = r + e
            if player_mask[r, c]:
                cur_len += 1
                cur_reach = cur_reach or reached[r, c]
            else:
                if cur_len >= 2 and cur_reach:
                    score += cur_len
                cur_len, cur_reach = 0, False
        if cur_len >= 2 and cur_reach:
            score += cur_len

    return score


@njit(cache=True, parallel=game_kernels_parralel)
def score_batch(boards, player_mark, height, width):
    """Scores every game in the batch for player_mark, regardless of active/finished state.
    Used once at import/construction time; the hot loop uses apply_and_score_kernel instead,
    which only rescoring games whose boards actually changed this step."""
    n = boards.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    for g in prange(n):
        scores[g] = _score_player(boards, g, player_mark, height, width)
    return scores


@njit(cache=True, parallel=game_kernels_parralel)
def legal_masks_kernel(boards, move_counts, half_width, height, width):
    """(N, H, W) float legal-move masks + (N,) counts for both players, honoring the
    first-move half-board rule."""
    n = boards.shape[0]
    mask_0 = np.zeros((n, height, width), dtype=np.float32)
    mask_1 = np.zeros((n, height, width), dtype=np.float32)
    count_0 = np.zeros(n, dtype=np.float32)
    count_1 = np.zeros(n, dtype=np.float32)
    for g in prange(n):
        first = move_counts[g] == 0
        c0_total, c1_total = 0.0, 0.0
        for r in range(height):
            for c in range(width):
                if boards[g, r, c] == PLAYABLE_SQUARE:
                    if not (first and c >= half_width):
                        mask_0[g, r, c] = 1.0
                        c0_total += 1.0
                    if not (first and c < half_width):
                        mask_1[g, r, c] = 1.0
                        c1_total += 1.0
        count_0[g] = c0_total
        count_1[g] = c1_total
    return mask_0, mask_1, count_0, count_1


@njit(cache=True, parallel=game_kernels_parralel)
def sample_move_kernel(dist, mask, active, height, width):
    """One flat (row, col) move per active game, sampled from dist restricted to legal squares.
    Inactive games get (0, 0) placeholders; the caller never applies moves for inactive games."""
    n = dist.shape[0]
    r_out = np.zeros(n, dtype=np.int64)
    c_out = np.zeros(n, dtype=np.int64)
    for g in prange(n):
        if not active[g]:
            continue
        total = 0.0
        for r in range(height):
            for c in range(width):
                total += dist[g, r, c] * mask[g, r, c]

        if total < 1e-8:
            idx = np.random.randint(0, height * width)
            r_out[g] = idx // width
            c_out[g] = idx % width
            continue

        threshold = np.random.random() * total
        cum = 0.0
        chosen_r, chosen_c = 0, 0
        found = False
        for r in range(height):
            for c in range(width):
                v = dist[g, r, c] * mask[g, r, c]
                if v > 0.0:
                    cum += v
                    if not found and cum >= threshold:
                        chosen_r, chosen_c = r, c
                        found = True
        r_out[g] = chosen_r
        c_out[g] = chosen_c
    return r_out, c_out


@njit(cache=True, parallel=game_kernels_parralel)
def apply_and_score_kernel(boards, move_counts, finished, scores, r0, c0, r1, c1, active, height, width):
    """For each active game: marks the move (or resolves a collision), increments move_counts,
    rescores both players, and updates finished. Fully fused — one Python-level call per step."""
    n = boards.shape[0]
    for g in prange(n):
        if not active[g]:
            continue

        rr0, cc0 = r0[g], c0[g]
        rr1, cc1 = r1[g], c1[g]

        if rr0 == rr1 and cc0 == cc1:
            boards[g, rr0, cc0] = REMOVED_SQUARE
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = rr0 + dr, cc0 + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr + nc) % 2 == 0:
                        boards[g, nr, nc] = REMOVED_SQUARE
        else:
            boards[g, rr0, cc0] = PLAYER_0_MARK
            boards[g, rr1, cc1] = PLAYER_1_MARK

        move_counts[g] += 1

        scores[g, 0] = _score_player(boards, g, PLAYER_0_MARK, height, width)
        scores[g, 1] = _score_player(boards, g, PLAYER_1_MARK, height, width)

        has_playable = False
        for r in range(height):
            for c in range(width):
                if boards[g, r, c] == PLAYABLE_SQUARE:
                    has_playable = True
                    break
            if has_playable:
                break
        if not has_playable:
            finished[g] = True