import numpy as np
import torch
from numba import njit, uint8, int32, boolean
from config import height, width

# Constants
non_playable_square = 0
playable_square = 1
removed_square = 2
player_0_mark = 3
player_1_mark = 4

# Eager compilation signature
@njit('int32[:](uint8[:,::1], int32, int32, boolean[:,::1], boolean[:,::1], int32[:])', cache=True)
def _numba_logic(board, height, width, connected_p0, connected_p1, stack):
    scores = np.zeros(2, dtype=int32)
    
    for p_idx in range(2):
        mark = player_0_mark if p_idx == 0 else player_1_mark
        conn = connected_p0 if p_idx == 0 else connected_p1
        conn.fill(False)
        top = 0
        
        # 1. Optimized O(H+W) Boundary Seed Scan
        for c in range(width):
            if board[0, c] == mark:
                conn[0, c] = True
                stack[top] = c
                top += 1
            if board[height-1, c] == mark and not conn[height-1, c]:
                conn[height-1, c] = True
                stack[top] = (height - 1) * width + c
                top += 1
                
        for r in range(1, height-1):
            if board[r, 0] == mark and not conn[r, 0]:
                conn[r, 0] = True
                stack[top] = r * width
                top += 1
            if board[r, width-1] == mark and not conn[r, width-1]:
                conn[r, width-1] = True
                stack[top] = r * width + (width - 1)
                top += 1
        
        # 2. BFS
        ptr = 0
        while ptr < top:
            curr = stack[ptr]
            ptr += 1
            r = curr // width
            c = curr % width
            
            r_min = r - 1 if r > 0 else 0
            r_max = r + 1 if r < height - 1 else height - 1
            c_min = c - 1 if c > 0 else 0
            c_max = c + 1 if c < width - 1 else width - 1
            
            for nr in range(r_min, r_max + 1):
                for nc in range(c_min, c_max + 1):
                    if board[nr, nc] == mark and not conn[nr, nc]:
                        conn[nr, nc] = True
                        stack[top] = nr * width + nc
                        top += 1

        # 3. Fast Scanning for active lines connected to the edge
        p_score = 0
        
        # Main diagonals (\)
        for s in range(-(width - 1), height):
            count = 0
            is_conn = False
            r_start = s if s > 0 else 0
            r_end = height if (width + s) > height else (width + s)
            
            for r in range(r_start, r_end):
                c = r - s
                if board[r, c] == mark:
                    if count == 0: 
                        is_conn = conn[r, c]
                    count += 1
                else:
                    if count > 1 and is_conn: 
                        p_score += count
                    count = 0
            if count > 1 and is_conn: 
                p_score += count

        # Anti-diagonals (/)
        for s in range(0, height + width - 1):
            count = 0
            is_conn = False
            r_start = s - width + 1 if s - width + 1 > 0 else 0
            r_end = s + 1 if s + 1 < height else height
            
            for r in range(r_start, r_end):
                c = s - r
                if board[r, c] == mark:
                    if count == 0: 
                        is_conn = conn[r, c]
                    count += 1
                else:
                    if count > 1 and is_conn: 
                        p_score += count
                    count = 0
            if count > 1 and is_conn: 
                p_score += count
        
        scores[p_idx] = p_score
    return scores

# Pre-compute structural data
_Y, _X = np.ogrid[:height, :width]
_CHECKER_MASK = (_Y + _X) % 2 == 0

class lines_game:
    def __init__(self):
        self.board_np = np.zeros((height, width), dtype=np.uint8)
        self.scores = [0, 0]
        self.move_count = 0
        self.finished = False
        self.mid = width // 2
        
        self.conn_p0 = np.zeros((height, width), dtype=np.bool_)
        self.conn_p1 = np.zeros((height, width), dtype=np.bool_)
        self.stack_buf = np.empty(height * width, dtype=np.int32)

        self.board_np[_CHECKER_MASK] = playable_square
        self.playable_count = int(_CHECKER_MASK.sum())
        
        self._cached_mask_np = np.zeros((height, width), dtype=np.float32)
        self._cached_mask_np[_CHECKER_MASK] = 1.0

    def make_move(self, move_player_0, move_player_1):
        if self.finished:
            print("game over, no more moves can be made")
            return

        r0, c0 = move_player_0
        r1, c1 = move_player_1

        mask_0 = self.get_legal_move_mask(0)
        mask_1 = self.get_legal_move_mask(1)

        if mask_0[r0, c0] != 1 or mask_1[r1, c1] != 1:
            print("illegal values passed to make_move")
            return

        if move_player_0 == move_player_1:
            r, c = move_player_0
            collision_squares = [(r, c)]
            diagonals = [(r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1)]
            for dr, dc in diagonals:
                if 0 <= dr < height and 0 <= dc < width:
                    if (dr + dc) % 2 == 0:
                        collision_squares.append((dr, dc))
            
            for nr, nc in collision_squares:
                # Decrement playable count only if the square was still playable
                if self.board_np[nr, nc] == playable_square:
                    self.playable_count -= 1
                    self._cached_mask_np[nr, nc] = 0.0
                
                # Unconditionally overwrite, allowing lines to break and scores to fall
                self.board_np[nr, nc] = removed_square
        else:
            self.board_np[r0, c0] = player_0_mark
            self.board_np[r1, c1] = player_1_mark
            self._cached_mask_np[r0, c0] = 0.0
            self._cached_mask_np[r1, c1] = 0.0
            self.playable_count -= 2

        self.move_count += 1
        res = _numba_logic(self.board_np, height, width, self.conn_p0, self.conn_p1, self.stack_buf)
        self.scores[0], self.scores[1] = res[0], res[1]
        
        self.check_game_end()

    def get_legal_move_mask(self, player):
        mask = self._cached_mask_np.copy()
        if self.move_count == 0:
            if player == 0: 
                mask[:, self.mid:] = 0.0
            elif player == 1: 
                mask[:, :self.mid] = 0.0
        return torch.from_numpy(mask)

    def make_move_from_distributions(self, distribution_player_0, distribution_player_1):
        masked_dist_0 = distribution_player_0 * self.get_legal_move_mask(0)
        masked_dist_1 = distribution_player_1 * self.get_legal_move_mask(1)
        
        prob_dist_0 = masked_dist_0 / masked_dist_0.sum()
        prob_dist_1 = masked_dist_1 / masked_dist_1.sum()
        
        flat_idx_0 = torch.multinomial(prob_dist_0.view(-1), 1).item()
        flat_idx_1 = torch.multinomial(prob_dist_1.view(-1), 1).item()
        
        move_0 = (flat_idx_0 // width, flat_idx_0 % width)
        move_1 = (flat_idx_1 // width, flat_idx_1 % width)
        
        self.make_move(move_0, move_1)

    def get_encoded_state(self, player):
        board_t = torch.from_numpy(self.board_np.astype(np.int64))
        encoding = torch.zeros((7, height, width), dtype=torch.float32)
        encoding.scatter_(0, board_t.unsqueeze(0), 1.0)
        
        encoding[1] = self.get_legal_move_mask(player)
        
        norm = (width * height) / 2.0
        s0, s1 = self.scores
        if player == 0:
            encoding[5, :, :], encoding[6, :, :] = s0 / norm, s1 / norm
        elif player == 1:
            encoding[6, :, :], encoding[5, :, :] = s0 / norm, s1 / norm
        return encoding

    def check_game_end(self):
        if self.playable_count <= 0:
            self.finished = True

    def print_board(self):
        print(self.scores)
        border = "+" + "-" * (width * 2) + "+"
        print(border)
        for col in range(height):
            line = "|"
            for row in range(width):
                if self.board_np[col, row] == non_playable_square:
                    line += "  "
                elif self.board_np[col, row] == playable_square:
                    line += "--"
                elif self.board_np[col, row] == removed_square:
                    line += "##"
                elif self.board_np[col, row] == player_0_mark:
                    line += "||"
                elif self.board_np[col, row] == player_1_mark:
                    line += "oo"
            line += "|"
            print(line)
        print(border)