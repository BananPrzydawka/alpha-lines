import numpy as np
import torch

from config import (height, width)

non_playable_square = 0
playable_square = 1
removed_square = 2
player_0_mark = 3
player_1_mark = 4


class lines_game:
    def __init__(self):
        self.board = torch.full((height, width), non_playable_square)
        self.scores = [0, 0]
        self.move_count = 0
        self.finished = False

        for row in range(width):
            for column in range(height):
                if (row + column) % 2 == 0:
                    self.board[column][row] = playable_square

    def make_move(self, move_player_0, move_player_1):
        if self.finished:
            print("game over, no more moves can be made")
            return

        mask_0 = self.get_legal_move_mask(0)
        mask_1 = self.get_legal_move_mask(1)

        if mask_0[move_player_0] != 1 or mask_1[move_player_1] != 1:
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
            for sq in collision_squares:
                self.board[sq] = removed_square
        else:
            self.board[move_player_0] = player_0_mark
            self.board[move_player_1] = player_1_mark

        self.move_count += 1
        self.scores = self._calculate_scores()
        self.check_game_end()

    def get_legal_move_mask(self, player):
        mask = (self.board == 1).to(torch.float)

        # 0 - player on the left  of the board
        # 1 - player on the right of the board
        if self.move_count == 0:
            if player == 0:
                mask[:, (int(width / 2)) :] = 0
            if player == 1:
                mask[:, : (int(width / 2))] = 0

        return mask

    def make_move_from_distributions(
        self, distribution_player_0, distribution_player_1
    ):

        masked_dist_0 = distribution_player_0 * self.get_legal_move_mask(0)
        masked_dist_1 = distribution_player_1 * self.get_legal_move_mask(1)
        prob_dist_0 = masked_dist_0 / masked_dist_0.sum()
        prob_dist_1 = masked_dist_1 / masked_dist_1.sum()
        flat_idx_0 = torch.multinomial(prob_dist_0.view(-1), 1)
        move_0 = torch.unravel_index(flat_idx_0, prob_dist_0.shape)
        flat_idx_1 = torch.multinomial(prob_dist_1.view(-1), 1)
        move_1 = torch.unravel_index(flat_idx_1, prob_dist_1.shape)

        self.make_move(move_0, move_1)

        pass

    def get_encoded_state(self, player):
        encoding = torch.zeros((7, height, width))
        encoding.scatter_(0, self.board.long().unsqueeze(0), 1.0)

        encoding[1] = self.get_legal_move_mask(player)

        if player == 0:
            encoding[5] = self.scores[0] / (width * height / 2)
            encoding[6] = self.scores[1] / (width * height / 2)
        elif player == 1:
            encoding[6] = self.scores[0] / (width * height / 2)
            encoding[5] = self.scores[1] / (width * height / 2)

        return encoding

    # slop warning!
    def _can_reach_edge(self, start_pos, player_marks):
        if start_pos[0] == 0 or start_pos[0] == height - 1 or start_pos[1] == 0 or start_pos[1] == width - 1:
            return True
        queue = [start_pos]
        visited = {start_pos}
        while queue:
            r, c = queue.pop(0)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in player_marks and (nr, nc) not in visited:
                        if nr == 0 or nr == height - 1 or nc == 0 or nc == width - 1:
                            return True
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return False

    # slop warning!
    def _calculate_scores(self):
        player_0_squares = set()
        player_1_squares = set()
        for row in range(height):
            for col in range(width):
                if self.board[row, col].item() == player_0_mark:
                    player_0_squares.add((row, col))
                elif self.board[row, col].item() == player_1_mark:
                    player_1_squares.add((row, col))

        scores = [0, 0]
        for player, squares in [(0, player_0_squares), (1, player_1_squares)]:
            if not squares:
                continue
            detected_lines = []
            for s in range(0, height + width, 2):
                current_line = []
                for r in range(height):
                    c = s - r
                    if 0 <= c < width:
                        if (r, c) in squares:
                            current_line.append((r, c))
                        else:
                            if len(current_line) >= 2:
                                detected_lines.append(list(current_line))
                            current_line = []
                if len(current_line) >= 2:
                    detected_lines.append(list(current_line))
            for d in range(-width, height + 1):
                if d % 2 != 0:
                    continue
                current_line = []
                for r in range(height):
                    c = r - d
                    if 0 <= c < width:
                        if (r, c) in squares:
                            current_line.append((r, c))
                        else:
                            if len(current_line) >= 2:
                                detected_lines.append(list(current_line))
                            current_line = []
                if len(current_line) >= 2:
                    detected_lines.append(list(current_line))
            for line in detected_lines:
                if self._can_reach_edge(line[0], squares):
                    scores[player] += len(line)
        return scores

    def check_game_end(self):
        playable_square_count = (self.board == playable_square).sum().item()
        if playable_square_count == 0:
            self.finished = True

    def print_board(self):
        print(self.scores)
        border = "+" + "-" * (width * 2) + "+"
        print(border)
        for col in range(height):
            line = "|"
            for row in range(width):
                if self.board[col, row]   == non_playable_square:
                    line += "  "
                elif self.board[col, row] == playable_square:
                    line += "--"
                elif self.board[col, row] == removed_square:
                    line += "##"
                elif self.board[col, row] == player_0_mark:
                    line += "||"
                elif self.board[col, row] == player_1_mark:
                    line += "oo"
            line += "|"
            print(line)
        print(border)