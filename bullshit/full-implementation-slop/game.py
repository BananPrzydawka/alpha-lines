"""
Game logic for Lines - implements the game rules for self-play
"""

import numpy as np
from config import BOARD_HEIGHT, BOARD_WIDTH, MAX_MOVES, MAX_SCORE


def get_playable_squares():
    """Get all squares on one diagonal set where (row + col) is even"""
    playable = set()
    for row in range(BOARD_HEIGHT):
        for col in range(BOARD_WIDTH):
            if (row + col) % 2 == 0:
                playable.add((row, col))
    return playable


PLAYABLE_SQUARES = get_playable_squares()
PLAYABLE_SQUARES_LIST = sorted(list(PLAYABLE_SQUARES))


def is_on_edge(pos):
    """Check if a position is on the edge of the board"""
    r, c = pos
    return r == 0 or r == BOARD_HEIGHT - 1 or c == 0 or c == BOARD_WIDTH - 1


def can_reach_edge(start_pos, player_squares):
    """BFS to check if a mark can reach the edge via adjacent marks"""
    if is_on_edge(start_pos):
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
                if (nr, nc) in player_squares and (nr, nc) not in visited:
                    if is_on_edge((nr, nc)):
                        return True
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return False


def calculate_scores(board):
    """
    Calculate scores based on diagonal lines connected to the edge.
    Returns (score_x, score_o)
    """
    scores = {'X': 0, 'O': 0}
    
    # Get player squares
    player_squares_x = {sq for sq, val in board.items() if val == 1}  # 1 = X
    player_squares_o = {sq for sq, val in board.items() if val == -1}  # -1 = O
    
    for player, squares in [('X', player_squares_x), ('O', player_squares_o)]:
        if not squares:
            continue
        
        detected_lines = []
        
        # 1. Leftward diagonals (row + col = constant, incrementing by 2)
        for s in range(0, BOARD_HEIGHT + BOARD_WIDTH, 2):
            current_line = []
            for r in range(BOARD_HEIGHT):
                c = s - r
                if 0 <= c < BOARD_WIDTH:
                    if (r, c) in squares:
                        current_line.append((r, c))
                    else:
                        if len(current_line) >= 2:
                            detected_lines.append(list(current_line))
                        current_line = []
            if len(current_line) >= 2:
                detected_lines.append(list(current_line))
        
        # 2. Rightward diagonals (row - col = constant, incrementing by 2)
        for d in range(-BOARD_WIDTH, BOARD_HEIGHT + 1):
            if d % 2 != 0:
                continue
            current_line = []
            for r in range(BOARD_HEIGHT):
                c = r - d
                if 0 <= c < BOARD_WIDTH:
                    if (r, c) in squares:
                        current_line.append((r, c))
                    else:
                        if len(current_line) >= 2:
                            detected_lines.append(list(current_line))
                        current_line = []
            if len(current_line) >= 2:
                detected_lines.append(list(current_line))
        
        # 3. Check if each line is connected to the edge
        for line in detected_lines:
            if can_reach_edge(line[0], squares):
                scores[player] += len(line)
    
    return scores['X'], scores['O']


class LinesGame:
    """
    Game state for Lines with simultaneous moves.
    board: dict mapping (row, col) to 1 (X) or -1 (O), 0 means empty
    removed_squares: set of (row, col) that have been removed
    moves_made: number of completed turns (pairs of moves)
    """
    
    def __init__(self):
        self.board = {}  # (row, col) -> 1 or -1
        self.removed_squares = set()
        self.moves_made = 0
        self.game_over = False
        self.winner = 0  # 0 = draw, 1 = X, -1 = O
        self.final_score_x = 0
        self.final_score_o = 0
    
    def get_legal_moves(self, player):
        """
        Get all legal moves for a specific player.
        player: 1 for X, -1 for O
        """
        if self.game_over:
            return []
        
        legal = []
        for sq in PLAYABLE_SQUARES_LIST:
            if sq in self.removed_squares:
                continue
            if sq in self.board:
                continue
            
            # First move restriction
            if self.moves_made == 0:
                if player == 1 and sq[1] >= BOARD_WIDTH // 2:
                    continue
                if player == -1 and sq[1] < BOARD_WIDTH // 2:
                    continue
            
            legal.append(sq)
        
        return legal
    
    def get_legal_moves_mask(self, player):
        """Get a binary mask of legal moves for a player (10x16)"""
        mask = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
        for sq in self.get_legal_moves(player):
            mask[sq] = 1.0
        return mask
    
    def execute_moves(self, move_x, move_o):
        """
        Execute both players' moves simultaneously.
        move_x: (row, col) for player X, or None if no legal moves
        move_o: (row, col) for player O, or None if no legal moves
        
        Returns True if moves were executed, False if game ended.
        """
        if self.game_over:
            return False
        
        # Handle case where one or both players have no legal moves
        if move_x is None and move_o is None:
            self._end_game()
            return False
        
        if move_x is not None and move_o is not None:
            if move_x == move_o:
                # Collision! Remove the square and 4 diagonal neighbors
                collision_squares = [move_x]
                r, c = move_x
                diagonals = [
                    (r-1, c-1), (r-1, c+1), (r+1, c-1), (r+1, c+1)
                ]
                for dr, dc in diagonals:
                    if (dr, dc) in PLAYABLE_SQUARES:
                        collision_squares.append((dr, dc))
                
                # Remove squares (remove any existing marks)
                for sq in collision_squares:
                    if sq in self.board:
                        del self.board[sq]
                    self.removed_squares.add(sq)
            else:
                # No collision, place both marks
                self.board[move_x] = 1  # X
                self.board[move_o] = -1  # O
        
        elif move_x is not None:
            # Only X can move (O has no legal moves)
            self.board[move_x] = 1
        elif move_o is not None:
            # Only O can move (X has no legal moves)
            self.board[move_o] = -1
        
        self.moves_made += 1
        
        # Check for game end
        self._check_game_end()
        
        return True
    
    def _check_game_end(self):
        """Check if the game has ended"""
        if self.moves_made >= MAX_MOVES:
            self._end_game()
            return
        
        # Check if there are any legal moves left for either player
        total_playable = len(PLAYABLE_SQUARES) - len(self.removed_squares)
        occupied = len(self.board)
        if occupied >= total_playable:
            self._end_game()
    
    def _end_game(self):
        """End the game and calculate scores"""
        self.game_over = True
        self.final_score_x, self.final_score_o = calculate_scores(self.board)
        
        if self.final_score_x > self.final_score_o:
            self.winner = 1
        elif self.final_score_o > self.final_score_x:
            self.winner = -1
        else:
            self.winner = 0
    
    def get_state_encoding(self, perspective=1):
        """
        Encode the game state as a 6-plane tensor.
        perspective: 1 for X's perspective, -1 for O's perspective
        
        Planes:
        0. Legal moves mask (for the current player from this perspective)
        1. Removed squares
        2. Current player's marks (from perspective)
        3. Opponent's marks (from perspective)
        4. Current player's score (normalized)
        5. Opponent's score (normalized)
        """
        state = np.zeros((6, BOARD_HEIGHT, BOARD_WIDTH), dtype=np.float32)
        
        # Determine which player from perspective
        if perspective == 1:
            current_player = 1  # X
            opponent_player = -1  # O
            my_score = self.final_score_x if self.game_over else self._get_current_score(1)
            opponent_score = self.final_score_o if self.game_over else self._get_current_score(-1)
        else:
            current_player = -1  # O
            opponent_player = 1  # X
            my_score = self.final_score_o if self.game_over else self._get_current_score(-1)
            opponent_score = self.final_score_x if self.game_over else self._get_current_score(1)
        
        # Plane 0: Legal moves for current player
        state[0] = self.get_legal_moves_mask(current_player)
        
        # Plane 1: Removed squares
        for sq in self.removed_squares:
            state[1, sq[0], sq[1]] = 1.0
        
        # Plane 2: Current player's marks
        for sq, val in self.board.items():
            if val == current_player:
                state[2, sq[0], sq[1]] = 1.0
        
        # Plane 3: Opponent's marks
        for sq, val in self.board.items():
            if val == opponent_player:
                state[3, sq[0], sq[1]] = 1.0
        
        # Plane 4: Current player's score (normalized)
        state[4, :, :] = my_score / MAX_SCORE
        
        # Plane 5: Opponent's score (normalized)
        state[5, :, :] = opponent_score / MAX_SCORE
        
        return state
    
    def _get_current_score(self, player):
        """Get current score for a player (only valid during game, not at end)"""
        if player == 1:
            score_x, _ = calculate_scores(self.board)
            return score_x
        else:
            _, score_o = calculate_scores(self.board)
            return score_o
    
    def get_point_difference(self):
        """
        Get the point difference from X's perspective.
        Positive means X is ahead, negative means O is ahead.
        """
        if not self.game_over:
            # During game, use current scores
            score_x, score_o = calculate_scores(self.board)
        else:
            score_x = self.final_score_x
            score_o = self.final_score_o
        return score_x - score_o
    
    def clone(self):
        """Create a deep copy of the game state"""
        new_game = LinesGame()
        new_game.board = dict(self.board)
        new_game.removed_squares = set(self.removed_squares)
        new_game.moves_made = self.moves_made
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.final_score_x = self.final_score_x
        new_game.final_score_o = self.final_score_o
        return new_game


def print_board(game):
    """
    Print the board to the terminal compactly.
    x for X marks, o for O marks, - for playable diagonal,
    space for non-playable diagonal, # for removed spots.
    """
    border = "+" + "-" * (BOARD_WIDTH) + "+"
    print(border)
    for row in range(BOARD_HEIGHT):
        line = "|"
        for col in range(BOARD_WIDTH):
            if (row, col) in game.removed_squares:
                line += "#"
            elif (row, col) in game.board:
                val = game.board[(row, col)]
                line += ("x" if val == 1 else "o")
            elif (row, col) in PLAYABLE_SQUARES:
                line += "-"
            else:
                line += " "
        # line = line.rstrip() + "|"
        line += "|"
        print(line)
    print(border)


def flip_board_double(state):
    """
    Flip the board both horizontally and vertically.
    This preserves the diagonal geometry.
    state: numpy array of shape (C, H, W) or (H, W)
    Returns flipped state.
    """
    # Flip both dimensions: horizontal (axis -1) and vertical (axis -2)
    return state[..., ::-1, ::-1]


def flip_policy_double(policy):
    """
    Flip the policy (10x16) both horizontally and vertically.
    policy: numpy array of shape (H, W) or (N, H, W)
    """
    return policy[..., ::-1, ::-1]