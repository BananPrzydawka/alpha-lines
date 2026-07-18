from game import batched_lines_game
import numpy as np

numbers = [1024, 2*1024, 4*1024, 8*1024, 16*1024, 32*1024]

def main():
    for n in numbers:
        game = batched_lines_game(n)
        while not game.finished.all():
            game.distribution_step(np.ones([n, 10, 16]), np.ones([n, 10, 16]))

        print(f"done with the games lol")
        values, counts = np.unique(game.move_counts, return_counts=True)
        for v, c in zip(values, counts):
            print(f"{v}: {c}")