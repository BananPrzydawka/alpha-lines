import torch

from game import lines_game
# from model import alpha_lines_net


game = lines_game()

# model = alpha_lines_net()

# def play_test_game():
#     while game.finished is not True:
#         encoded_game_0 = game.get_encoded_state(player=0)
#         policy_0, value_0, value_error_0, points_0 = model.forward(encoded_game_0, True)

#         encoded_game_1 = game.get_encoded_state(player=0)
#         policy_1, value_1, value_error_1, points_1 = model.forward(encoded_game_1, True)

#         game.make_move_from_distributions(policy_0.squeeze(), policy_1.squeeze())
#         game.print_board()

while game.finished is not True:
    game.make_move_from_distributions(torch.ones(10, 16), torch.ones(10, 16))
    game.print_board()