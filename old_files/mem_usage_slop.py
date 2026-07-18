import sys
import torch
import gc

from game_slop import lines_game, batched_lines_game

def get_total_size(obj, seen=None):
    """Recursively finds the size of objects including nested structures."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([get_total_size(v, seen) for v in obj.values()])
        size += sum([get_total_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_total_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_total_size(i, seen) for i in obj])
    
    # Special handling for PyTorch Tensors
    if torch.is_tensor(obj):
        size += obj.element_size() * obj.nelement()
        
    return size

# Example measurement
# Assuming height=10, width=10 for demonstration
single_game = lines_game()
second_game = lines_game()
batched_game = batched_lines_game(num_games=1500)

print(f"Single Game Size: {get_total_size(single_game) / 1024:.2f} KB")
print(f"Second Game Size: {get_total_size(second_game) / 1024:.2f} KB")
# print(f"Batched Games (100) Size: {get_total_size(batched_game) / 1024:.2f} KB")
print(f"in game batch: {(get_total_size(batched_game) / 100) / 1024:.2f} KB")