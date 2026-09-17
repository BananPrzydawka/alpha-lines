"""ctypes bridge to the single-threaded Rust batch, without NumPy or workers."""
import ctypes

import torch
from config import height, width


class RandomScoreBatch:
    def __init__(self, library, batch, seed):
        if batch < 1 or not 0 <= seed < 2**64:
            raise ValueError("batch must be positive and seed must fit u64")
        self.lib = ctypes.CDLL(str(library))
        self.lib.score_batch_new.argtypes = [ctypes.c_size_t, ctypes.c_uint64]
        self.lib.score_batch_new.restype = ctypes.c_void_p
        self.lib.score_batch_step.argtypes = [ctypes.c_void_p] * 3
        self.lib.score_batch_step.restype = ctypes.c_uint64
        self.lib.score_batch_free.argtypes = [ctypes.c_void_p]
        self.lib.score_batch_free.restype = None
        self.handle = self.lib.score_batch_new(batch, seed)
        if not self.handle:
            raise RuntimeError("Rust batch allocation failed")
        self.cells = torch.empty(batch, 80, dtype=torch.uint8)
        self.targets = torch.empty(batch, 2, dtype=torch.int64)
        self.completed = 0

    def step(self):
        """Returned tensors are borrowed buffers, overwritten by the next step."""
        if not self.handle:
            raise RuntimeError("batch is closed")
        self.completed += self.lib.score_batch_step(
            self.handle, self.cells.data_ptr(), self.targets.data_ptr())
        return self.cells, self.targets

    def close(self):
        if getattr(self, "handle", None):
            self.lib.score_batch_free(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ScoreEncoder(torch.nn.Module):
    """Five board planes in the existing encoder's order, fixed player 0 view.

    Unplayable, empty, removed, player 0, player 1. No scores or scorer levels
    enter this module. The opening legal mask is irrelevant to score prediction.
    """
    def __init__(self):
        super().__init__()
        squares = torch.arange(80)
        self.register_buffer("indices", squares // 8 * 16 + 2 * (squares % 8) + (squares // 8 % 2))
        self.register_buffer("marks", torch.tensor([1, 3, 4, 2], dtype=torch.uint8))
        self.register_buffer("values", torch.arange(5).view(1, 5, 1, 1))

    def forward(self, cells):
        board = cells.new_zeros((cells.shape[0], height * width))
        board.scatter_(1, self.indices.expand(cells.shape[0], -1), self.marks[cells.long()])
        return (board.view(-1, 1, height, width) == self.values).float().contiguous(
            memory_format=torch.channels_last)
