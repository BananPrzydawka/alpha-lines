"""Owned ctypes wrapper around the Rust PUCT collector."""
import ctypes as C
from typing import NamedTuple

import torch


def default_node_capacity(g, s, factor=8):
    """Later calibrated MCTS capacity: factor * g * s; factor 8 was tested."""
    return g * s * factor


class Stats(NamedTuple):
    records: int
    model_calls: int
    evaluated: int
    finished_games: int
    live_nodes: int
    exhausted: int


def check(value):
    if value < 0:
        raise RuntimeError('Rust MCTS operation failed; see the preceding diagnostic')
    return value


class Arena:
    def __init__(self, library, options, *, seed=None, pinned=False):
        self.lib = C.CDLL(str(library))
        signatures = {
            'new': ([C.c_size_t]*3+[C.c_uint32,C.c_size_t]+[C.c_float]*3+[C.c_uint64], C.c_void_p),
            'free': ([C.c_void_p], None),
            'inputs': ([C.c_void_p]*2, C.c_int),
            'advance': ([C.c_void_p]*3, C.c_int),
            'stats': ([C.c_void_p]*2, None),
            'batch': ([C.c_void_p,C.c_size_t]+[C.c_void_p]*6, C.c_int),
            'network_update': ([C.c_void_p], C.c_int),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, 'mcts_'+name)
            fn.argtypes, fn.restype = args, result
        self.b = options['b']
        capacity = options.get('node_capacity')
        if capacity is None:
            capacity = default_node_capacity(options['g'], options['s'],
                                             options.get('node_capacity_factor', 8))
        self.handle = self.lib.mcts_new(options['g'], options['b'], options['t'],
            options['s'], capacity, options['c_puct'],
            options['dirichlet_alpha'], options['dirichlet_epsilon'],
            options['seed'] if seed is None else seed)
        if not self.handle:
            raise RuntimeError('Could not create Rust MCTS arena')
        self.boards = torch.empty((2*self.b,5,10,16), dtype=torch.float32, pin_memory=pinned)

    def inputs(self):
        valid = check(self.lib.mcts_inputs(self.handle, self.boards.data_ptr()))
        return self.boards, valid

    def advance(self, logits, q):
        for tensor in (logits, q):
            if tensor.device.type != 'cpu' or tensor.dtype != torch.float32 or not tensor.is_contiguous() or tensor.shape != (2*self.b,80):
                raise ValueError('MCTS model outputs must be contiguous CPU float32 [2*b,80]')
        return check(self.lib.mcts_advance(self.handle, logits.data_ptr(), q.data_ptr()))

    def stats(self):
        values = (C.c_size_t*6)()
        self.lib.mcts_stats(self.handle, values)
        return Stats(*values)

    def batch(self, ids, transforms, batch):
        if ids.dtype != torch.uint32 or transforms.dtype != torch.uint8 or ids.shape != transforms.shape:
            raise ValueError('ids and transforms must be CPU uint32 and uint8 vectors')
        if not ids.is_contiguous() or not transforms.is_contiguous() or ids.device.type != 'cpu' or transforms.device.type != 'cpu':
            raise ValueError('ids and transforms must be contiguous CPU vectors')
        count = ids.numel()
        if count > batch.positions:
            raise ValueError('batch too small')
        return check(self.lib.mcts_batch(self.handle, count, ids.data_ptr(), transforms.data_ptr(),
            *(tensor.data_ptr() for tensor in batch.tensors)))

    def network_update(self):
        check(self.lib.mcts_network_update(self.handle))

    def close(self):
        if self.handle:
            self.lib.mcts_free(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Batch:
    def __init__(self, positions, *, pinned=False):
        self.positions = positions
        def empty(shape):
            return torch.empty(shape, dtype=torch.float32, pin_memory=pinned)
        self.tensors = (
            empty((2*positions,5,10,16)),
            empty((2*positions,160)),
            empty((2*positions,160)),
            empty((2*positions,160)),
        )
