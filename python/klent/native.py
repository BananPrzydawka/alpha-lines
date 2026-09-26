"""Owned ctypes handles and reusable CPU tensors for the Rust KLENT core."""
import ctypes as C
from typing import NamedTuple
import torch


class ArenaStats(NamedTuple):
    positions: int
    wins: int
    draws: int
    losses: int


def check(status):
    if status < 0:
        raise RuntimeError("Rust KLENT operation failed; see the preceding Rust diagnostic")
    return status


class Arena:
    def __init__(self, library, options, *, evaluation=False, seed=None, pinned=False):
        self.lib = C.CDLL(str(library))
        signatures = {
            'new': ([C.c_size_t]*2+[C.c_float]*4+[C.c_uint64,C.c_bool], C.c_void_p),
            'free': ([C.c_void_p], None),
            'inputs': ([C.c_void_p]*2, C.c_int),
            'step': ([C.c_void_p]*5, C.c_int),
            'stats': ([C.c_void_p]*2, None),
            'reset': ([C.c_void_p], C.c_size_t),
            'shuffle': ([C.c_void_p], None),
            'clear': ([C.c_void_p], None),
            'batch': ([C.c_void_p,C.c_size_t,C.c_size_t]+[C.c_void_p]*4, C.c_int),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, 'klent_'+name)
            fn.argtypes, fn.restype = args, result
        self.n = options['test_games'] if evaluation else options['n']
        self.handle = self.lib.klent_new(self.n, options['m'],
            options['alpha'], options['beta'], options['lambda'], options['exploration_fraction'],
            options['seed'] if seed is None else seed, evaluation)
        if not self.handle:
            raise RuntimeError('Could not create Rust KLENT arena')
        self.boards = torch.empty((2*self.n,5,10,16), dtype=torch.float32, pin_memory=pinned)

    def inputs(self):
        check(self.lib.klent_inputs(self.handle,self.boards.data_ptr()))
        return self.boards

    def step(self, logits, q, opponent_logits=None, opponent_q=None):
        if opponent_logits is None:
            opponent_logits = logits
        if opponent_q is None:
            opponent_q = q
        for value in (logits,q,opponent_logits,opponent_q):
            if value.device.type != 'cpu' or value.dtype != torch.float32 or not value.is_contiguous() or value.shape != (2*self.n,80):
                raise ValueError('native outputs must be contiguous CPU float32 [2*n,80]')
        return bool(check(self.lib.klent_step(self.handle,logits.data_ptr(),q.data_ptr(),
                                              opponent_logits.data_ptr(),opponent_q.data_ptr())))

    def stats(self):
        values = (C.c_size_t*4)()
        self.lib.klent_stats(self.handle,values)
        return ArenaStats(*values)

    def reset(self):
        return self.lib.klent_reset(self.handle)

    def shuffle(self):
        self.lib.klent_shuffle(self.handle)

    def clear(self):
        self.lib.klent_clear(self.handle)

    def batch(self, start, batch):
        return check(self.lib.klent_batch(self.handle,start,batch.rows//2,
            *(t.data_ptr() for t in batch.tensors)))

    def close(self):
        if self.handle:
            self.lib.klent_free(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()


class Batch:
    def __init__(self, rows, pinned=False):
        self.rows = rows
        def empty(shape,dtype):
            return torch.empty(shape,dtype=dtype,pin_memory=pinned)
        # Keep this order aligned with Arena.batch()'s native pointer arguments.
        self.tensors = (
            empty((rows,5,10,16),torch.float32),
            empty((rows,160),torch.float32),
            empty((rows,),torch.int64),
            empty((rows,),torch.float32),
        )
