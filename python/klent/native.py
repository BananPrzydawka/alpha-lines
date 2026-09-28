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
    def __init__(self, library, options, *, evaluation=False, fixed_opponent=False, seed=None, pinned=False):
        self.lib = C.CDLL(str(library))
        signatures = {
            'new': ([C.c_size_t]*2+[C.c_float]*4+[C.c_uint64,C.c_bool], C.c_void_p),
            'free': ([C.c_void_p], None),
            'inputs': ([C.c_void_p]*2, C.c_int),
            'step': ([C.c_void_p]*3, C.c_int),
            'step_with_targets': ([C.c_void_p]*5, C.c_int),
            'set_fixed_opponent': ([C.c_void_p,C.c_bool], None),
            'stats': ([C.c_void_p]*2, None),
            'reset': ([C.c_void_p], C.c_size_t),
            'shuffle': ([C.c_void_p], None),
            'clear': ([C.c_void_p], None),
            'batch': ([C.c_void_p,C.c_size_t,C.c_size_t]+[C.c_void_p]*5, C.c_int),
            'batch_one_side': ([C.c_void_p,C.c_size_t,C.c_size_t]+[C.c_void_p]*4, C.c_int),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.lib, 'klent_'+name)
            fn.argtypes, fn.restype = args, result
        self.n = options['test_games'] if evaluation else options['n']
        if fixed_opponent and (evaluation or self.n % 2):
            raise ValueError('fixed-opponent training requires an even number of games')
        self.handle = self.lib.klent_new(self.n, options['m'],
            options['alpha'], options['beta'], options['lambda'], options['exploration_fraction'],
            options['seed'] if seed is None else seed, evaluation)
        if not self.handle:
            raise RuntimeError('Could not create Rust KLENT arena')
        if fixed_opponent:
            self.lib.klent_set_fixed_opponent(self.handle, True)
        self.boards = torch.empty((2*self.n,5,10,16), dtype=torch.float32, pin_memory=pinned)

    def inputs(self):
        check(self.lib.klent_inputs(self.handle,self.boards.data_ptr()))
        return self.boards

    def step(self, logits, q, *, target_logits=None, target_q=None):
        if (target_logits is None) != (target_q is None):
            raise ValueError('target logits and values must be supplied together')
        for value in (logits,q) + (() if target_logits is None else (target_logits,target_q)):
            if value.device.type != 'cpu' or value.dtype != torch.float32 or not value.is_contiguous() or value.shape != (2*self.n,80):
                raise ValueError('native outputs must be contiguous CPU float32 [2*n,80]')
        if target_logits is None:
            status = self.lib.klent_step(self.handle,logits.data_ptr(),q.data_ptr())
        else:
            status = self.lib.klent_step_with_targets(self.handle,logits.data_ptr(),q.data_ptr(),
                target_logits.data_ptr(),target_q.data_ptr())
        return bool(check(status))

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
        if batch.perspectives == 1:
            return check(self.lib.klent_batch_one_side(self.handle,start,batch.positions,
                *(t.data_ptr() for t in batch.tensors)))
        return check(self.lib.klent_batch(self.handle,start,batch.positions,
            *(t.data_ptr() for t in batch.tensors),batch.players.data_ptr()))

    def close(self):
        if self.handle:
            self.lib.klent_free(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()


class Batch:
    def __init__(self, rows, pinned=False, perspectives=2):
        if perspectives not in (1, 2) or rows % perspectives:
            raise ValueError('rows must be divisible by one or two perspectives')
        self.rows = rows
        self.perspectives = perspectives
        self.positions = rows//perspectives
        def empty(shape,dtype):
            return torch.empty(shape,dtype=dtype,pin_memory=pinned)
        # Keep this order aligned with Arena.batch()'s native pointer arguments.
        self.tensors = (
            empty((rows,5,10,16),torch.float32),
            empty((rows,160),torch.float32),
            empty((rows,),torch.int64),
            empty((rows,),torch.float32),
        )
        self.players = empty((self.positions,),torch.int64)
