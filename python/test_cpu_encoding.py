"""CPU encoder parity: run with `uv run python python/test_cpu_encoding.py`."""
import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch
from export_mcts import MCTSModel


class CpuEncodingTest(unittest.TestCase):
    def test_matches_reference(self):
        root = Path(__file__).resolve().parents[1]
        include = Path(torch.__file__).parent / 'include'
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'encode.cpp'
            library = Path(directory) / 'encode.so'
            source.write_text('''#include <c10/util/BFloat16.h>
#include "encode.h"
extern "C" void encode(const uint8_t* cells, const int32_t* scores,
                       size_t batch, c10::BFloat16* out) {
    encode_planes(cells, scores, batch, out);
}
''')
            subprocess.run(['c++', '-std=c++17', '-O3', '-shared', '-fPIC',
                            f'-I{include}', f'-I{root / "rust/native"}',
                            str(source), '-o', str(library)], check=True)
            native = ctypes.CDLL(str(library)).encode
            native.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
            native.restype = None
            torch.manual_seed(7)
            cells = torch.randint(0, 4, (37, 80), dtype=torch.uint8)
            cells[:4] = torch.arange(4, dtype=torch.uint8).view(4, 1)
            scores = torch.randint(-100000, 100000, (37, 2), dtype=torch.int32)
            scores[0] = 0
            model = MCTSModel()
            reference = model.encode(cells, scores)
            actual = torch.empty_like(reference)
            # Repeated writes must replace prior contents, including opening masks.
            for boards in (cells, torch.zeros_like(cells), cells):
                native(boards.data_ptr(), scores.data_ptr(), len(boards), actual.data_ptr())
                expected = model.encode(boards, scores)
                self.assertEqual(actual.stride(), expected.stride())
                self.assertTrue(torch.equal(actual, expected))


if __name__ == '__main__':
    unittest.main()
