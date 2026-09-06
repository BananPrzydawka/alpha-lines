"""Bitwise parity, timing, and PTX/SASS comparison for the fused SE kernel.

Modal (via the existing harness):
    modal run modal_harness.py --script verify_fused
    modal volume get alphalines /fused ./fused_artifacts

Local (if you ever have a Hopper card attached):
    uv run verify_fused.py
"""
import os
import re
import copy
import shutil
import difflib
import subprocess
from pathlib import Path
from collections import Counter

# ---- caches on the volume so the extension compiles once -------------------
ON_MODAL = Path("/outputs").is_dir()
ART = Path("/outputs/fused") if ON_MODAL else Path("./fused_artifacts")
ART.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(ART / "torch_ext"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ART / "inductor_cache"))
os.environ.setdefault("TRITON_CACHE_DIR", str(ART / "triton_cache"))

import torch
import torch._inductor.config as ind

ind.triton.store_cubin = True
ind.triton.unique_kernel_names = True

from config import device, height, width, num_parallel_games
from model import alpha_lines_net
from se_fused import fuse_model

SRC = Path(__file__).parent
HEADS = ("policy", "value", "error", "points")
ENTRY = re.compile(r"\.visible\s+\.entry\s+(\S+?)\s*\(")


# ---- ptx helpers -----------------------------------------------------------

def _norm(path):
    keep = []
    for line in Path(path).read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith(("//", "$L__", ".loc", ".file", ".section")):
            continue
        s = re.sub(r"%(rd|rs|fd|f|r|p)\d+", r"%\1", s)
        keep.append(re.sub(r"\s+", " ", s))
    return keep


def _opcodes(lines):
    return Counter(l.split()[0] for l in lines if l and not l.startswith("."))


def find_triton_ptx():
    roots = [Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]), Path(os.environ["TRITON_CACHE_DIR"])]
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.ptx")):
            txt = p.read_text(errors="ignore")
            m = ENTRY.search(txt)
            if m and "sigmoid" in m.group(1) and "silu" in m.group(1):
                return p
    return None


def build_our_ptx():
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return None
    out = ART / "mine.ptx"
    subprocess.run([nvcc, "-arch=sm_90a", "-O3", "-ptx",
                    str(SRC / "se_fused_kernel.cu"), "-o", str(out)], check=True)
    return out


def build_sass(ptx, tag):
    """assemble with triton's own ptxas so the SASS comparison is apples to apples"""
    try:
        import triton
        ptxas = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "ptxas"
    except Exception:
        ptxas = None
    if not (ptxas and ptxas.exists()):
        ptxas = shutil.which("ptxas")
    if not ptxas:
        return None
    cubin = ART / f"{tag}.cubin"
    subprocess.run([str(ptxas), "-arch=sm_90a", "-O3", str(ptx), "-o", str(cubin)], check=True)
    sass = subprocess.run(["cuobjdump", "-sass", str(cubin)], capture_output=True, text=True)
    (ART / f"{tag}.sass").write_text(sass.stdout)
    res = subprocess.run(["cuobjdump", "-res-usage", str(cubin)], capture_output=True, text=True)
    return res.stdout.strip()


def compare_ptx():
    ours, theirs = build_our_ptx(), find_triton_ptx()
    if not (ours and theirs):
        print(f"\n[ptx] skipped (ours={bool(ours)} triton={bool(theirs)})")
        return
    shutil.copy(theirs, ART / "triton.ptx")
    a, b = _norm(ours), _norm(theirs)
    oa, ob = _opcodes(a), _opcodes(b)

    print("\n[ptx] opcode histogram (mine / triton), differences only")
    for op in sorted(set(oa) | set(ob)):
        if oa[op] != ob[op]:
            print(f"  {op:28s} {oa[op]:4d} / {ob[op]:4d}")
    if oa == ob:
        print("  none - opcode mix is identical")

    diff = list(difflib.unified_diff(b, a, "triton", "mine", n=1, lineterm=""))
    (ART / "ptx.diff").write_text("\n".join(diff))
    print(f"[ptx] {len(diff)} diff lines -> {ART / 'ptx.diff'}")

    for tag, p in (("mine", ours), ("triton", ART / "triton.ptx")):
        usage = build_sass(p, tag)
        if usage:
            reg = [l.strip() for l in usage.splitlines() if "reg" in l.lower()]
            print(f"[sass] {tag}: {' | '.join(reg) if reg else 'no res-usage line'}")


# ---- parity ----------------------------------------------------------------

def run(model, x):
    with torch.no_grad():
        for _ in range(3):
            model(x)
        torch.cuda.synchronize()
        out = [t.clone() for t in model(x)]
        torch.cuda.synchronize()
    return out


def main():
    torch.manual_seed(0)
    base = (alpha_lines_net().to(device).eval().to(torch.bfloat16)
            .to(memory_format=torch.channels_last))
    fused = fuse_model(copy.deepcopy(base))

    x = torch.randn(num_parallel_games, 7, height, width,
                    device=device, dtype=torch.bfloat16).to(memory_format=torch.channels_last)

    cb, cf = torch.compile(base), torch.compile(fused)
    a, b = run(cb, x), run(cf, x)

    ok = True
    for name, u, v in zip(HEADS, a, b):
        same = torch.equal(u, v)
        ok &= same
        d = (u.float() - v.float()).abs().max().item()
        print(f"{name:7s} bitwise={str(same):5s}  max|d|={d:.3e}  "
              f"differing={(u != v).sum().item()}/{u.numel()}")
    pa, pb = a[0].flatten(1).argmax(1), b[0].flatten(1).argmax(1)
    print(f"policy argmax agreement: {(pa == pb).float().mean().item():.6f}")
    print("PARITY" if ok else "MISMATCH")

    from triton.testing import do_bench
    with torch.no_grad():
        t_base, t_fused = do_bench(lambda: cb(x)), do_bench(lambda: cf(x))
    print(f"\nbaseline {t_base * 1e3:8.1f} us\nfused    {t_fused * 1e3:8.1f} us")

    hidden = [p for p in Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).rglob("output_code.py")
              if "auto_functionalized" in p.read_text(errors="ignore")]
    if hidden:
        print(f"\n[warn] auto_functionalized in {len(hidden)} graph(s) - "
              f"check for a clone of the bn2 buffer before alpha.se_fused")

    compare_ptx()

    if ON_MODAL:
        try:
            import modal
            modal.Volume.from_name("alphalines").commit()
        except Exception as e:
            print(f"[warn] volume commit skipped: {e}")


if __name__ == "__main__":
    main()
