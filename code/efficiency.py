"""
efficiency.py -- compute/efficiency + coverage comparison across context length.

Schemes (8-layer, d=512, batch 1):
  dense          full attention (FlashSDPA)                     ceiling, O(T^2)
  logsparse      pow2 fixed gather                              fixed geometric
  fib_fixed      Fibonacci fixed gather (alpha=1 all layers)    fixed Fibonacci
  fib_staggered  Fibonacci gather, per-layer alpha 0.5..1.5     staggered (diverse) Fibonacci

Reports per (T, scheme):
  attnGFLOP  analytic attention FLOPs (sum over layers)         <- the cost
  unionCov   distinct distances reachable across all layers     <- the coverage
  Katt/L     per-layer attended count (sparsity)                <- per-layer cost
  fwd_ms     measured forward wall-clock (idle GPU)
  peakGB     measured peak VRAM

KEY POINT (what Chad asked to quantify): alpha-scaling is sparsity-NEUTRAL (Katt/L
is ~flat across alpha), so fib_fixed and fib_staggered have the SAME per-layer FLOPs
-- but staggering the per-layer alpha multiplies the UNION coverage (~4x) for free.
The learned-from-neutral model collapses to the fib_fixed row (alpha stays ~1).

CAVEATS (unchanged): bool-mask sparse saves no FLOPs (this uses the gather path);
the gather is not a fused kernel, so wall-clock trails fused dense even where analytic
FLOPs are far lower -- the clean win is the FLOPs/coverage curves.

  python code/efficiency.py        (needs an idle GPU for valid timing)
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch

for p in (str(Path(__file__).parent),):
    if p not in sys.path:
        sys.path.insert(0, p)

from baseline import BaselineConfig, BaselineTransformer        # noqa: E402
from fib_attention import FibonacciAttention, attended_distances  # noqa: E402

D, NH, NKV, DFFN, L = 512, 8, 2, 1366, 8
LENGTHS = [1024, 2048, 4096, 8192, 16384]
KMAX = 25
SCHEMES = ["dense", "logsparse", "fib_fixed", "fib_staggered"]


def staggered_alphas(n):
    return [0.5 + 1.0 * i / (n - 1) for i in range(n)]


def per_layer_alphas(scheme):
    if scheme == "fib_staggered":
        return staggered_alphas(L)
    return [1.0] * L   # fib_fixed / logsparse: alpha=1 every layer (dense: unused)


def _offbase(scheme):
    return "pow2" if scheme == "logsparse" else "fib"


def per_layer_katt(scheme, T):
    if scheme == "dense":
        return [T] * L
    ob = _offbase(scheme)
    return [len(attended_distances(ob, 8, KMAX, T, a)) for a in per_layer_alphas(scheme)]


def union_coverage(scheme, T):
    if scheme == "dense":
        return T
    ob = _offbase(scheme)
    u = set()
    for a in per_layer_alphas(scheme):
        u |= set(attended_distances(ob, 8, KMAX, T, a))
    return len(u)


def analytic_attn_gflops(scheme, T):
    hd = D // NH
    katt = per_layer_katt(scheme, T)
    # per layer: QK + AV ~ 2 * 2(MAC) * NH * T * (keys) * hd ;  dense keys=T, sparse keys=Katt
    flops = sum(2 * 2 * NH * T * ka * hd for ka in katt)
    return flops / 1e9


def build(scheme, T):
    cfg = BaselineConfig(d_model=D, n_layers=L, n_heads=NH, n_kv_heads=NKV,
                         vocab_size=512, seq_len=T, dropout=0.0)
    if scheme == "dense":
        pass
    else:
        ob = _offbase(scheme)
        cfg.attn_factory = lambda c, i: FibonacciAttention(
            c, i, offset_base=ob, alpha_mode="fixed", impl="stream",
            use_checkpoint=False, W=8, K=KMAX)
    model = BaselineTransformer(cfg)
    if scheme == "fib_staggered":
        alphas = staggered_alphas(L)
        fibmods = [m for m in model.modules() if isinstance(m, FibonacciAttention)]
        for m, a in zip(fibmods, alphas):
            ac = min(1.45, max(0.55, a))                     # keep off the sigmoid asymptotes
            theta = math.log((ac - 0.5) / (1.5 - ac))        # alpha = 0.5 + sigmoid(theta)
            m.theta.fill_(theta)
    return model


@torch.no_grad()
def measure(scheme, T, device, warmup=2, timed=5):
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        m = build(scheme, T).to(device).eval()
        idx = torch.randint(0, 512, (1, T), device=device)
        ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        for _ in range(warmup):
            with ctx: m(idx)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(timed):
            with ctx: m(idx)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / timed * 1000
        pk = torch.cuda.max_memory_allocated() / 1e9
        del m; torch.cuda.empty_cache()
        return ms, pk
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"d={D} L={L} heads={NH}/{NKV} batch=1 | dense=FlashSDPA, sparse=streaming gather\n")
    hdr = f"{'T':>6} {'scheme':>13} {'Katt/L':>7} {'attnGFLOP':>10} {'unionCov':>9} {'fwd_ms':>8} {'peakGB':>7}"
    print(hdr)
    for T in LENGTHS:
        for sc in SCHEMES:
            katt = per_layer_katt(sc, T)
            kstr = "T" if sc == "dense" else f"{min(katt)}-{max(katt)}" if min(katt) != max(katt) else f"{katt[0]}"
            gf = analytic_attn_gflops(sc, T)
            uc = union_coverage(sc, T)
            ms, pk = measure(sc, T, device)
            msr = f"{ms:8.1f}" if ms is not None else "    OOM "
            pkr = f"{pk:7.1f}" if pk is not None else "    -- "
            print(f"{T:>6} {sc:>13} {kstr:>7} {gf:10.1f} {uc:>9} {msr} {pkr}")
        print()
    print("Read: attnGFLOP = cost (dense ~T^2, sparse ~T*Katt); unionCov = distinct distances reachable.")
    print("fib_fixed vs fib_staggered: SAME Katt/L and ~same attnGFLOP, but staggered's unionCov is ~4x.")
    print("(A learned-from-neutral model lands on the fib_fixed row: alpha stays ~1.)")


if __name__ == "__main__":
    main()
