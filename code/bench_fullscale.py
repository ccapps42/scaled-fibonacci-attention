"""
bench_fullscale.py -- memory + speed feasibility check for FibonacciAttention at
the real screen scale (d=512, L=16, seq=1024). Measures peak CUDA memory and
forward+backward microbatch time vs the dense baseline. See spec section 12.4.

Run on an IDLE GPU.  python code/bench_fullscale.py
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from baseline import BaselineTransformer, make_config           # noqa: E402
from fib_attention import make_factories                        # noqa: E402

DEV = torch.device("cuda")
SEQ = 1024
GRAD_ACCUM = 4
TOTAL_STEPS = 13000
WARMUP, TIMED = 3, 10


def build(attn_kw):
    cfg = make_config(512, seq_len=SEQ)
    if attn_kw is not None:
        f = make_factories(**attn_kw)
        cfg.attn_factory = f["attn_factory"]
    return BaselineTransformer(cfg).to(DEV)


def bench(label, attn_kw, batch):
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build(attn_kw)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        idx = torch.randint(0, 32768, (batch, SEQ), device=DEV)
        tgt = torch.randint(0, 32768, (batch, SEQ), device=DEV)

        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(idx, tgt)
            loss.backward()
            opt.step()

        for _ in range(WARMUP):
            step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(TIMED):
            step()
        torch.cuda.synchronize()
        fb = (time.perf_counter() - t0) / TIMED
        peak = torch.cuda.max_memory_allocated() / 1e9

        step_s = fb * GRAD_ACCUM
        hrs = step_s * TOTAL_STEPS / 3600
        print(f"  {label:28s} B={batch}: peak {peak:5.1f} GB | "
              f"micro-fb {fb*1000:6.1f} ms | step {step_s*1000:6.0f} ms | "
              f"~{hrs:4.1f} h/run")
        del model, opt
        torch.cuda.empty_cache()
        return peak, fb
    except torch.cuda.OutOfMemoryError:
        print(f"  {label:28s} B={batch}: OOM")
        torch.cuda.empty_cache()
        return None, None


print("=" * 78)
print("Full-scale feasibility: d=512 L=16 seq=1024, fwd+bwd+opt, bf16 autocast")
print("=" * 78)
print("DENSE baseline (reference):")
bench("dense GQA", None, 8)
print("FIBONACCI (W=8, learned, K=15):")
for b in (8, 4, 2):
    p, _ = bench("fib W=8", dict(offset_base="fib", alpha_mode="learned", W=8, K=15), b)
    if p is not None:
        break
print("FIBONACCI (W=12 worst-case C, learned):")
for b in (8, 4, 2):
    p, _ = bench("fib W=12", dict(offset_base="fib", alpha_mode="learned", W=12, K=15), b)
    if p is not None:
        break
print("=" * 78)
