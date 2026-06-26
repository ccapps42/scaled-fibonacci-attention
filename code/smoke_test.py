"""
smoke_test.py -- correctness + differentiability for all three FibonacciAttention paths.

A. materialized, W=T-1, alpha=1  == full causal GQA
B. materialized, W=2,  alpha=1   == masked SDPA over the same distance set
C. materialized gradient to theta: nonzero at exact alpha=1 + finite-diff off-lattice
D. streaming  == materialized (forward) at alpha=1 and off-lattice
E. bool-mask (option B) == materialized at fixed alpha=1
F. streaming gradient to theta nonzero (through checkpoint, training mode)

Run: python code/smoke_test.py   (CPU, float32)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from baseline import BaselineConfig, GQAAttention, _build_rope_cache       # noqa: E402
from fib_attention import (FibonacciAttention, FibBoolMaskAttention,       # noqa: E402
                           fib_base)

torch.manual_seed(0)
DT = torch.float32
cfg = BaselineConfig(d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                     vocab_size=256, seq_len=64, dropout=0.0)
B, T, hd = 2, cfg.seq_len, cfg.d_model // cfg.n_heads
cos, sin = _build_rope_cache(T, hd, cfg.rope_theta, torch.device("cpu"))
cos, sin = cos.to(DT), sin.to(DT)
x = torch.randn(B, T, cfg.d_model, dtype=DT)
results = {}


def copy_proj(dst, src):
    for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
        getattr(dst, n).weight.data.copy_(getattr(src, n).weight.data)


def boolean_ref_mask(distances):
    i = torch.arange(T).unsqueeze(1); j = torch.arange(T).unsqueeze(0); diff = i - j
    allowed = torch.zeros(T, T, dtype=torch.bool)
    for d in distances:
        allowed |= (diff == d)
    return allowed & (diff >= 0)


def mk(**kw):
    return FibonacciAttention(cfg, 0, **kw).to(DT)


# anchor module whose weights everything else copies
anchor = mk(offset_base="fib", alpha_mode="fixed", W=2, K=15, impl="materialized").eval()

print("=" * 70)
mat_a = mk(offset_base="fib", alpha_mode="fixed", W=T - 1, K=15, impl="materialized").eval()
copy_proj(mat_a, anchor)
gqa = GQAAttention(cfg).to(DT).eval(); copy_proj(gqa, anchor)
with torch.no_grad():
    da = (mat_a(x, cos, sin) - gqa(x, cos, sin, mask=None)).abs().max().item()
results["A materialized==causal"] = da < 1e-4
print(f"A  materialized (W=T-1,a=1) == causal GQA      max {da:.2e}  {'PASS' if da<1e-4 else 'FAIL'}")

dists = set(range(0, 3)) | {f for f in fib_base(15) if 2 < f <= T - 1}
with torch.no_grad():
    out_mat = anchor(x, cos, sin)
    ref = GQAAttention(cfg).to(DT).eval(); copy_proj(ref, anchor)
    out_ref = ref(x, cos, sin, mask=boolean_ref_mask(dists))
    db = (out_mat - out_ref).abs().max().item()
results["B materialized==masked SDPA"] = db < 1e-4
print(f"B  materialized (W=2,a=1)   == masked SDPA      max {db:.2e}  {'PASS' if db<1e-4 else 'FAIL'}")

# C: materialized gradient
tgt = torch.randn(B, T, cfg.d_model, dtype=DT)
def loss_mat(theta, W=8):
    m = mk(offset_base="fib", alpha_mode="learned", W=W, K=15, impl="materialized")
    copy_proj(m, anchor)
    with torch.no_grad(): m.theta.fill_(theta)
    return ((m(x, cos, sin) - tgt) ** 2).mean(), m
l0, m0 = loss_mat(0.0); l0.backward(); g0 = m0.theta.grad.item()
lA, mA = loss_mat(0.30); lA.backward(); ga = mA.theta.grad.item()
eps = 1e-3
with torch.no_grad():
    lp, _ = loss_mat(0.30 + eps); lm, _ = loss_mat(0.30 - eps)
fd = ((lp - lm) / (2 * eps)).item(); rel = abs(ga - fd) / (abs(fd) + 1e-12)
results["C grad nonzero@a=1 + FD"] = abs(g0) > 1e-9 and rel < 1e-2
print(f"C  grad @a=1 nonzero={g0:.2e}; autograd={ga:.3e} FD={fd:.3e} rel={rel:.1e}"
      f"  {'PASS' if abs(g0)>1e-9 and rel<1e-2 else 'FAIL'}")

# D: streaming == materialized
worst = 0.0
for th in (0.0, 0.30):
    mat = mk(offset_base="fib", alpha_mode="learned", W=8, impl="materialized").eval()
    strm = mk(offset_base="fib", alpha_mode="learned", W=8, impl="stream", use_checkpoint=False).eval()
    copy_proj(mat, anchor); copy_proj(strm, anchor)
    with torch.no_grad():
        mat.theta.fill_(th); strm.theta.fill_(th)
        worst = max(worst, (mat(x, cos, sin) - strm(x, cos, sin)).abs().max().item())
results["D streaming==materialized"] = worst < 1e-5
print(f"D  streaming == materialized (a=1 & off)       max {worst:.2e}  {'PASS' if worst<1e-5 else 'FAIL'}")

# E: bool-mask == materialized at fixed alpha=1
bm = FibBoolMaskAttention(cfg, 0, offset_base="fib", W=2, K=15).to(DT).eval()
copy_proj(bm, anchor)
with torch.no_grad():
    de = (bm(x, cos, sin) - anchor(x, cos, sin)).abs().max().item()
results["E boolmask==materialized"] = de < 1e-5
print(f"E  bool-mask (option B) == materialized        max {de:.2e}  {'PASS' if de<1e-5 else 'FAIL'}")

# F: streaming gradient through checkpoint (training mode)
sg = mk(offset_base="fib", alpha_mode="learned", W=8, impl="stream", use_checkpoint=True).train()
copy_proj(sg, anchor)
out = sg(x, cos, sin)
((out - tgt) ** 2).mean().backward()
gf = sg.theta.grad
ok_f = gf is not None and torch.isfinite(gf).all() and gf.abs().item() > 1e-9
results["F streaming grad (ckpt)"] = bool(ok_f)
print(f"F  streaming grad through checkpoint = {gf.item():.2e}    {'PASS' if ok_f else 'FAIL'}")

print("=" * 70)
allok = all(results.values())
print("OVERALL:", "PASS" if allok else "FAIL  -> " + ", ".join(k for k, v in results.items() if not v))
sys.exit(0 if allok else 1)
