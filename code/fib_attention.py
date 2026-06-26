"""
fib_attention.py -- Scaled-Fibonacci sparse attention.

Three code paths (all validated to agree by smoke_test.py):
  - FibonacciAttention(impl="materialized")  reference; stacks (B,H,T,C,hd). Heavy.
  - FibonacciAttention(impl="stream")        option C: online-softmax over the C
        sparse columns + gradient checkpointing -> memory O(1) in C. Used for the
        LEARNED-alpha runs (needs gradient to the spring).
  - FibBoolMaskAttention                     option B: fixed alpha=1 -> integer
        offsets -> a boolean [T,T] mask through SDPA at dense speed/memory. Used
        for the FIXED-alpha and pow2 CONTROL runs (no gradient to alpha needed).

make_factories() dispatches: alpha_mode="fixed" -> bool-mask (B); "learned" -> stream (C).

See ../fibonacci-attention-spec.md. Imports _apply_rope from baseline.py in this directory.

Interpolation bracket: lo=floor(target), hi=lo+1, frac=target-lo, further index
CLAMPED to >=0. Keeps grad-to-theta alive even at integer alpha (floor/ceil would
zero it). Validity from the NEARER index (lo). Masked logits use -1e9, never -inf.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from baseline import _apply_rope  # noqa: E402

if TYPE_CHECKING:
    from baseline import BaselineConfig

NEG = -1e9


def fib_base(k: int) -> list[int]:
    fibs = [1, 2]
    while len(fibs) < k:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs[:k]


def pow2_base_reach_matched(seq_len: int) -> list[int]:
    offs, d = [], 1
    while d <= seq_len - 1:
        offs.append(d)
        d *= 2
    return offs


def attended_distances(offset_base: str, W: int, K: int, seq_len: int, alpha: float = 1.0) -> list[int]:
    """Integer attended distances: window {0..W} U {round(alpha*base rungs) > W}."""
    base = fib_base(K) if offset_base == "fib" else pow2_base_reach_matched(seq_len)
    dist = set(range(0, W + 1))
    for r in base:
        rr = round(alpha * r)
        if rr > W and rr <= seq_len - 1:
            dist.add(rr)
    return sorted(dist)


def _gather(x: torch.Tensor, d: int) -> torch.Tensor:
    """x:(B,H,T,hd) -> x[...,clamp(t-d,0),:] (clamped gather)."""
    T = x.shape[2]
    idx = (torch.arange(T, device=x.device) - d).clamp_(min=0)
    return x.index_select(2, idx)


class _ProjMixin:
    """Shared q/k/v/o projections + rope + GQA expand."""

    def _init_proj(self, cfg):
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        d = cfg.d_model
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.o_proj.is_residual_proj = True
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def _qkv(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        return q, k, v


class FibonacciAttention(nn.Module, _ProjMixin):
    """Gather-based Fibonacci attention with a learned (or fixed) per-layer spring alpha."""

    def __init__(self, cfg: "BaselineConfig", layer_idx: int,
                 offset_base="fib", alpha_mode="learned", alpha_scope="per_layer",
                 W=8, K=15, alpha_min=0.5, alpha_span=1.0,
                 impl="stream", use_checkpoint=True):
        super().__init__()
        self._init_proj(cfg)
        self.layer_idx = layer_idx
        self.W, self.alpha_min, self.alpha_span = W, alpha_min, alpha_span
        self.impl, self.use_checkpoint = impl, use_checkpoint

        base = fib_base(K) if offset_base == "fib" else pow2_base_reach_matched(cfg.seq_len)
        self.register_buffer("base", torch.tensor(base, dtype=torch.float32), persistent=False)

        if alpha_scope != "per_layer":
            raise NotImplementedError("per_head alpha (f10) not yet implemented")
        theta0 = torch.zeros(1)
        if alpha_mode == "learned":
            self.theta = nn.Parameter(theta0)
        elif alpha_mode == "fixed":
            self.register_buffer("theta", theta0, persistent=False)
        else:
            raise ValueError(alpha_mode)

    def alpha(self):
        return self.alpha_min + self.alpha_span * torch.sigmoid(self.theta)

    def _column_specs(self, targets, T):
        """List of (lo, hi, frac_or_None, valid_from). Window cols: frac=None (exact)."""
        Wc = min(self.W, T - 1)
        specs = [(d, d, None, d) for d in range(0, Wc + 1)]
        used = set(range(0, Wc + 1))
        for kk in range(self.base.shape[0]):
            tgt = targets[kk]
            lo = int(torch.floor(tgt).item())
            rnd = int(torch.round(tgt).item())
            if rnd <= self.W or lo > T - 1 or lo in used:
                continue
            used.add(lo)
            specs.append((lo, lo + 1, tgt - lo, lo))
        return specs

    def _col_kv(self, k, v, lo, hi, frac):
        if frac is None:
            return _gather(k, lo), _gather(v, lo)
        kc = (1 - frac) * _gather(k, lo) + frac * _gather(k, hi)
        vc = (1 - frac) * _gather(v, lo) + frac * _gather(v, hi)
        return kc, vc

    def _core(self, q, k, v):
        B, H, T, hd = q.shape
        targets = self.alpha().to(q.device) * self.base.to(q.device)
        specs = self._column_specs(targets, T)
        scale = hd ** -0.5
        pos = torch.arange(T, device=q.device)

        if self.impl == "materialized":
            cols_k, cols_v, vfrom = [], [], []
            for lo, hi, frac, vf in specs:
                kc, vc = self._col_kv(k, v, lo, hi, frac)
                cols_k.append(kc); cols_v.append(vc); vfrom.append(vf)
            Kg = torch.stack(cols_k, 3); Vg = torch.stack(cols_v, 3)   # B,H,T,C,hd
            scores = (q.unsqueeze(3) * Kg).sum(-1) * scale             # B,H,T,C
            valid = pos.unsqueeze(1) >= torch.tensor(vfrom, device=q.device).unsqueeze(0)
            scores = scores.masked_fill(~valid.view(1, 1, T, -1), NEG)
            attn = torch.softmax(scores, -1)
            return (attn.unsqueeze(-1) * Vg).sum(3)

        # streaming online-softmax (fp32 stats), memory O(1) in C
        m = torch.full((B, H, T), NEG, device=q.device, dtype=torch.float32)
        l = torch.zeros((B, H, T), device=q.device, dtype=torch.float32)
        acc = torch.zeros((B, H, T, hd), device=q.device, dtype=torch.float32)
        for lo, hi, frac, vf in specs:
            kc, vc = self._col_kv(k, v, lo, hi, frac)
            s = (q * kc).sum(-1).float() * scale
            s = s.masked_fill((pos < vf).view(1, 1, T), NEG)
            m_new = torch.maximum(m, s)
            corr = torch.exp(m - m_new)
            p = torch.exp(s - m_new)
            l = l * corr + p
            acc = acc * corr.unsqueeze(-1) + p.unsqueeze(-1) * vc.float()
            m = m_new
        return (acc / l.clamp_min(1e-9).unsqueeze(-1)).to(q.dtype)

    def forward(self, x, cos, sin, mask=None):
        B, T, _ = x.shape
        q, k, v = self._qkv(x, cos, sin)
        if self.impl == "stream" and self.use_checkpoint and self.training:
            out = checkpoint(self._core, q, k, v, use_reentrant=False)
        else:
            out = self._core(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.drop(self.o_proj(out))


class FibBoolMaskAttention(nn.Module, _ProjMixin):
    """Option B: fixed alpha=1 integer offsets -> boolean [T,T] mask through SDPA.

    Numerically identical to FibonacciAttention at alpha=1 (smoke-verified), at
    dense speed and memory. Use for fixed-alpha (fib) and pow2 CONTROL runs.
    """

    def __init__(self, cfg, layer_idx, offset_base="fib", W=8, K=15, alpha=1.0):
        super().__init__()
        self._init_proj(cfg)
        self.layer_idx = layer_idx
        self.alpha = alpha
        self.distances = attended_distances(offset_base, W, K, cfg.seq_len, alpha)
        self._mask_cache: dict[int, torch.Tensor] = {}

    def _mask(self, T, device):
        key = T
        if key not in self._mask_cache or self._mask_cache[key].device != device:
            i = torch.arange(T, device=device).unsqueeze(1)
            j = torch.arange(T, device=device).unsqueeze(0)
            diff = i - j
            allowed = torch.zeros(T, T, dtype=torch.bool, device=device)
            for d in self.distances:
                allowed |= (diff == d)
            self._mask_cache[key] = allowed & (diff >= 0)
        return self._mask_cache[key]

    def forward(self, x, cos, sin, mask=None):
        B, T, _ = x.shape
        q, k, v = self._qkv(x, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=self._mask(T, x.device),
                                             dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.drop(self.o_proj(out))


def make_factories(config=None, offset_base="fib", alpha_mode="learned",
                   alpha_scope="per_layer", W=8, K=15,
                   impl="materialized", use_checkpoint=True, **kw) -> dict:
    """Dispatch: fixed alpha -> bool-mask fast path (B); learned -> gather (C/materialized).

    Default learned impl is "materialized" (fastest at B=4, fits ~12 GB). Pass
    impl="stream" for the streaming+checkpoint path (long-context / large-batch).
    """
    def attn_factory(cfg, layer_idx):
        if alpha_mode in ("staggered", "staggered_hdc"):
            # per-layer FIXED alpha tiled across [0.5,1.5] -> layers collectively widen
            # union coverage at the same per-layer sparsity. Bool-mask path (fast).
            L = cfg.n_layers
            slot = layer_idx
            if alpha_mode == "staggered_hdc" and L > 2:
                # HDC port: assign the SAME alpha set via a coprime stride so ADJACENT layers
                # are maximally spread (anti-gridding), instead of the monotonic ramp where
                # neighbouring layers nearly duplicate. Same union coverage; cascade order differs.
                S = next(s for s in range(L // 2, 1, -1) if math.gcd(s, L) == 1)
                slot = (layer_idx * S) % L
            a = 0.5 + 1.0 * (slot / (L - 1)) if L > 1 else 1.0
            return FibBoolMaskAttention(cfg, layer_idx, offset_base=offset_base, W=W, K=K, alpha=a)
        if alpha_mode == "fixed":
            return FibBoolMaskAttention(cfg, layer_idx, offset_base=offset_base, W=W, K=K)
        return FibonacciAttention(cfg, layer_idx, offset_base=offset_base,
                                  alpha_mode="learned", alpha_scope=alpha_scope,
                                  W=W, K=K, impl=impl, use_checkpoint=use_checkpoint, **kw)
    return {"attn_factory": attn_factory}
