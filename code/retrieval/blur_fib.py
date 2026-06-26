"""
blur_fib.py -- forced-gather annealed-blur Fibonacci attention (learned scalar alpha).

This is the ONE mechanism in which the learned per-layer spring alpha actually moves
onto a useful distance (validated: single-target d=350 -> alpha 1.0->0.92, acc=1.0,
while every non-blur mechanism stays at alpha~1 / acc~0). The trick:

  Each Fibonacci rung gathers a GAUSSIAN BAND of keys centered at d_k = alpha*fib_k,
  width w annealed wide->narrow over training. Because the KEYS THEMSELVES are blurred
  blends (not sharp), the model cannot get a clean copy via QK alone -- it must slide a
  rung center onto the target, which is what puts gradient on alpha. There is no dense
  escape hatch (only the blurred-rung keys exist), unlike an additive-bias blur.

Memory: the per-rung band tensor (B,H,T,M,hd), M~5w, is retained for the alpha gradient.
At wide w this is GBs PER LAYER. Gradient checkpointing on _core bounds peak to ONE
layer's bands at a time, so depth does not multiply peak. Still needs a reduced batch at
T=1024 (one layer at w0=40, B=2, T=1024 ~ 5.5 GB).

Reuses fib_attention primitives (_ProjMixin, _gather, fib_base, NEG) READ side only.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_HERE = Path(__file__).parent
_CODE = _HERE.parent
for p in (str(_HERE), str(_CODE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fib_attention import _ProjMixin, _gather, fib_base, NEG  # noqa: E402


class BlurFibAttention(nn.Module, _ProjMixin):
    """Learned per-layer spring alpha; rungs gather an annealed Gaussian band (forced)."""

    def __init__(self, cfg, layer_idx, offset_base="fib", W=8, K=15,
                 w0=40.0, alpha_min=0.5, alpha_span=1.0, use_checkpoint=True):
        super().__init__()
        self._init_proj(cfg)
        self.layer_idx = layer_idx
        self.W = W
        self.alpha_min, self.alpha_span = alpha_min, alpha_span
        self.use_checkpoint = use_checkpoint
        if offset_base != "fib":
            raise NotImplementedError("blur mechanism is Fibonacci-only")
        rungs = [r for r in fib_base(K) if r > W]
        self.register_buffer("fib", torch.tensor(rungs, dtype=torch.float32), persistent=False)
        self.theta = nn.Parameter(torch.zeros(1))
        self._w = float(w0)   # current blur width, set per-step by the training schedule

    def alpha(self):
        return self.alpha_min + self.alpha_span * torch.sigmoid(self.theta)

    def _core(self, q, k, v):
        B, H, T, hd = q.shape
        scale = hd ** -0.5
        pos = torch.arange(T, device=q.device)
        dks = self.alpha().to(q.device) * self.fib.to(q.device)   # (K,) continuous rung centers
        w = max(self._w, 0.25)
        R = int(math.ceil(2.5 * w))

        cols_k, cols_v, vfrom = [], [], []
        # local window {0..W}: exact (sharp) gathers
        for d in range(0, min(self.W, T - 1) + 1):
            cols_k.append(_gather(k, d)); cols_v.append(_gather(v, d)); vfrom.append(d)
        # Fibonacci rungs: forced Gaussian band gather (no sharp escape)
        for kk in range(self.fib.shape[0]):
            dk = dks[kk]
            c = int(round(float(dk.detach())))   # band center: non-diff index (grad flows via gw)
            if c <= self.W or c > T - 1:
                continue
            js = torch.arange(c - R, c + R + 1, device=q.device).clamp(1, T - 1)   # (M,)
            gw = torch.exp(-0.5 * ((js.float() - dk) / w) ** 2)
            gw = gw / gw.sum()                                                      # (M,)
            idx = (pos[:, None] - js[None, :]).clamp(min=0)                         # (T, M)
            kb = (gw[None, None, :, None] * k[:, :, idx, :]).sum(3)                 # (B,H,T,hd)
            vb = (gw[None, None, :, None] * v[:, :, idx, :]).sum(3)
            cols_k.append(kb); cols_v.append(vb); vfrom.append(c)

        Kg = torch.stack(cols_k, 3)                                                 # B,H,T,C,hd
        Vg = torch.stack(cols_v, 3)
        scores = (q.unsqueeze(3) * Kg).sum(-1) * scale                             # B,H,T,C
        valid = pos[:, None] >= torch.tensor(vfrom, device=q.device)[None, :]      # (T,C)
        scores = scores.masked_fill(~valid.view(1, 1, T, -1), NEG)
        attn = torch.softmax(scores, -1)
        return (attn.unsqueeze(-1) * Vg).sum(3)

    def forward(self, x, cos, sin, mask=None):
        B, T, _ = x.shape
        q, k, v = self._qkv(x, cos, sin)
        if self.use_checkpoint and self.training:
            out = checkpoint(self._core, q, k, v, use_reentrant=False)
        else:
            out = self._core(q, k, v)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.drop(self.o_proj(out))

    def report(self):
        return {"alpha": round(float(self.alpha()), 4)}
