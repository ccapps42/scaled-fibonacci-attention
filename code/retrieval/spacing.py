"""
spacing.py -- SoftWidthSpacingAttention: a learnable sparse-attention spacing made
TRAINABLE by a non-local soft-width relaxation.

The naive interpolated gather (fib_attention) gives the spacing parameter only a
local (+-1) gradient, so it never moves from init (see memory: flat exploration
landscape). Here each rung contributes a GAUSSIAN bump (width w, annealed wide->narrow)
to a relative-position log-bias added to dense attention scores. Wide w early -> a rung
"feels" targets hundreds of tokens away -> real gradient on the spacing param. As w->0
the bias collapses to a hard sparse mask at the rung offsets (true sparsity at inference).

Spacing parameterizations (the reviewer ablation):
  fib_scale : offsets = alpha * fib_k         (one learned scalar alpha)   <- the headline
  geom_base : offsets = b**k                  (one learned base b)         <- "learnable geometric"
  free      : offsets = fib_k + delta_k       (K learned per-offset deltas) <- DCLS-style free

Set module._w each step to anneal the width. Dense O(T^2) during training; fine for
the probe scale. Reuses baseline projections + RoPE.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_LDA = Path(r"K:\projects\Loop_Dev_AI\code")
if str(_LDA) not in sys.path:
    sys.path.insert(0, str(_LDA))
from baseline import _apply_rope                       # noqa: E402

_HERE = Path(__file__).parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
from fib_attention import fib_base                     # noqa: E402


class SoftWidthSpacingAttention(nn.Module):
    def __init__(self, cfg, layer_idx, mode="fib_scale", W=8, K=15,
                 alpha_min=0.5, alpha_span=1.0, w_init=60.0):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.W, self.mode = W, mode
        self.alpha_min, self.alpha_span = alpha_min, alpha_span
        self._w = w_init                      # current Gaussian width (annealed externally)
        d = cfg.d_model

        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.o_proj.is_residual_proj = True

        base = torch.tensor(fib_base(K), dtype=torch.float32)
        self.register_buffer("fib", base, persistent=False)
        self.K = K
        if mode == "fib_scale":
            self.theta = nn.Parameter(torch.zeros(1))          # alpha = amin + span*sigmoid(theta)
        elif mode == "geom_base":
            self.logb = nn.Parameter(torch.tensor([math.log(1.618)]))  # base b = exp(logb), init phi
        elif mode == "free":
            self.delta = nn.Parameter(torch.zeros(K))          # offsets = fib + delta
        else:
            raise ValueError(mode)

    def offsets(self):
        if self.mode == "fib_scale":
            alpha = self.alpha_min + self.alpha_span * torch.sigmoid(self.theta)
            return alpha * self.fib
        if self.mode == "geom_base":
            b = self.logb.exp()
            ks = torch.arange(1, self.K + 1, device=self.logb.device, dtype=torch.float32)
            return b ** ks
        return self.fib + self.delta                           # free

    def report(self):
        if self.mode == "fib_scale":
            return {"alpha": round(float(self.alpha_min + self.alpha_span * torch.sigmoid(self.theta)), 4)}
        if self.mode == "geom_base":
            return {"base": round(float(self.logb.exp()), 4)}
        return {"offsets": [round(float(o), 1) for o in self.offsets().detach().cpu()]}

    def forward(self, x, cos, sin, mask=None):
        B, T, _ = x.shape
        H, hd = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, hd).transpose(1, 2)
        q = _apply_rope(q, cos, sin); k = _apply_rope(k, cos, sin)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1); v = v.repeat_interleave(self.n_rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) * (hd ** -0.5)      # (B,H,T,T)

        dev = x.device
        ii = torch.arange(T, device=dev).unsqueeze(1)
        jj = torch.arange(T, device=dev).unsqueeze(0)
        delta = (ii - jj).float()                              # (T,T), >=0 is causal/past
        centers = self.offsets().to(dev)                       # (K,)
        w = max(self._w, 0.25)
        bump = torch.exp(-0.5 * ((delta.unsqueeze(-1) - centers) / w) ** 2).sum(-1)  # (T,T)
        window = ((delta >= 0) & (delta <= self.W)).float()    # self + local window always on
        weight = bump + window
        bias = torch.log(weight + 1e-6)
        bias = bias.masked_fill(delta < 0, float("-inf"))      # causal
        scores = scores + bias.unsqueeze(0).unsqueeze(0)

        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H * hd)
        return self.o_proj(out)


def make_softwidth_factory(mode="fib_scale", W=8, K=15, w_init=60.0):
    def attn_factory(cfg, layer_idx):
        return SoftWidthSpacingAttention(cfg, layer_idx, mode=mode, W=W, K=K, w_init=w_init)
    return {"attn_factory": attn_factory}
