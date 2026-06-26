"""
gumbel_fib.py -- the ORIGINAL design: scaled-Fibonacci sparse attention with one
learned per-layer scalar alpha, made trainable by Gumbel-softmax selection over a
candidate-distance SUPERSET (Capps concept doc).

Unlike the soft-width additive bias (escapable by dense QK), this HARD-SELECTS:
for each Fibonacci step k it Gumbel-picks the superset candidate nearest the current
ideal distance alpha*fib_k, and the model attends ONLY to the K selected rungs (+ the
local window). No dense path -> alpha is forced to move to reach a useful distance.
Temperature tau is annealed high->low (soft exploration -> hard sparse selection).

  candidates = { round(a*f) : a in linspace(0.5,1.5,n), f in fib, 0<d<seq }
  ideal_k    = alpha * fib_k
  sel_k      = gumbel_softmax(-|candidates - ideal_k|, tau)        # soft, ->one-hot as tau->0
  key_k      = sum_c sel_k[c] * K[t - candidates[c]]
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_LDA = Path(r"K:\projects\Loop_Dev_AI\code")
if str(_LDA) not in sys.path:
    sys.path.insert(0, str(_LDA))
from baseline import _apply_rope                        # noqa: E402
_HERE = Path(__file__).parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))
from fib_attention import fib_base, _gather             # noqa: E402

NEG = -1e9


def build_superset(base, amin, amax, n, seq):
    cands = set()
    for i in range(n):
        a = amin + (amax - amin) * i / max(1, n - 1)
        for f in base:
            d = round(a * f)
            if 0 < d < seq:
                cands.add(d)
    return sorted(cands)


class GumbelFibAttention(nn.Module):
    def __init__(self, cfg, layer_idx, W=8, K=15, alpha_min=0.5, alpha_span=1.0,
                 superset_n=13, tau_init=2.0):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.W = W
        self.alpha_min, self.alpha_span = alpha_min, alpha_span
        self._tau = tau_init
        d = cfg.d_model

        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.o_proj.is_residual_proj = True

        base = fib_base(K)
        self.register_buffer("fib", torch.tensor(base, dtype=torch.float32), persistent=False)
        cands = build_superset(base, alpha_min, alpha_span + alpha_min, superset_n, cfg.seq_len)
        self.register_buffer("cands", torch.tensor(cands, dtype=torch.float32), persistent=False)
        self.theta = nn.Parameter(torch.zeros(1))   # alpha = amin + span*sigmoid(theta); init alpha=1.0

    def alpha(self):
        return self.alpha_min + self.alpha_span * torch.sigmoid(self.theta)

    def report(self):
        return {"alpha": round(float(self.alpha()), 4)}

    def forward(self, x, cos, sin, mask=None):
        B, T, _ = x.shape
        H, hd = self.n_heads, self.head_dim
        q = self.q_proj(x).view(B, T, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, hd).transpose(1, 2)
        q = _apply_rope(q, cos, sin); k = _apply_rope(k, cos, sin)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1); v = v.repeat_interleave(self.n_rep, dim=1)

        ideal = self.alpha() * self.fib                       # (K,)
        logits = -(self.cands.unsqueeze(0) - ideal.unsqueeze(1)).abs()   # (K, C)
        sel = F.gumbel_softmax(logits, tau=max(self._tau, 0.05), hard=not self.training, dim=-1)  # (K,C)

        # gather candidate keys/values once: (C, B, H, T, hd)
        Kc = torch.stack([_gather(k, int(c)) for c in self.cands.tolist()], dim=0)
        Vc = torch.stack([_gather(v, int(c)) for c in self.cands.tolist()], dim=0)
        Kdim = self.fib.shape[0]
        # combine to K selected rungs via matmul (K,C)@(C, B*H*T*hd) -> no K*C*... intermediate
        Cn = Kc.shape[0]
        Kr = (sel @ Kc.reshape(Cn, -1)).reshape(Kdim, B, H, T, hd).permute(1, 2, 3, 0, 4)
        Vr = (sel @ Vc.reshape(Cn, -1)).reshape(Kdim, B, H, T, hd).permute(1, 2, 3, 0, 4)
        cand_dist = (sel * self.cands.unsqueeze(0)).sum(-1)   # (K,) effective distance per rung

        # window (self + local) keys: distances 0..W
        win = list(range(0, min(self.W, T - 1) + 1))
        Kw = torch.stack([_gather(k, dd) for dd in win], dim=0)   # (Wn,B,H,T,hd)
        Vw = torch.stack([_gather(v, dd) for dd in win], dim=0)
        Kw = Kw.permute(1, 2, 3, 0, 4); Vw = Vw.permute(1, 2, 3, 0, 4)  # (B,H,T,Wn,hd)

        Kall = torch.cat([Kw, Kr], dim=3); Vall = torch.cat([Vw, Vr], dim=3)  # (B,H,T,M,hd)
        scale = hd ** -0.5
        scores = (q.unsqueeze(3) * Kall).sum(-1) * scale     # (B,H,T,M)

        # causal validity: a column at distance dist valid for position t iff t >= dist
        pos = torch.arange(T, device=x.device)
        win_d = torch.tensor(win, dtype=torch.float32, device=x.device)
        all_d = torch.cat([win_d, cand_dist])                # (M,)
        valid = pos.unsqueeze(1) >= all_d.unsqueeze(0)        # (T,M)
        scores = scores.masked_fill(~valid.view(1, 1, T, -1), NEG)

        attn = torch.softmax(scores, dim=-1)
        out = (attn.unsqueeze(-1) * Vall).sum(3).transpose(1, 2).contiguous().view(B, T, H * hd)
        return self.o_proj(out)


def make_gumbel_factory(W=8, K=15, superset_n=13, tau_init=2.0):
    def attn_factory(cfg, layer_idx):
        return GumbelFibAttention(cfg, layer_idx, W=W, K=K, superset_n=superset_n, tau_init=tau_init)
    return {"attn_factory": attn_factory}
