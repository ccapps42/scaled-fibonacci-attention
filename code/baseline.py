"""
baseline.py — Modern vanilla decoder-only transformer.

Architecture: RMSNorm pre-norm, SwiGLU FFN, RoPE, GQA, tied input/output embeddings.

NovelComponent hook system
--------------------------
BaselineConfig accepts optional factory callables. When set, these replace the
corresponding default component everywhere it would be instantiated. Factories
are called once per layer (or once globally for norm_factory). Signatures:

    attn_factory(config: BaselineConfig, layer_idx: int) -> nn.Module
        Must implement forward(x: Tensor, cos: Tensor, sin: Tensor,
                              mask: Tensor | None) -> Tensor
        Receives pre-normalized x, returns residual-ready output (same shape).

    ffn_factory(config: BaselineConfig, layer_idx: int) -> nn.Module
        Must implement forward(x: Tensor) -> Tensor
        Receives pre-normalized x, returns residual-ready output (same shape).

    norm_factory(config: BaselineConfig) -> Callable[[int], nn.Module]
        Returns a per-norm constructor f(d_model) -> nn.Module.
        Used for all RMSNorm instances (attn pre-norm, ffn pre-norm, final norm).

    pre_layer_hook(x: Tensor, layer_idx: int) -> Tensor
        Called on the residual stream BEFORE attn pre-norm at each layer.
        Must return a tensor with the same shape as x.

    post_layer_hook(x: Tensor, layer_idx: int) -> Tensor
        Called on the residual stream AFTER the FFN residual add at each layer.
        Must return a tensor with the same shape as x.

Example usage (a variant that swaps in a custom attention module):
    def my_attn_factory(cfg, layer_idx):
        return MyCustomAttention(cfg.d_model, cfg.n_heads, cfg.n_kv_heads)

    cfg = BaselineConfig(
        d_model=256, n_layers=12, ...,
        attn_factory=my_attn_factory
    )
    model = BaselineTransformer(cfg)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BaselineConfig:
    d_model: int = 256
    n_layers: int = 12
    n_heads: int = 8
    n_kv_heads: int = 2           # GQA: n_heads must be divisible by n_kv_heads
    vocab_size: int = 32768
    seq_len: int = 1024
    ffn_ratio: float = 8 / 3     # SwiGLU hidden = round(d_model * ffn_ratio / 2) * 2
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True
    dropout: float = 0.0

    # --- NovelComponent hooks (all default None → use baseline implementations) ---
    attn_factory: Callable | None = field(default=None, repr=False)
    ffn_factory: Callable | None = field(default=None, repr=False)
    norm_factory: Callable | None = field(default=None, repr=False)
    pre_layer_hook: Callable | None = field(default=None, repr=False)
    post_layer_hook: Callable | None = field(default=None, repr=False)

    # --- MoE aux loss coefficient (0.0 = no-op for non-MoE models) ---
    moe_aux_coef: float = 0.0

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"


# Default layer counts for deep-narrow ratios (used by factory helpers)
DEEP_NARROW_DEFAULTS: dict[int, dict] = {
    128: {"n_layers": 16, "n_heads": 4,  "n_kv_heads": 2},
    256: {"n_layers": 12, "n_heads": 8,  "n_kv_heads": 2},
    512: {"n_layers": 16, "n_heads": 8,  "n_kv_heads": 2},
    768: {"n_layers": 18, "n_heads": 12, "n_kv_heads": 4},
}


def make_config(d_model: int, **overrides) -> BaselineConfig:
    """Convenience: create a BaselineConfig with deep-narrow defaults for common widths."""
    defaults = DEEP_NARROW_DEFAULTS.get(d_model, {"n_layers": 12, "n_heads": d_model // 32, "n_kv_heads": 2})
    kwargs = {**defaults, "d_model": d_model, **overrides}
    return BaselineConfig(**kwargs)


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def _make_norm(cfg: BaselineConfig) -> Callable[[int], nn.Module]:
    """Return a norm constructor, hooked if norm_factory is provided."""
    if cfg.norm_factory is not None:
        return cfg.norm_factory(cfg)
    return lambda d: RMSNorm(d, cfg.rms_norm_eps)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

def _build_rope_cache(seq_len: int, head_dim: int, theta: float, device: torch.device):
    """Precompute cos/sin for RoPE. Returns (seq_len, head_dim/2) each."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)  # (seq_len, head_dim/2)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, head_dim); cos/sin: (T, head_dim/2) → broadcast."""
    # Expand cos/sin to match x dims
    cos = cos[: x.shape[2], :].unsqueeze(0).unsqueeze(0)   # (1, 1, T, head_dim/2)
    sin = sin[: x.shape[2], :].unsqueeze(0).unsqueeze(0)
    cos = torch.cat([cos, cos], dim=-1)  # (1, 1, T, head_dim)
    sin = torch.cat([sin, sin], dim=-1)
    return (x * cos) + (_rotate_half(x) * sin)


# ---------------------------------------------------------------------------
# GQA Attention
# ---------------------------------------------------------------------------

class GQAAttention(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads  # groups per KV head
        self.head_dim = cfg.d_model // cfg.n_heads
        d = cfg.d_model

        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.o_proj.is_residual_proj = True  # depth-scaled init (GPT-2 residual init)

        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Expand KV heads to match Q heads for standard matmul
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled dot-product attention (uses FlashAttention when available)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=0.0,  # dropout handled separately
            is_causal=(mask is None),  # use causal mask if no explicit mask
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.drop(self.o_proj(attn_out))


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------

def _swiglu_hidden(d_model: int, ratio: float) -> int:
    """Round to nearest even number for tensor core alignment."""
    h = int(d_model * ratio)
    return (h + 1) // 2 * 2


class SwiGLUFFN(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        hidden = _swiglu_hidden(cfg.d_model, cfg.ffn_ratio)
        d = cfg.d_model
        self.gate = nn.Linear(d, hidden, bias=False)
        self.up   = nn.Linear(d, hidden, bias=False)
        self.down  = nn.Linear(hidden, d, bias=False)
        self.down.is_residual_proj = True  # depth-scaled init (GPT-2 residual init)
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


# ---------------------------------------------------------------------------
# Transformer Layer
# ---------------------------------------------------------------------------

class TransformerLayer(nn.Module):
    def __init__(self, cfg: BaselineConfig, layer_idx: int, make_norm: Callable):
        super().__init__()
        self.layer_idx = layer_idx
        self.pre_layer_hook = cfg.pre_layer_hook
        self.post_layer_hook = cfg.post_layer_hook

        self.attn_norm = make_norm(cfg.d_model)
        self.ffn_norm  = make_norm(cfg.d_model)

        self.attn = cfg.attn_factory(cfg, layer_idx) if cfg.attn_factory else GQAAttention(cfg)
        self.ffn  = cfg.ffn_factory(cfg, layer_idx)  if cfg.ffn_factory  else SwiGLUFFN(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.pre_layer_hook is not None:
            x = self.pre_layer_hook(x, self.layer_idx)

        # Attention with pre-norm and residual
        x = x + self.attn(self.attn_norm(x), cos, sin, mask)

        # FFN with pre-norm and residual
        x = x + self.ffn(self.ffn_norm(x))

        if self.post_layer_hook is not None:
            x = self.post_layer_hook(x, self.layer_idx)

        return x


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class BaselineTransformer(nn.Module):
    def __init__(self, cfg: BaselineConfig):
        super().__init__()
        self.cfg = cfg

        make_norm = _make_norm(cfg)

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(
            [TransformerLayer(cfg, i, make_norm) for i in range(cfg.n_layers)]
        )
        self.final_norm = make_norm(cfg.d_model)

        # Tied output projection: weight shared with embed
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # RoPE cache: registered as buffer so it moves with .to(device) calls
        head_dim = cfg.d_model // cfg.n_heads
        cos, sin = _build_rope_cache(cfg.seq_len, head_dim, cfg.rope_theta, device=torch.device("cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self):
        # Base std scales with width; residual output projections (attn o_proj,
        # FFN down) are additionally scaled by 1/sqrt(2*n_layers) so residual-
        # stream variance stays bounded with depth (GPT-2 residual init).
        std = self.cfg.d_model ** -0.5
        residual_std = std / math.sqrt(2 * self.cfg.n_layers)
        nn.init.normal_(self.embed.weight, std=std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                s = residual_std if getattr(module, "is_residual_proj", False) else std
                nn.init.normal_(module.weight, std=s)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Re-apply MoE internal init so router/gate_up/down keep RTCC's std=0.02
        # (the generic loop above overwrites them via the router's nn.Linear weight).
        for layer in self.layers:
            ffn = layer.ffn
            if hasattr(ffn, "moe") and hasattr(ffn.moe, "_init_weights"):
                ffn.moe._init_weights()

    def forward(
        self,
        idx: torch.Tensor,              # (B, T) token ids
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert T <= self.cfg.seq_len, f"Input length {T} exceeds seq_len {self.cfg.seq_len}"

        x = self.embed(idx)

        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Standard next-token CE; flatten for F.cross_entropy
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
            )
            # MoE aux loss: collect from any FFN that set .aux_loss during forward
            if self.cfg.moe_aux_coef:
                aux_total = sum(
                    layer.ffn.aux_loss
                    for layer in self.layers
                    if getattr(layer.ffn, "aux_loss", None) is not None
                )
                if aux_total != 0:
                    loss = loss + self.cfg.moe_aux_coef * aux_total

        return logits, loss


# ---------------------------------------------------------------------------
# Param count helper
# ---------------------------------------------------------------------------

def param_count(model: BaselineTransformer) -> dict:
    """
    Returns total param count and breakdown by component type.
    With tied embeddings the lm_head params are NOT double-counted.
    """
    cfg = model.cfg

    def _n(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters())

    # Collect per-layer attn and ffn counts
    attn_total = sum(_n(l.attn) for l in model.layers)
    ffn_total  = sum(_n(l.ffn)  for l in model.layers)
    norm_total = (
        sum(_n(l.attn_norm) + _n(l.ffn_norm) for l in model.layers)
        + _n(model.final_norm)
    )

    embed_total = _n(model.embed)
    # If tied, lm_head.weight IS embed.weight — count embed once
    lm_head_extra = 0 if cfg.tie_embeddings else _n(model.lm_head)

    other = (
        sum(p.numel() for p in model.parameters())
        - embed_total - lm_head_extra - attn_total - ffn_total - norm_total
    )

    total = sum(p.numel() for p in model.parameters())

    return {
        "total": total,
        "embed": embed_total + lm_head_extra,
        "attn": attn_total,
        "ffn": ffn_total,
        "norm": norm_total,
        "other": other,
    }
