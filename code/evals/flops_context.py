"""
flops_context.py -- FLOPs / params / inference-speed reporter. CONTEXT ONLY.

Per the spec, FLOPs and speed are reported for context and are NOT a promotion
criterion (the gather path is not faster than fused dense at seq=1024; the study
is about representation quality). Uses the analytic 2*params_nonembed forward
estimate (the L^2 attention term is ~16% of per-layer compute at d=512/seq=1024,
so this estimate is close for the dense part and a slight under-count for the
sparse-attention math; labeled analytic_estimate).

Returns {flops_per_token_fwd, params_total, params_nonembed, tokens_per_sec_infer};
eval_name = "flops_context".
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

_LDA_CODE = Path(r"K:\projects\Loop_Dev_AI\code")
if str(_LDA_CODE) not in sys.path:
    sys.path.insert(0, str(_LDA_CODE))
from baseline import param_count  # noqa: E402


@torch.no_grad()
def flops_context_eval(model: nn.Module, device, dtype,
                       batch=8, seq=None, warmup=2, timed=5) -> dict:
    seq = seq or model.cfg.seq_len
    pc = param_count(model)
    params_total = pc["total"]
    params_nonembed = params_total - pc["embed"]
    flops_per_token_fwd = 2 * params_nonembed  # analytic forward estimate

    tps = 0.0
    if device.type == "cuda":
        model.eval()
        idx = torch.randint(0, model.cfg.vocab_size, (batch, seq), device=device)
        ctx = torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32))
        for _ in range(warmup):
            with ctx:
                model(idx)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(timed):
            with ctx:
                model(idx)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / timed
        tps = (batch * seq) / dt

    return {
        "flops_per_token_fwd": float(flops_per_token_fwd),
        "params_total": float(params_total),
        "params_nonembed": float(params_nonembed),
        "tokens_per_sec_infer": round(tps, 1),
    }
