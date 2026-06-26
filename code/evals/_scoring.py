"""Shared scoring helper for the synthetic multi-hop evals."""
from __future__ import annotations

import torch


@torch.no_grad()
def choose_among(model, prefix_ids, candidate_ids, device, dtype) -> int:
    """One forward pass; return the candidate token with the highest next-token logit.

    Forced choice among the values actually present in the context -> isolates the
    multi-hop resolution from whether the model learned the output format, and gives
    a clean chance baseline of 1/len(candidate_ids).
    """
    ids = prefix_ids[-model.cfg.seq_len:]
    t = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
        logits, _ = model(t)
    last = logits[0, -1, :]
    cand = torch.tensor(candidate_ids, device=device, dtype=torch.long)
    return int(cand[last[cand].argmax()].item())
