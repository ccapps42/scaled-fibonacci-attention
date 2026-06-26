"""
eval_multi.py — Multi-dataset perplexity evaluator.

Primary entry point: evaluate()
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from util import load_memmap


def evaluate(
    model: nn.Module,
    step: int,
    run_dir: str,
    dataset_paths: dict[str, str],
    batch_size: int = 8,
    n_tokens_per_dataset: int = 1 << 20,   # 1M tokens default
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    Compute mean cross-entropy and perplexity on the first N tokens of each dataset.

    Evaluation is fully deterministic (no shuffling). Reads data in contiguous
    windows of seq_len, padding the final incomplete window only if needed.

    Args:
        model:               A BaselineTransformer (or hook variant). Must have
                             cfg.seq_len and cfg.vocab_size attributes.
        step:                Current training step — used in output filename.
        run_dir:             Experiment directory (e.g. experiments/e0001__...).
        dataset_paths:       {name: path_to_.bin} — only datasets whose .bin files
                             exist are evaluated.
        batch_size:          Eval batch size. Use smaller values if VRAM is tight.
        n_tokens_per_dataset: How many leading tokens to evaluate per dataset.
        device:              torch device string or object.
        dtype:               Autocast dtype. Use torch.bfloat16.

    Returns:
        dict with keys: step, wall_seconds, datasets (nested per-dataset results).
        Also writes experiments/<run_dir>/eval_<step>.json.
    """
    device = torch.device(device) if isinstance(device, str) else device

    seq_len = model.cfg.seq_len

    was_training = model.training
    model.eval()

    t_start = time.monotonic()
    results: dict[str, dict] = {}

    with torch.no_grad():
        for name, path in dataset_paths.items():
            if not os.path.isfile(path):
                continue

            # Infer bin dtype from vocab size (nanoGPT convention)
            vocab_size = model.cfg.vocab_size
            bin_dtype = np.uint16 if vocab_size <= 65536 else np.uint32

            data = load_memmap(path, bin_dtype)
            n_available = len(data)
            n_eval = min(n_tokens_per_dataset, n_available - seq_len - 1)
            if n_eval <= 0:
                continue

            total_loss = 0.0
            total_tokens = 0

            # Walk data in contiguous non-overlapping windows
            # Each window: x=[i:i+seq_len], y=[i+1:i+seq_len+1]
            # We iterate over complete windows only (avoids padding complexity).
            i = 0
            batch_x_list: list[np.ndarray] = []
            batch_y_list: list[np.ndarray] = []

            def _flush_batch():
                nonlocal total_loss, total_tokens
                if not batch_x_list:
                    return
                bx = torch.from_numpy(
                    np.stack([arr.astype(np.int64) for arr in batch_x_list])
                ).to(device)
                by = torch.from_numpy(
                    np.stack([arr.astype(np.int64) for arr in batch_y_list])
                ).to(device)
                with torch.autocast(device_type=device.type, dtype=dtype):
                    _, loss = model(bx, by)
                # loss is mean CE over all tokens in the batch
                n_toks = bx.numel()
                total_loss += loss.item() * n_toks
                total_tokens += n_toks
                batch_x_list.clear()
                batch_y_list.clear()

            while i + seq_len + 1 <= n_eval + seq_len:
                if i + seq_len + 1 > n_available:
                    break
                batch_x_list.append(data[i : i + seq_len])
                batch_y_list.append(data[i + 1 : i + seq_len + 1])
                i += seq_len  # non-overlapping: advance by full seq_len

                if len(batch_x_list) >= batch_size:
                    _flush_batch()

                if total_tokens >= n_tokens_per_dataset:
                    break

            _flush_batch()

            if total_tokens == 0:
                continue

            mean_loss = total_loss / total_tokens
            ppl = math.exp(mean_loss)
            results[name] = {
                "ppl": round(ppl, 4),
                "loss": round(mean_loss, 6),
                "n_tokens": total_tokens,
            }

    wall = time.monotonic() - t_start

    output = {
        "step": step,
        "wall_seconds": round(wall, 2),
        "datasets": results,
    }

    # Write eval JSON
    os.makedirs(run_dir, exist_ok=True)
    out_path = os.path.join(run_dir, f"eval_{step}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    if was_training:
        model.train()

    return output
