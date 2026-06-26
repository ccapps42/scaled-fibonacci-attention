"""
util.py — Shared helpers for the Loop_Dev_AI training harness.
"""

from __future__ import annotations

import os
import re

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Experiment ID generation
# ---------------------------------------------------------------------------

_EXPDIR_RE = re.compile(r"^e(\d{4})__")


def gen_run_id(short_slug: str, experiments_dir: str = "experiments") -> str:
    """
    Generate the next available run ID in the form e<NNNN>__<short_slug>.

    Scans experiments_dir for directories matching e<NNNN>__ and picks
    max(existing) + 1. Safe for single-orchestrator use (no file locking needed).
    """
    base = os.path.abspath(experiments_dir)
    os.makedirs(base, exist_ok=True)

    max_n = -1
    try:
        for name in os.listdir(base):
            m = _EXPDIR_RE.match(name)
            if m:
                n = int(m.group(1))
                if n > max_n:
                    max_n = n
    except FileNotFoundError:
        pass

    next_n = max_n + 1
    return f"e{next_n:04d}__{short_slug}"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set seeds for torch, numpy, and cuda. Also requests deterministic algorithms."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Avoids non-deterministic CUDA ops; may slow some kernels but is acceptable
    # for our single-GPU, no-compile setup.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_memmap(path: str, dtype: np.dtype | str) -> np.memmap:
    """Open a tokenized .bin file as a read-only numpy memmap."""
    return np.memmap(path, dtype=dtype, mode="r")


def sample_batch(
    data: np.memmap,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample batch_size random windows from data (with replacement).

    Returns (x, y) where y = x shifted right by one token (next-token targets).
    Each window is [i : i+seq_len+1]; x uses the first seq_len tokens, y the last.
    """
    n = len(data)
    max_start = n - seq_len - 1
    assert max_start > 0, f"Data too small: {n} tokens for seq_len={seq_len}"

    starts = np.random.randint(0, max_start, size=(batch_size,))

    # Build int64 buffers to avoid overflow issues with uint16/uint32 on some ops
    x = np.stack([data[s : s + seq_len].astype(np.int64) for s in starts])
    y = np.stack([data[s + 1 : s + seq_len + 1].astype(np.int64) for s in starts])

    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)
    return x, y


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_int(n: int | float) -> str:
    """
    Format a large integer in human-readable SI units.

    Examples: 128 → '128', 12_345_678 → '12.3M', 1_234_567_890 → '1.2B'
    """
    n = int(n)
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
