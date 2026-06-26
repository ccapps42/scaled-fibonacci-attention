"""
posloss.py -- position-resolved validation loss for trained fib runs.

Tests the core thesis: does the staggering / Fibonacci win CONCENTRATE at long
context positions (where wider coverage should help), or is it uniform? Bucketed
NLL vs token-position within the seq_len window. Eval-only; reuses checkpoints.

  python code/posloss.py                         # default key runs, all val sets
  python code/posloss.py --runs f11__fib_stagger_w8 f19__logsparse_stagger_w8
  python code/posloss.py --tokens 1048576 --device cuda

Writes posloss.json {run: {dataset: {per_position:[...], binned:[...], n_windows}}}.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from util import load_memmap                          # noqa: E402
from eval_posthoc import build_my_model, EXPERIMENTS_DIR  # noqa: E402

LDA_DATA_DIR = r"K:\projects\Loop_Dev_AI\data"
VAL_BINS = {
    "fineweb_edu_val": "val/fineweb_edu_val.bin",
    "wikipedia_val":   "val/wikipedia_val.bin",
    "tinystories_val": "val/tinystories_val.bin",
    "math_val":        "val/math_val.bin",
}
# all 21 runs; plain logsparse only exists at W8, the rest sweep W6/8/10/12
DEFAULT_RUNS = [
    "f09__logsparse_w8",
    "f18__logsparse_stagger_w6", "f19__logsparse_stagger_w8",
    "f20__logsparse_stagger_w10", "f21__logsparse_stagger_w12",
    "f05__fib_fixed_w6", "f06__fib_fixed_w8", "f07__fib_fixed_w10", "f08__fib_fixed_w12",
    "f01__fib_learned_w6", "f02__fib_learned_w8", "f03__fib_learned_w10", "f04__fib_learned_w12",
    "f10__fib_stagger_w6", "f11__fib_stagger_w8", "f12__fib_stagger_w10", "f13__fib_stagger_w12",
    "f14__fib_hdc_w6", "f15__fib_hdc_w8", "f16__fib_hdc_w10", "f17__fib_hdc_w12",
]
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
N_BINS = 8


@torch.no_grad()
def posloss_one(model, paths, n_tokens, batch_size, device, dtype):
    seq_len = model.cfg.seq_len
    vocab = model.cfg.vocab_size
    bd = np.uint16 if vocab <= 65536 else np.uint32
    out = {}
    for name, path in paths.items():
        if not os.path.isfile(path):
            continue
        data = load_memmap(path, bd)
        n = len(data)
        sl = torch.zeros(seq_len - 1, dtype=torch.float64, device=device)  # summed loss / position
        nw = 0
        toks = 0
        batch: list[np.ndarray] = []

        def flush():
            nonlocal toks, nw
            if not batch:
                return
            x = torch.from_numpy(np.stack(batch).astype(np.int64)).to(device)
            with torch.autocast(device_type=device.type, dtype=dtype):
                logits, _ = model(x)
            lp = logits[:, :-1, :].reshape(-1, vocab).float()
            tgt = x[:, 1:].reshape(-1)
            ce = F.cross_entropy(lp, tgt, reduction="none").view(x.size(0), seq_len - 1)
            sl.add_(ce.sum(0).double())
            nw += x.size(0)
            toks += x.numel()
            batch.clear()

        i = 0
        while i + seq_len + 1 <= n:
            batch.append(data[i:i + seq_len])
            i += seq_len
            if len(batch) >= batch_size:
                flush()
            if toks >= n_tokens:
                break
        flush()
        if nw == 0:
            continue
        per = (sl / nw).cpu().tolist()
        # bin the seq_len-1 positions into N_BINS contiguous buckets (mean NLL each)
        edges = np.linspace(0, seq_len - 1, N_BINS + 1, dtype=int)
        binned = [round(float(np.mean(per[edges[b]:edges[b + 1]])), 5) for b in range(N_BINS)]
        out[name] = {"per_position": [round(v, 5) for v in per],
                     "binned": binned, "n_windows": nw, "seq_len": seq_len}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--tokens", type=int, default=1 << 20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    ap.add_argument("--out", default=os.path.join(_HERE.parent, "posloss.json"))
    args = ap.parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    dtype = DTYPES[args.dtype]
    paths = {k: os.path.join(LDA_DATA_DIR, v) for k, v in VAL_BINS.items()}

    results = {}
    if os.path.exists(args.out):
        results = json.load(open(args.out, encoding="utf-8"))
    for rid in args.runs:
        run_dir = os.path.join(EXPERIMENTS_DIR, rid)
        print(f"=== {rid} (seq_len-bucketed NLL, {args.tokens} tok/set) ===", flush=True)
        model = build_my_model(run_dir, device)
        res = posloss_one(model, paths, args.tokens, args.batch_size, device, dtype)
        results[rid] = res
        for ds, r in res.items():
            print(f"  {ds:16} binned NLL: " + " ".join(f"{v:6.3f}" for v in r["binned"]))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        json.dump(results, open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {args.out}  (bins = early-context -> late-context within the window)")


if __name__ == "__main__":
    main()
