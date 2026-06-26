"""
lenextrap.py -- length-extrapolation perplexity for trained fib runs.

Models were trained at seq_len=1024 (RoPE theta=10000). Here we rebuild each model
at LONGER eval lengths and measure aggregate PPL: does the structured sparse reach
degrade more gracefully than plain logsparse (and than dense, if --include_dense)?

Rebuild-at-length is valid: RoPE cache and the bool-mask rung set are non-persistent
buffers (not in the state_dict), so a 1024-trained checkpoint loads into a model built
at any seq_len. CAVEAT to report: this is vanilla RoPE extrapolation (no scaling), and
the fib rung set (K=15) caps reach at ~1.5*987~=1481, so beyond that distance only the
local window + multi-hop reach applies -- a real property of fixed-K sparse attention.

  python code/lenextrap.py                          # default runs, lengths 1024/2048/4096
  python code/lenextrap.py --lengths 1024 2048 --device cuda
  python code/lenextrap.py --include_dense          # add e0000 dense reference line

Writes lenextrap.json {run: {length: {dataset: ppl}}}.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from util import load_memmap                          # noqa: E402
from eval_posthoc import EXPERIMENTS_DIR, COMPARATORS  # noqa: E402

LDA_DATA_DIR = r"K:\projects\Loop_Dev_AI\data"
VAL_BINS = {
    "fineweb_edu_val": "val/fineweb_edu_val.bin",
    "wikipedia_val":   "val/wikipedia_val.bin",
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
DEFAULT_LENGTHS = [1024, 2048, 4096]
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
# shrink batch as length grows to stay within VRAM (bool-mask path is O(T^2))
BATCH_FOR_LEN = {1024: 8, 2048: 4, 4096: 2, 8192: 1}


def build_at_len(run_dir, seq_len, device):
    """Rebuild a fib run's model at a target seq_len and load its trained weights."""
    from baseline import BaselineConfig, BaselineTransformer
    from fib_attention import make_factories
    variant = json.load(open(os.path.join(run_dir, "fib_variant.json"), encoding="utf-8"))
    model_cfg = json.load(open(os.path.join(run_dir, "config.json"), encoding="utf-8"))["model"]
    variant = {**variant, "impl": "materialized"}
    bcfg = BaselineConfig(
        d_model=model_cfg["d_model"], n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"], n_kv_heads=model_cfg["n_kv_heads"],
        vocab_size=model_cfg.get("vocab_size", 32768),
        seq_len=seq_len,
        attn_factory=make_factories(**variant)["attn_factory"],
    )
    model = BaselineTransformer(bcfg)
    ckpt = torch.load(os.path.join(run_dir, "ckpt_final.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def build_dense_at_len(run_dir, seq_len, device):
    """Rebuild the dense baseline (attn_factory=None) at a target seq_len -- lets the
    dense reference extrapolate the same way the fib runs do, instead of the LDA loader
    which pins seq_len to 1024. Reads e0000 READ-ONLY; same arch defaults as the fib runs."""
    from baseline import BaselineConfig, BaselineTransformer
    m = json.load(open(os.path.join(run_dir, "config.json"), encoding="utf-8"))["model"]
    kw = dict(d_model=m["d_model"], n_layers=m["n_layers"], n_heads=m["n_heads"],
              n_kv_heads=m["n_kv_heads"], vocab_size=m.get("vocab_size", 32768),
              seq_len=seq_len, attn_factory=None)
    for k in ("ffn_ratio", "rope_theta", "rms_norm_eps", "tie_embeddings", "dropout"):
        if k in m:
            kw[k] = m[k]
    model = BaselineTransformer(BaselineConfig(**kw))
    ckpt = torch.load(os.path.join(run_dir, "ckpt_final.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def ppl_at_len(model, paths, n_tokens, batch_size, device, dtype):
    seq_len = model.cfg.seq_len
    vocab = model.cfg.vocab_size
    bd = np.uint16 if vocab <= 65536 else np.uint32
    out = {}
    for name, path in paths.items():
        if not os.path.isfile(path):
            continue
        data = load_memmap(path, bd)
        n = len(data)
        if n < seq_len + 2:
            continue
        tot_loss = 0.0
        tot_tok = 0
        batch: list[np.ndarray] = []

        def flush():
            nonlocal tot_loss, tot_tok
            if not batch:
                return
            x = torch.from_numpy(np.stack(batch).astype(np.int64)).to(device)
            y = torch.from_numpy(np.stack([a for a in batch_y]).astype(np.int64)).to(device)
            with torch.autocast(device_type=device.type, dtype=dtype):
                _, loss = model(x, y)
            tot_loss += loss.item() * x.numel()
            tot_tok += x.numel()
            batch.clear()
            batch_y.clear()

        batch_y: list[np.ndarray] = []
        i = 0
        while i + seq_len + 1 <= n:
            batch.append(data[i:i + seq_len])
            batch_y.append(data[i + 1:i + seq_len + 1])
            i += seq_len
            if len(batch) >= batch_size:
                flush()
            if tot_tok >= n_tokens:
                break
        flush()
        if tot_tok == 0:
            continue
        out[name] = round(math.exp(tot_loss / tot_tok), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=DEFAULT_RUNS,
                    help="pass '--runs' with no names + --include_dense to fill ONLY the dense curve")
    ap.add_argument("--lengths", nargs="+", type=int, default=DEFAULT_LENGTHS)
    ap.add_argument("--tokens", type=int, default=1 << 20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    ap.add_argument("--include_dense", action="store_true",
                    help="also run the e0000 dense baseline as a reference line")
    ap.add_argument("--out", default=os.path.join(_HERE.parent, "lenextrap.json"))
    args = ap.parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    dtype = DTYPES[args.dtype]
    paths = {k: os.path.join(LDA_DATA_DIR, v) for k, v in VAL_BINS.items()}

    results = {}
    if os.path.exists(args.out):
        results = json.load(open(args.out, encoding="utf-8"))

    builders = [(rid, lambda rd, L: build_at_len(rd, L, device),
                 os.path.join(EXPERIMENTS_DIR, rid)) for rid in args.runs]
    if args.include_dense:
        # dense reference rebuilt at each eval length (same as the fib runs), so it
        # extrapolates to 2048/4096 instead of capping at its trained 1024.
        builders.append(("ref_e0000", lambda rd, L: build_dense_at_len(rd, L, device),
                         COMPARATORS["e0000"]))

    for rid, build, run_dir in builders:
        results.setdefault(rid, {})
        for L in args.lengths:
            bs = BATCH_FOR_LEN.get(L, 1)
            try:
                model = build(run_dir, L)
                if model.cfg.seq_len < L:
                    print(f"  {rid} @ {L}: model caps at seq_len={model.cfg.seq_len}, skipping")
                    del model
                    continue
                res = ppl_at_len(model, paths, args.tokens, bs, device, dtype)
                results[rid][str(L)] = res
                cells = " ".join(f"{k.split('_')[0]}={v}" for k, v in res.items())
                print(f"  {rid:28} @ T={L:5} (bs={bs}): {cells}", flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as exc:
                print(f"  {rid} @ {L}: FAILED {exc!r}")
            json.dump(results, open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
