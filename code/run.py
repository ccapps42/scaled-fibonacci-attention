"""
run.py -- launcher for the scaled-Fibonacci matrix (f01..f09).

Uses the shared training loop (train.py / baseline.py), copied into this code/
directory, with db_path / experiments_dir / data_dir pointed at THIS project.

Recipe matched to comparators e0000/e0030/e0084: 13000 steps, 32,768 tok/step
(learned: B=4 x accum 8; controls: B=8 x accum 4), lr 3e-4, wd 0.1, betas 0.9/0.95,
warmup 300, bf16, seed 42, screen_train.bin.

  python code/run.py --list
  python code/run.py --run f02__fib_learned_w8
  python code/run.py --run all            # sequential, in matrix order
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).parent
# baseline/train/db/util/eval_multi/eval_cheap live in this directory, so this
# dir alone satisfies the imports below.
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import torch                                              # noqa: E402
from baseline import BaselineConfig, BaselineTransformer, make_config  # noqa: E402
from train import TrainConfig, train                      # noqa: E402
from fib_attention import make_factories                  # noqa: E402

# --- project paths ---
PROJECT_ROOT = str(_HERE.parent)
DB_PATH = os.path.join(PROJECT_ROOT, "db", "fib.db")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
LDA_DATA_DIR = r"K:\projects\Loop_Dev_AI\data"   # read-only direct path
TRAIN_BIN = "screen_train.bin"
VAL_BINS = {
    "fineweb_edu_val": "val/fineweb_edu_val.bin",
    "wikipedia_val":   "val/wikipedia_val.bin",
    "tinystories_val": "val/tinystories_val.bin",
    "math_val":        "val/math_val.bin",
}

# --- matched schedule ---
COMMON = dict(seed=42, total_steps=13000, lr=3e-4, weight_decay=0.1,
              beta1=0.9, beta2=0.95, grad_clip=1.0, warmup_steps=300,
              eval_interval=500, save_interval=2500, log_interval=50,
              light_eval_tokens=262144, eval_batch_size=8, eval_tokens=1 << 20,
              autocast_dtype="bfloat16")

# --- the matrix: (run_id, fib variant kwargs, batch_size, grad_accum) ---
# learned-alpha: materialized gather, B=4 x accum8.  controls: bool-mask, B=8 x accum4.
MATRIX: dict[str, tuple[dict, int, int]] = {
    "f01__fib_learned_w6":  (dict(offset_base="fib",  alpha_mode="learned", W=6,  K=15, impl="materialized"), 4, 8),
    "f02__fib_learned_w8":  (dict(offset_base="fib",  alpha_mode="learned", W=8,  K=15, impl="materialized"), 4, 8),
    "f03__fib_learned_w10": (dict(offset_base="fib",  alpha_mode="learned", W=10, K=15, impl="materialized"), 4, 8),
    "f04__fib_learned_w12": (dict(offset_base="fib",  alpha_mode="learned", W=12, K=15, impl="materialized"), 4, 8),
    "f05__fib_fixed_w6":    (dict(offset_base="fib",  alpha_mode="fixed",   W=6,  K=15), 8, 4),
    "f06__fib_fixed_w8":    (dict(offset_base="fib",  alpha_mode="fixed",   W=8,  K=15), 8, 4),
    "f07__fib_fixed_w10":   (dict(offset_base="fib",  alpha_mode="fixed",   W=10, K=15), 8, 4),
    "f08__fib_fixed_w12":   (dict(offset_base="fib",  alpha_mode="fixed",   W=12, K=15), 8, 4),
    "f09__logsparse_w8":    (dict(offset_base="pow2", alpha_mode="fixed",   W=8,  K=15), 8, 4),
    # per-layer STAGGERED fixed alpha (tiled 0.5..1.5) -- broad union coverage, bool-mask path
    "f10__fib_stagger_w6":  (dict(offset_base="fib",  alpha_mode="staggered", W=6,  K=15), 8, 4),
    "f11__fib_stagger_w8":  (dict(offset_base="fib",  alpha_mode="staggered", W=8,  K=15), 8, 4),
    "f12__fib_stagger_w10": (dict(offset_base="fib",  alpha_mode="staggered", W=10, K=15), 8, 4),
    "f13__fib_stagger_w12": (dict(offset_base="fib",  alpha_mode="staggered", W=12, K=15), 8, 4),
    # HDC port: same staggered alpha set, coprime-stride layer assignment (anti-gridding)
    "f14__fib_hdc_w6":      (dict(offset_base="fib",  alpha_mode="staggered_hdc", W=6,  K=15), 8, 4),
    "f15__fib_hdc_w8":      (dict(offset_base="fib",  alpha_mode="staggered_hdc", W=8,  K=15), 8, 4),
    "f16__fib_hdc_w10":     (dict(offset_base="fib",  alpha_mode="staggered_hdc", W=10, K=15), 8, 4),
    "f17__fib_hdc_w12":     (dict(offset_base="fib",  alpha_mode="staggered_hdc", W=12, K=15), 8, 4),
    # logsparse (pow2) base WITH the same per-layer linear alpha stagger -- ablation:
    # isolates whether the staggering win is from per-layer spacing diversity or the fib base.
    "f18__logsparse_stagger_w6":  (dict(offset_base="pow2", alpha_mode="staggered", W=6,  K=15), 8, 4),
    "f19__logsparse_stagger_w8":  (dict(offset_base="pow2", alpha_mode="staggered", W=8,  K=15), 8, 4),
    "f20__logsparse_stagger_w10": (dict(offset_base="pow2", alpha_mode="staggered", W=10, K=15), 8, 4),
    "f21__logsparse_stagger_w12": (dict(offset_base="pow2", alpha_mode="staggered", W=12, K=15), 8, 4),
    # SEED=43 replication slice (W=12 only) -- calibrates the seed-noise band and re-tests
    # the smallest load-bearing margins (stagger vs learned vs coprime; fib vs pow2 base).
    # 4th tuple element overrides the default seed; variants match their f0x/f1x/f2x twins.
    "g01__fib_fixed_w12_s43":         (dict(offset_base="fib",  alpha_mode="fixed",        W=12, K=15), 8, 4, 43),
    "g02__fib_learned_w12_s43":       (dict(offset_base="fib",  alpha_mode="learned",      W=12, K=15, impl="materialized"), 4, 8, 43),
    "g03__fib_stagger_w12_s43":       (dict(offset_base="fib",  alpha_mode="staggered",    W=12, K=15), 8, 4, 43),
    "g04__fib_hdc_w12_s43":           (dict(offset_base="fib",  alpha_mode="staggered_hdc", W=12, K=15), 8, 4, 43),
    "g05__logsparse_stagger_w12_s43": (dict(offset_base="pow2", alpha_mode="staggered",    W=12, K=15), 8, 4, 43),
    # SEED=44 third replicate, bool-mask configs only (learned/gather excluded): with seeds
    # 42/43/44 these four get a real 3-seed std on the small "ties"/"base-adds-gain" margins.
    "h01__fib_fixed_w12_s44":         (dict(offset_base="fib",  alpha_mode="fixed",        W=12, K=15), 8, 4, 44),
    "h02__fib_stagger_w12_s44":       (dict(offset_base="fib",  alpha_mode="staggered",    W=12, K=15), 8, 4, 44),
    "h03__fib_hdc_w12_s44":           (dict(offset_base="fib",  alpha_mode="staggered_hdc", W=12, K=15), 8, 4, 44),
    "h04__logsparse_stagger_w12_s44": (dict(offset_base="pow2", alpha_mode="staggered",    W=12, K=15), 8, 4, 44),
}

# variant kwargs persisted to fib_variant.json (impl excluded: rebuilt as materialized for eval)
_VARIANT_KEYS = ("offset_base", "alpha_mode", "alpha_scope", "W", "K")


def build_model(variant: dict, device) -> BaselineTransformer:
    cfg = make_config(512, seq_len=1024)
    cfg.attn_factory = make_factories(**variant)["attn_factory"]
    return BaselineTransformer(cfg).to(device)


def run_one(run_id: str) -> None:
    if run_id not in MATRIX:
        raise SystemExit(f"unknown run_id {run_id!r}; see --list")
    entry = MATRIX[run_id]
    variant, batch_size, grad_accum = entry[0], entry[1], entry[2]
    seed = entry[3] if len(entry) > 3 else COMMON["seed"]
    assert batch_size * grad_accum * 1024 == 32768, "effective batch must match comparators"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(variant, device)

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    run_dir = os.path.join(EXPERIMENTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    # persist the fib variant so eval_posthoc can rebuild the model
    persist = {k: variant.get(k, "per_layer" if k == "alpha_scope" else None) for k in _VARIANT_KEYS}
    with open(os.path.join(run_dir, "fib_variant.json"), "w", encoding="utf-8") as f:
        json.dump(persist, f, indent=2)

    tcfg = TrainConfig(
        run_id=run_id, short_slug=run_id, title=run_id,
        candidate_id=None, variant_or_baseline="variant", phase=2,   # run_id carries identity; no catalog FK
        data_dir=LDA_DATA_DIR, train_bin=TRAIN_BIN, val_bins=VAL_BINS,
        experiments_dir=EXPERIMENTS_DIR, db_path=DB_PATH, log_to_db=True,
        batch_size=batch_size, grad_accum=grad_accum, **{**COMMON, "seed": seed},
    )
    print(f"launching {run_id}: variant={variant} B={batch_size} accum={grad_accum} seed={seed}")
    train(tcfg, model)


def is_run_done(run_id: str) -> bool:
    """True if a 'done' row for this run already exists in fib.db (resumable queue)."""
    if not os.path.exists(DB_PATH):
        return False
    import db as _db
    conn = _db.connect(DB_PATH)
    row = conn.execute("SELECT 1 FROM runs WHERE title=? AND status='done' LIMIT 1",
                       (run_id,)).fetchone()
    conn.close()
    return row is not None


def eval_run(run_id: str, device) -> None:
    """Post-hoc RULER-VT / LEGO / FLOPs on a finished run, written to fib.db."""
    import gc
    import eval_posthoc as EP
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    rd = os.path.join(EXPERIMENTS_DIR, run_id)
    model = EP.build_my_model(rd, device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    EP.write_results(DB_PATH, run_id, rd, EP.run_new_evals(model, device, dtype), is_reference=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--run", help="run id, or 'all' for the full matrix (resumable, auto-evals each)")
    args = ap.parse_args()

    if args.list or not args.run:
        print("matrix:")
        for rid, entry in MATRIX.items():
            v, b, ga = entry[0], entry[1], entry[2]
            sd = entry[3] if len(entry) > 3 else COMMON["seed"]
            print(f"  {rid:26s} {v}  B={b} accum={ga} seed={sd}")
        return
    if args.run == "all":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        done, failed = [], []
        for rid in MATRIX:
            if is_run_done(rid):
                print(f"[queue] skip {rid} (already done)"); continue
            try:
                run_one(rid)
                eval_run(rid, device)
                done.append(rid)
            except Exception as exc:   # keep the queue going past a single failure
                print(f"[queue] FAILED {rid}: {exc!r}"); failed.append(rid)
        print(f"\n[queue] complete. done={done} failed={failed}")
    else:
        run_one(args.run)


if __name__ == "__main__":
    main()
