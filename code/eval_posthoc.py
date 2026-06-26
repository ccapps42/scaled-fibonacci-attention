"""
eval_posthoc.py -- run the NEW evals (RULER-VT, LEGO, FLOPs-context) on a final
checkpoint and write to THIS project's DB.

Two sources:
  --run f02__fib_learned_w8      one of my runs (rebuilt from fib_variant.json)
  --comparator e0000|e0030|e0084 a Loop_Dev_AI run (rebuilt via its load_checkpoint),
                                  results stored under a reference row in fib.db

Loop_Dev_AI is read-only; all eval rows land in this project's db/fib.db.

  python code/eval_posthoc.py --run f02__fib_learned_w8 --device cuda
  python code/eval_posthoc.py --comparator e0000 --device cuda
  python code/eval_posthoc.py --run f02__fib_learned_w8 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import torch                                              # noqa: E402

PROJECT_ROOT = str(_HERE.parent)
DB_PATH = os.path.join(PROJECT_ROOT, "db", "fib.db")
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")
FINAL_STEP = 13000

COMPARATORS = {
    "e0000": r"K:\projects\Loop_Dev_AI\experiments\e0000__dense_baseline_d512",
    "e0030": r"K:\projects\Loop_Dev_AI\experiments\e0030__stride_early",
    "e0084": r"K:\projects\Loop_Dev_AI\experiments\e0084__nsa_all16",
}

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def build_my_model(run_dir: str, device):
    from baseline import BaselineConfig, BaselineTransformer
    from fib_attention import make_factories
    variant = json.load(open(os.path.join(run_dir, "fib_variant.json"), encoding="utf-8"))
    model_cfg = json.load(open(os.path.join(run_dir, "config.json"), encoding="utf-8"))["model"]
    variant = {**variant, "impl": "materialized"}  # eval path; output-identical to stream
    bcfg = BaselineConfig(
        d_model=model_cfg["d_model"], n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"], n_kv_heads=model_cfg["n_kv_heads"],
        vocab_size=model_cfg.get("vocab_size", 32768),
        seq_len=model_cfg.get("seq_len", 1024),
        attn_factory=make_factories(**variant)["attn_factory"],
    )
    model = BaselineTransformer(bcfg)
    ckpt = torch.load(os.path.join(run_dir, "ckpt_final.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def build_comparator_model(run_dir: str, device):
    from eval_cheap import load_checkpoint   # LDA loader: rebuilds dense/stride/nsa via load_novelty
    model, _ = load_checkpoint(os.path.join(run_dir, "ckpt_final.pt"), device)
    return model.eval()


def run_new_evals(model, device, dtype) -> dict:
    from evals import ruler_vt_eval, lego_eval, flops_context_eval
    out = {}
    print("  [ruler_vt] ...", flush=True); out["ruler_vt"] = ruler_vt_eval(model, device, dtype); print("   ", out["ruler_vt"])
    print("  [lego] ...", flush=True);     out["lego"] = lego_eval(model, device, dtype);         print("   ", out["lego"])
    print("  [flops_context] ...", flush=True); out["flops_context"] = flops_context_eval(model, device, dtype); print("   ", out["flops_context"])
    return out


def _find_or_make_run_row(conn, title, run_dir, is_reference):
    import db as _db
    basename = Path(run_dir).name
    row = conn.execute("SELECT id FROM runs WHERE run_dir LIKE ? ORDER BY id DESC LIMIT 1",
                       (f"%{basename}",)).fetchone()
    if row is not None:
        return row["id"]
    if not is_reference:
        raise SystemExit(f"no run row for {basename} in fib.db (was it trained via run.py?)")
    # reference comparator row
    return _db.start_run(conn, title=title, run_dir=run_dir, candidate_id=None,
                         variant_or_baseline="reference", phase=2, config_json="{}",
                         seed=42, d_model=512, n_layers=16, params_effective=0)


def write_results(db_path, title, run_dir, results, is_reference):
    import db as _db
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    _db.init_db(db_path)
    conn = _db.connect(db_path)
    run_id = _find_or_make_run_row(conn, title, run_dir, is_reference)
    _db.add_eval_results(conn, run_id, FINAL_STEP, 512, results)
    conn.close()
    print(f"  [wrote {list(results)} to {db_path} run_id={run_id}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="my run id, e.g. f02__fib_learned_w8")
    ap.add_argument("--comparator", choices=list(COMPARATORS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=list(DTYPES))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    dtype = DTYPES[args.dtype]

    if args.comparator:
        run_dir = COMPARATORS[args.comparator]
        title = f"ref_{args.comparator}"
        model = build_comparator_model(run_dir, device)
        is_ref = True
    elif args.run:
        run_dir = os.path.join(EXPERIMENTS_DIR, args.run)
        title = args.run
        model = build_my_model(run_dir, device)
        is_ref = False
    else:
        raise SystemExit("pass --run or --comparator")

    print(f"=== eval_posthoc: {title} ({run_dir}) device={device} dtype={dtype} ===")
    results = run_new_evals(model, device, dtype)
    if args.dry_run:
        print("  [dry-run: not writing to DB]")
    else:
        write_results(DB_PATH, title, run_dir, results, is_ref)


if __name__ == "__main__":
    main()
