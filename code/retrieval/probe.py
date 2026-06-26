"""
probe.py -- train small models on retrieval-at-distance and measure accuracy(d)
+ the learned per-layer alpha, across a depth ladder and four attention conditions.

Conditions: dense | logsparse (pow2) | fib_fixed (alpha=1) | fib_learned.
Sub-experiments:
  coverage   : train d ~ uniform[lo,hi]; eval accuracy(d) over a sweep.
  band       : train d ~ band(center,width); eval band + controls; read alpha movement.
  multiband  : train d ~ several bands; eval per band; read per-layer alpha spread.

Self-contained; writes results JSON to experiments/retrieval/. Does not touch the
perplexity matrix. Reuses the Loop_Dev_AI baseline + this project's fib_attention.

  python code/retrieval/probe.py --smoke
  python code/retrieval/probe.py --plan coverage --depths 2 4 8 --steps 4000
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).parent
_CODE = _HERE.parent
_LDA = Path(r"K:\projects\Loop_Dev_AI\code")
for p in (str(_HERE), str(_CODE), str(_LDA)):
    if p not in sys.path:
        sys.path.insert(0, p)

from baseline import BaselineConfig, BaselineTransformer    # noqa: E402
from fib_attention import make_factories                    # noqa: E402
import task as TK                                           # noqa: E402

OUT_DIR = os.path.join(str(_CODE.parent), "experiments", "retrieval")
CONDITIONS = ("dense", "logsparse", "fib_fixed", "fib_learned")


def build(cond, depth, d_model=256, seq_len=1024, alpha_min=0.5, alpha_span=1.0):
    cfg = BaselineConfig(d_model=d_model, n_layers=depth, n_heads=4, n_kv_heads=2,
                         vocab_size=32768, seq_len=seq_len, dropout=0.0)
    fib = dict(W=8, K=15, alpha_min=alpha_min, alpha_span=alpha_span)
    if cond == "dense":
        pass
    elif cond == "logsparse":
        cfg.attn_factory = make_factories(offset_base="pow2", alpha_mode="fixed", W=8, K=15)["attn_factory"]
    elif cond == "fib_fixed":
        cfg.attn_factory = make_factories(offset_base="fib", alpha_mode="fixed", W=8, K=15)["attn_factory"]
    elif cond == "fib_learned":
        cfg.attn_factory = make_factories(offset_base="fib", alpha_mode="learned",
                                          impl="materialized", **fib)["attn_factory"]
    else:
        raise ValueError(cond)
    return BaselineTransformer(cfg)


def train(model, d_sampler, steps, B, T, n_distract, lr, device, rng):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    use_amp = device.type == "cuda"
    for s in range(steps):
        d = d_sampler(rng)
        x, tgt, _ = TK.make_batch(B, T, d, n_distract, rng, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits, _ = model(x)
        loss = F.cross_entropy(logits[:, -1, :].float(), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return float(loss.item())


@torch.no_grad()
def acc_curve(model, distances, n_per_d, B, T, n_distract, device, rng):
    model.eval()
    use_amp = device.type == "cuda"
    out = {}
    for d in distances:
        correct = total = 0
        reps = max(1, n_per_d // B)
        for _ in range(reps):
            x, tgt, pv = TK.make_batch(B, T, d, n_distract, rng, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits, _ = model(x)
            ql = logits[:, -1, :].float()
            pick = ql.gather(1, pv).argmax(1)            # forced choice among present values
            chosen = pv[torch.arange(pv.shape[0], device=device), pick]
            correct += (chosen == tgt).sum().item(); total += pv.shape[0]
        out[int(d)] = round(correct / total, 4)
    return out


def read_alphas(model):
    out = []
    for k, v in model.state_dict().items():
        if k.endswith("theta"):
            out.append(round(0.5 + 1.0 / (1.0 + math.exp(-float(v.reshape(-1)[0]))), 4))
    return out


def run_cell(cond, depth, sampler, eval_distances, steps, B, T, n_distract, lr, device, seed):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model = build(cond, depth).to(device)
    t0 = time.time()
    final_loss = train(model, sampler, steps, B, T, n_distract, lr, device, rng)
    curve = acc_curve(model, eval_distances, n_per_d=4 * B, B=B, T=T,
                      n_distract=n_distract, device=device, rng=rng)
    res = {"cond": cond, "depth": depth, "final_loss": round(final_loss, 4),
           "alphas": read_alphas(model), "curve": curve, "secs": round(time.time() - t0, 1)}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plan", choices=["coverage", "band", "multiband"], default="coverage")
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--conds", nargs="+", default=list(CONDITIONS))
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-distract", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    T = 1024
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.smoke:
        # dense, depth 2, short distances (dense covers them) -> accuracy should beat chance
        d = run_cell("dense", 2, TK.uniform_sampler(16, 200), [32, 64, 128],
                     steps=1500, B=32, T=T, n_distract=4, lr=3e-4, device=device, seed=0)
        print("SMOKE dense depth2:", d["curve"], "loss", d["final_loss"],
              "(chance 0.20; expect >0.6 at covered distances)")
        return

    if args.plan == "coverage":
        sampler = TK.uniform_sampler(16, 900)
        eval_d = list(range(40, 901, 40))
    elif args.plan == "band":
        sampler = TK.band_sampler(798, 50)
        eval_d = [120, 250, 400, 610, 700, 798, 850, 987 - 50] + [770, 798, 820]
        eval_d = sorted(set(eval_d))
    else:  # multiband
        sampler = TK.multiband_sampler([(300, 40), (600, 40), (800, 40)])
        eval_d = [120, 300, 450, 600, 720, 800, 880]

    results = []
    for depth in args.depths:
        for cond in args.conds:
            r = run_cell(cond, depth, sampler, eval_d, args.steps, args.batch,
                         T, args.n_distract, args.lr, device, args.seed)
            print(f"[{args.plan}] depth={depth} {cond:12s} loss={r['final_loss']:.3f} "
                  f"alphas={r['alphas']} secs={r['secs']}")
            print("    curve:", r["curve"])
            results.append(r)
    path = os.path.join(OUT_DIR, f"{args.plan}.json")
    json.dump({"plan": args.plan, "n_distract": args.n_distract, "steps": args.steps,
               "results": results}, open(path, "w"), indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
