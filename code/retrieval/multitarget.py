"""
multitarget.py -- staggered-init multi-target shift-copy bench.

Closes (as far as this task allows) the "do per-layer learned springs divide labor"
question. ONE model is trained to copy at a SET of distances DS: each step picks one
d in DS at random and supervises the whole batch to copy from t-d. Forced annealed-blur
(BlurFibAttention) so the scalars can actually move; per-layer STAGGERED init so they
start in different basins (the fix for the depth-4 collapse seen without it).

INTERPRETATION (important): the inputs carry no marker of which distance is active on a
step, and a single output position emits one token, so acc@d1 + acc@d2 is CAPPED near 1.0
by construction -- "both ~1" is unreachable. Read the result as:
  sum ~1 with BOTH clearly > 0  -> model reaches both distances (success / no collapse)
  sum ~0 (both ~0)              -> collapse (what depth-4 did WITHOUT staggered init)
Prior baselines (no staggered init): depth-2 acc 0.66/0.44 (sum 1.10, reaches both);
depth-4 acc 0.005/0.009 (collapse). This run re-tests with staggered init across depths.

  python code/retrieval/multitarget.py --depths 2 4 8 --dists 200 350
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
from blur_fib import BlurFibAttention                       # noqa: E402

OUT = os.path.join(str(_CODE.parent), "experiments", "retrieval")
T, V, VOCAB, B = 512, 256, 512, 2   # T=512/B=2: blur fits depth-8 with checkpointing
BLUR_W0 = 40.0


def build(depth, staggered):
    cfg = BaselineConfig(d_model=256, n_layers=depth, n_heads=4, n_kv_heads=2,
                         vocab_size=VOCAB, seq_len=T, dropout=0.0)
    L = depth
    def af(c, li, L=L):
        m = BlurFibAttention(c, li, offset_base="fib", W=8, K=15, w0=BLUR_W0)
        if staggered:
            a = min(1.45, max(0.55, 0.5 + 1.0 * (li / (L - 1)) if L > 1 else 1.0))
            m.theta.data.fill_(math.log((a - 0.5) / (1.5 - a)))
        return m
    cfg.attn_factory = af
    return BaselineTransformer(cfg)


def _mods(model):
    return [m for m in model.modules() if isinstance(m, BlurFibAttention)]


def _alphas(model):
    return [round(float(m.alpha().detach()), 3) for m in _mods(model)]


def run_cell(depth, dists, seed, steps, lr, staggered, device):
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = build(depth, staggered).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for s in range(steps):
        frac = min(1.0, s / (0.8 * steps))
        w = BLUR_W0 * (1 - frac) + 0.5 * frac
        for m in _mods(model):
            m._w = w
        d = rng.choice(dists)
        x = torch.randint(0, V, (B, T), device=device)
        tgt = torch.full((B, T), -100, device=device); tgt[:, d:] = x[:, :T - d]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB).float(), tgt.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    model.eval()

    def _ev(d):
        with torch.no_grad():
            x = torch.randint(0, V, (B, T), device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
            return round((logits[:, d:, :].argmax(-1) == x[:, :T - d]).float().mean().item(), 3)

    accs = {d: _ev(d) for d in dists}
    controls = {cd: _ev(cd) for cd in (233, 300)}
    pk = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    alphas = _alphas(model)
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"depth": depth, "dists": dists, "seed": seed, "staggered": staggered,
            "accs": accs, "acc_sum": round(sum(accs.values()), 3),
            "controls": controls, "alphas": alphas, "peakGB": pk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--dists", type=int, nargs="+", default=[200, 350])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-stagger", action="store_true", help="reproduce the collapse baseline")
    ap.add_argument("--out", default=os.path.join(OUT, "multitarget_stagger.json"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)
    staggered = not args.no_stagger

    done = {}
    if os.path.exists(args.out):
        for r in json.load(open(args.out)):
            done[(r["depth"], r["seed"], r["staggered"])] = r
    results = list(done.values())

    cells = [(dp, s) for dp in args.depths for s in args.seeds if (dp, s, staggered) not in done]
    print(f"multitarget: {len(cells)} cells (stagger={staggered}, dists={args.dists}, {len(done)} done)")
    t0 = time.time()
    for i, (dp, s) in enumerate(cells):
        try:
            r = run_cell(dp, args.dists, s, args.steps, args.lr, staggered, device)
            results.append(r)
            json.dump(results, open(args.out, "w"), indent=1)
            el = time.time() - t0
            print(f"[{i+1}/{len(cells)}] depth={dp} seed={s}: accs={r['accs']} sum={r['acc_sum']} "
                  f"alphas={r['alphas']} pk={r['peakGB']} ({el/(i+1):.0f}s/cell)")
        except Exception as exc:
            print(f"[{i+1}/{len(cells)}] depth={dp} seed={s}: FAILED {exc!r}")
    print(f"done: {len(results)} results -> {args.out}  ({time.time()-t0:.0f}s)")
    print("read: acc_sum~1 with both>0 = reaches both distances; both~0 = collapse")


if __name__ == "__main__":
    main()
