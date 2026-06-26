"""
sweep.py -- overnight shift-copy coverage/adaptivity sweep.

For each (mechanism, depth, distance, seed): train a small model on shift-copy
(target[t]=input[t-d], dense supervision), evaluate hard-sparse copy accuracy at d,
record accuracy + the learned spacing report. Resumable (skips cells already in the
output json), continue-on-error, conservative batch.

Mechanisms:
  dense       full attention (coverage ceiling)
  logsparse   pow2 fixed bool-mask        (fixed geometric baseline)
  fib_fixed   Fibonacci fixed bool-mask   (fixed Fibonacci baseline)
  gumbel_fib  learned alpha, Gumbel-superset hard-select (the original design)
  fib_blur_stag  learned alpha + forced annealed-blur + STAGGERED init (the only
                 mechanism whose alpha actually moves; B=2, gradient-checkpointed)
  sw_fib/sw_geom/sw_free  soft-width learned (scale / base / free-DCLS)  [escapable - context only]

  python code/retrieval/sweep.py --mechs dense logsparse fib_fixed gumbel_fib \
      --depths 2 4 8 --seeds 0 1 --steps 3000
"""
from __future__ import annotations

import argparse
import json
import os
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

import math                                                  # noqa: E402
from baseline import BaselineConfig, BaselineTransformer    # noqa: E402
from fib_attention import make_factories, FibBoolMaskAttention, FibonacciAttention  # noqa: E402
from gumbel_fib import GumbelFibAttention, make_gumbel_factory      # noqa: E402
from spacing import SoftWidthSpacingAttention, make_softwidth_factory  # noqa: E402
from blur_fib import BlurFibAttention                              # noqa: E402

BLUR_W0 = 40.0   # initial Gaussian band width for fib_blur_stag (annealed -> 0.5)

OUT = os.path.join(str(_CODE.parent), "experiments", "retrieval")
T, V, VOCAB, B = 1024, 256, 512, 8   # conservative batch so depth-8 cells stay well under VRAM cliff
# fib rungs (where fixed-Fibonacci reaches) interleaved with the midpoint gaps between
# consecutive rungs (the holes) -- samples actual reachable offsets, not arbitrary points.
# rungs: 55 89 144 233 377 610   gaps: 72 116 188 305 494 798
DISTANCES = [55, 72, 89, 116, 144, 188, 233, 305, 377, 494, 610, 798]
# fixed control set evaluated on EVERY trained model: fib rungs (233,377), pow2 rungs (128,512), gaps (350,700)
CONTROL_D = [128, 233, 350, 377, 512, 700]


def build(mech, depth):
    cfg = BaselineConfig(d_model=256, n_layers=depth, n_heads=4, n_kv_heads=2,
                         vocab_size=VOCAB, seq_len=T, dropout=0.0)
    if mech == "dense":
        pass
    elif mech == "logsparse":
        cfg.attn_factory = make_factories(offset_base="pow2", alpha_mode="fixed", W=8, K=15)["attn_factory"]
    elif mech == "fib_fixed":
        cfg.attn_factory = make_factories(offset_base="fib", alpha_mode="fixed", W=8, K=15)["attn_factory"]
    elif mech == "fib_spread":
        # per-layer FIXED alpha staggered across [0.5,1.5] -> layers collectively tile distances
        L = depth
        def af(c, li, L=L):
            a = 0.5 + 1.0 * (li / (L - 1)) if L > 1 else 1.0
            return FibBoolMaskAttention(c, li, offset_base="fib", W=8, K=15, alpha=a)
        cfg.attn_factory = af
    elif mech == "fib_stag_learn":
        # LEARNABLE alpha, STAGGERED init -> does learning bridge gaps the frozen stagger misses?
        L = depth
        def af(c, li, L=L):
            m = FibonacciAttention(c, li, offset_base="fib", alpha_mode="learned",
                                   impl="materialized", W=8, K=15)
            a = min(1.45, max(0.55, 0.5 + 1.0 * (li / (L - 1)) if L > 1 else 1.0))
            m.theta.data.fill_(math.log((a - 0.5) / (1.5 - a)))   # init alpha = staggered value
            return m
        cfg.attn_factory = af
    elif mech == "fib_blur_stag":
        # LEARNED alpha + forced annealed-blur (the only mechanism whose alpha actually
        # moves) + STAGGERED init so the per-layer springs start in different basins and
        # can divide labor instead of collapsing to alpha~1 (which depth-4 did otherwise).
        L = depth
        def af(c, li, L=L):
            m = BlurFibAttention(c, li, offset_base="fib", W=8, K=15, w0=BLUR_W0)
            a = min(1.45, max(0.55, 0.5 + 1.0 * (li / (L - 1)) if L > 1 else 1.0))
            m.theta.data.fill_(math.log((a - 0.5) / (1.5 - a)))   # init alpha = staggered value
            return m
        cfg.attn_factory = af
    elif mech == "gumbel_fib":
        cfg.attn_factory = make_gumbel_factory(W=8, K=15, superset_n=13, tau_init=2.0)["attn_factory"]
    elif mech.startswith("sw_"):
        mode = {"sw_fib": "fib_scale", "sw_geom": "geom_base", "sw_free": "free"}[mech]
        cfg.attn_factory = make_softwidth_factory(mode=mode, W=8, K=15, w_init=60.0)["attn_factory"]
    else:
        raise ValueError(mech)
    return BaselineTransformer(cfg)


def _sched(model, mech, frac):
    """Set annealing schedule (tau for gumbel, width for soft-width) by training fraction."""
    if mech == "gumbel_fib":
        tau = 2.0 * (1 - min(1.0, frac / 0.8)) + 0.1 * min(1.0, frac / 0.8)
        for m in model.modules():
            if isinstance(m, GumbelFibAttention):
                m._tau = tau
    elif mech.startswith("sw_"):
        w = 60.0 * (1 - min(1.0, frac / 0.8)) + 0.4 * min(1.0, frac / 0.8)
        for m in model.modules():
            if isinstance(m, SoftWidthSpacingAttention):
                m._w = w
    elif mech == "fib_blur_stag":
        w = BLUR_W0 * (1 - min(1.0, frac / 0.8)) + 0.5 * min(1.0, frac / 0.8)
        for m in model.modules():
            if isinstance(m, BlurFibAttention):
                m._w = w


def _report(model, mech):
    rep = []
    for m in model.modules():
        if isinstance(m, (GumbelFibAttention, SoftWidthSpacingAttention, BlurFibAttention)):
            rep.append(m.report())
        elif isinstance(m, FibonacciAttention):
            rep.append({"alpha": round(float(m.alpha()), 4)})
    return rep


def _batch(mech):
    # forced-blur retains a (B,H,T,M,hd) band per rung -> needs a reduced batch to fit T=1024
    return 2 if mech == "fib_blur_stag" else B


def run_cell(mech, depth, d, seed, steps, lr, device):
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(seed)
    bsz = _batch(mech)
    model = build(mech, depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    amp = device.type == "cuda"
    for s in range(steps):
        _sched(model, mech, s / steps)
        x = torch.randint(0, V, (bsz, T), device=device)
        tgt = torch.full((bsz, T), -100, device=device); tgt[:, d:] = x[:, :T - d]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB).float(), tgt.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()   # hard-sparse for gumbel/fixed; soft-width stays soft (escapable - flagged)

    def _eval_at(dd):
        with torch.no_grad():
            x = torch.randint(0, V, (bsz, T), device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                logits, _ = model(x)
            return round((logits[:, dd:, :].argmax(-1) == x[:, :T - dd]).float().mean().item(), 4)

    acc = _eval_at(d)                                   # accuracy at the trained distance
    acc_control = {cd: _eval_at(cd) for cd in CONTROL_D}  # coverage at fixed rung/gap controls
    rep = _report(model, mech)
    pk = torch.cuda.max_memory_allocated() / 1e9
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"mech": mech, "depth": depth, "d": d, "seed": seed,
            "acc": acc, "acc_control": acc_control, "report": rep, "peakGB": round(pk, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechs", nargs="+", default=["dense", "logsparse", "fib_fixed", "gumbel_fib"])
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--distances", type=int, nargs="+", default=DISTANCES)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=os.path.join(OUT, "coverage_sweep.json"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)

    done = {}
    if os.path.exists(args.out):
        for r in json.load(open(args.out)):
            done[(r["mech"], r["depth"], r["d"], r["seed"])] = r
    results = list(done.values())

    cells = [(m, dp, d, s) for m in args.mechs for dp in args.depths
             for d in args.distances for s in args.seeds]
    # fixed mechanisms are seed-invariant in coverage -> only seed 0
    cells = [c for c in cells if not (c[0] in ("dense", "logsparse", "fib_fixed", "fib_spread") and c[3] != 0)]
    todo = [c for c in cells if c not in done]
    print(f"sweep: {len(todo)} cells to run ({len(done)} already done), {len(cells)} total")
    t0 = time.time()
    for i, (m, dp, d, s) in enumerate(todo):
        try:
            tc = time.time()
            r = run_cell(m, dp, d, s, args.steps, args.lr, device)
            dt = time.time() - tc                       # THIS cell's wall time (not cumulative avg)
            eta = (len(todo) - i - 1) * dt / 60          # ETA from the latest cell's rate
            results.append(r)
            json.dump(results, open(args.out, "w"), indent=1)   # incremental save
            print(f"[{i+1}/{len(todo)}] {m} depth={dp} d={d} seed={s}: "
                  f"acc={r['acc']:.3f} {r['report']} pk={r['peakGB']} "
                  f"({dt:.0f}s this cell, ETA {eta:.0f} min)")
        except Exception as exc:
            print(f"[{i+1}/{len(todo)}] {m} depth={dp} d={d} seed={s}: FAILED {exc!r}")
    print(f"sweep complete: {len(results)} results -> {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
