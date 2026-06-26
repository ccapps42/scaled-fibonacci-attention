"""
marked.py -- MARKED multi-target shift-copy: the true union-coverage test.

Unlike multitarget.py (where the active distance is unmarked, so acc@d1+acc@d2 is
capped near 1 by construction), here every position carries a per-position marker
telling it WHICH distance to copy from. So different output positions demand different
distances within the SAME forward, and the model must reach BOTH at once. acc@d1 and
acc@d2 can each approach 1 independently -- a clean test of whether per-layer learned
springs divide labor to give union coverage.

Encoding (no change to the read-only baseline -- ids only):
  payload p_t ~ U[0, VP);  marker c_t ~ U{0..len(DS)-1};  input id  x_t = p_t + c_t*VP
  query at t wants distance d = DS[c_t];  target_t = x_{t-d}  (copy the full source id)
  VP = VOCAB // len(DS); with VOCAB=512, len(DS)=2 -> VP=256, ids in [0,512).

Read the result:
  both acc@d1 and acc@d2 ~1  -> union coverage achieved (the per-layer springs cover both)
  one ~1, other ~0           -> model collapsed to a single distance (no labor division)
  both ~0                    -> failed to copy at all

  python code/retrieval/marked.py --depths 2 4 8 --dists 200 350 [--no-stagger]
"""
from __future__ import annotations

import argparse
import json
import math
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

from baseline import BaselineConfig, BaselineTransformer    # noqa: E402
from blur_fib import BlurFibAttention                       # noqa: E402

OUT = os.path.join(str(_CODE.parent), "experiments", "retrieval")
T, VOCAB, B = 512, 512, 2
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


def gen_batch(dists, device, gen):
    """Return (x, marker) with x_t = payload + marker*VP, marker in {0..len(dists)-1}."""
    vp = VOCAB // len(dists)
    payload = torch.randint(0, vp, (B, T), device=device, generator=gen)
    marker = torch.randint(0, len(dists), (B, T), device=device, generator=gen)
    x = payload + marker * vp
    return x, marker


def make_target(x, marker, dists):
    """target_t = x_{t-d}, d = dists[marker_t]; -100 where t<d. (per-position distance)"""
    pos = torch.arange(T, device=x.device)
    tgt = torch.full((B, T), -100, device=x.device, dtype=torch.long)
    for j, d in enumerate(dists):
        src = x.index_select(1, (pos - d).clamp(min=0))   # x shifted back by d  (B,T)
        sel = (marker == j) & (pos.unsqueeze(0) >= d)      # this class, valid range
        tgt = torch.where(sel, src, tgt)
    return tgt


def run_cell(depth, dists, seed, steps, lr, staggered, device):
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    model = build(depth, staggered).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for s in range(steps):
        frac = min(1.0, s / (0.8 * steps))
        w = BLUR_W0 * (1 - frac) + 0.5 * frac
        for m in _mods(model):
            m._w = w
        x, marker = gen_batch(dists, device, gen)
        tgt = make_target(x, marker, dists)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB).float(), tgt.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    model.eval()

    @torch.no_grad()
    def _eval():
        pos = torch.arange(T, device=device)
        x, marker = gen_batch(dists, device, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)
        pred = logits.argmax(-1)
        out = {}
        for j, d in enumerate(dists):
            src = x.index_select(1, (pos - d).clamp(min=0))
            sel = (marker == j) & (pos.unsqueeze(0) >= d)
            out[d] = round(((pred == src) & sel).sum().float().div(sel.sum().clamp(min=1)).item(), 3)
        return out

    accs = _eval()
    pk = round(torch.cuda.max_memory_allocated() / 1e9, 1)
    alphas = _alphas(model)
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"depth": depth, "dists": dists, "seed": seed, "staggered": staggered,
            "accs": accs, "min_acc": round(min(accs.values()), 3),
            "alphas": alphas, "peakGB": pk}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--dists", type=int, nargs="+", default=[200, 350])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-stagger", action="store_true")
    ap.add_argument("--out", default=os.path.join(OUT, "marked_union.json"))
    args = ap.parse_args()
    if VOCAB % len(args.dists) != 0:
        raise SystemExit(f"VOCAB={VOCAB} must divide evenly by len(dists)={len(args.dists)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT, exist_ok=True)
    staggered = not args.no_stagger

    done = {}
    if os.path.exists(args.out):
        for r in json.load(open(args.out)):
            done[(r["depth"], r["seed"], r["staggered"])] = r
    results = list(done.values())

    cells = [(dp, s) for dp in args.depths for s in args.seeds if (dp, s, staggered) not in done]
    print(f"marked-union: {len(cells)} cells (stagger={staggered}, dists={args.dists}, {len(done)} done)")
    t0 = time.time()
    for i, (dp, s) in enumerate(cells):
        try:
            r = run_cell(dp, args.dists, s, args.steps, args.lr, staggered, device)
            results.append(r)
            json.dump(results, open(args.out, "w"), indent=1)
            el = time.time() - t0
            print(f"[{i+1}/{len(cells)}] depth={dp} seed={s}: accs={r['accs']} min={r['min_acc']} "
                  f"alphas={r['alphas']} pk={r['peakGB']} ({el/(i+1):.0f}s/cell)")
        except Exception as exc:
            print(f"[{i+1}/{len(cells)}] depth={dp} seed={s}: FAILED {exc!r}")
    print(f"done: {len(results)} results -> {args.out}  ({time.time()-t0:.0f}s)")
    print("read: BOTH accs ~1 = union coverage; one~1/other~0 = single-distance collapse")


if __name__ == "__main__":
    main()
