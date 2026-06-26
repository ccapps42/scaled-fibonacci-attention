"""
task.py -- retrieval-at-distance (single-hop induction copy) generator.

Each sequence: random filler, with a target (K*, V*) bigram planted so that V* sits
at a CONTROLLED offset d from the query, plus n_distract distractor (K_i, V_i)
bigrams elsewhere. The query is K* at the final position; the model must copy V*.
Distractors force genuine distance-dependent retrieval (it can't just "find the only
value token"). Loss/accuracy is read only at the query position.

Token ranges (disjoint, within vocab 32768):
  filler [0, 20000)   keys [20000, 24000)   values [24000, 28000)
"""
from __future__ import annotations

import torch

FILLER = (0, 20000)
KEYS = (20000, 24000)
VALS = (24000, 28000)


def make_batch(B, T, d, n_distract, rng, device):
    """Return x (B,T) long, target (B,) long, present_vals (B, 1+n_distract) long.

    d = offset from the query (position T-1) back to V*. All examples in the batch
    share the same d. present_vals lists the value tokens in the sequence (target
    first) for forced-choice scoring (chance = 1/(1+n_distract)).
    """
    x = torch.randint(FILLER[0], FILLER[1], (B, T), dtype=torch.long)
    target = torch.empty(B, dtype=torch.long)
    present = torch.empty(B, 1 + n_distract, dtype=torch.long)

    tgt_start = T - 2 - d            # K* at tgt_start, V* at tgt_start+1 (offset d from query)
    assert 0 <= tgt_start and tgt_start + 1 < T - 1, f"d={d} does not fit in T={T}"

    for i in range(B):
        keys = _distinct(rng, KEYS, 1 + n_distract)
        vals = _distinct(rng, VALS, 1 + n_distract)
        kstar, vstar = keys[0], vals[0]

        occupied = set()
        x[i, tgt_start] = kstar
        x[i, tgt_start + 1] = vstar
        occupied.update((tgt_start, tgt_start + 1, T - 1))

        # distractor bigrams at random non-overlapping starts (not the query slot)
        for j in range(n_distract):
            while True:
                s = rng.randint(0, T - 3)
                if s not in occupied and s + 1 not in occupied and s != T - 1 and s + 1 != T - 1:
                    break
            x[i, s] = keys[1 + j]
            x[i, s + 1] = vals[1 + j]
            occupied.update((s, s + 1))

        x[i, T - 1] = kstar          # query
        target[i] = vstar
        present[i, 0] = vstar
        for j in range(n_distract):
            present[i, 1 + j] = vals[1 + j]

    return x.to(device), target.to(device), present.to(device)


def _distinct(rng, rng_range, n):
    lo, hi = rng_range
    out: list[int] = []
    seen: set[int] = set()
    while len(out) < n:
        t = rng.randint(lo, hi - 1)
        if t not in seen:
            seen.add(t); out.append(t)
    return out


# --- distance samplers (return a single d per training step) ---

def uniform_sampler(lo, hi):
    return lambda rng: rng.randint(lo, hi)


def band_sampler(center, width):
    lo, hi = center - width // 2, center + width // 2
    return lambda rng: rng.randint(lo, hi)


def multiband_sampler(bands):  # bands: list of (center, width)
    def s(rng):
        c, w = bands[rng.randint(0, len(bands) - 1)]
        return rng.randint(c - w // 2, c + w // 2)
    return s
