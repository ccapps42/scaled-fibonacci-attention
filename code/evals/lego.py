"""
lego.py -- LEGO (Learning Equality and Group Operations), single-target scoring.

After Zhang, Backurs et al., "Unveiling Transformers with LEGO" (arXiv:2206.04301).
A chain of variable assignments combined with group operations: x0 is bound to a
group element, each later variable applies an operation to the previous one. The
model must resolve a queried variable by following the chain AND applying the ops.
We use the Z/2 group (two value tokens; ops = identity / flip), which makes the
task a clean composition test scorable as a single next-token prediction.

This exercises both long-range "association" (binding the same variable across the
context) and short-range "manipulation" (applying the op) -- the two head types
LEGO found transformers develop, and exactly the specialization a learned per-layer
Fibonacci spring is meant to induce.

Returns {acc_overall, acc_len<l>..., n}; eval_name = "lego".
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

_LDA_CODE = Path(r"K:\projects\Loop_Dev_AI\code")
if str(_LDA_CODE) not in sys.path:
    sys.path.insert(0, str(_LDA_CODE))
from eval_cheap import _get_tokenizer  # noqa: E402

from ._scoring import choose_among


def lego_eval(model: nn.Module, device, dtype,
              lengths=(2, 3, 4), n_per_len=120, seed=5252) -> dict:
    tok = _get_tokenizer()
    vocab = model.cfg.vocab_size
    var_alph = list(range(1000, min(1000 + 200, vocab)))
    val_tokens = [1300, 1301]                  # two group elements (0, 1)
    op_id, op_flip = 1400, 1401                # identity, flip
    eq  = tok.encode(" =", add_special_tokens=False)
    sep = tok.encode(" ;", add_special_tokens=False)

    rng = random.Random(seed)
    model.eval()
    per_len_correct = {l: 0 for l in lengths}
    per_len_total = {l: 0 for l in lengths}

    for L in lengths:
        for _ in range(n_per_len):
            chain_vars = rng.sample(var_alph, L)
            b = rng.randint(0, 1)              # starting group element
            stmts = [[chain_vars[0]] + eq + [val_tokens[b]]]
            cur = b
            for i in range(1, L):
                op = rng.choice([op_id, op_flip])
                if op == op_flip:
                    cur = 1 - cur
                # statement: "var_i = <op> var_{i-1}"
                stmts.append([chain_vars[i]] + eq + [op, chain_vars[i - 1]])
            resolved = val_tokens[cur]
            rng.shuffle(stmts)                 # scatter (resolution is order-independent)

            seq: list[int] = []
            for s in stmts:
                seq += s + sep
            prefix = seq + [chain_vars[-1]] + eq   # query final var -> resolved element
            if len(prefix) > model.cfg.seq_len:
                continue
            chosen = choose_among(model, prefix, val_tokens, device, dtype)  # 2-way forced choice
            per_len_correct[L] += int(chosen == resolved)
            per_len_total[L] += 1

    out = {}
    tot_c = tot_n = 0
    for L in lengths:
        n = per_len_total[L]
        if n:
            out[f"acc_len{L}"] = round(per_len_correct[L] / n, 4)
            tot_c += per_len_correct[L]; tot_n += n
    out["acc_overall"] = round(tot_c / tot_n, 4) if tot_n else 0.0
    out["n"] = tot_n
    return out
