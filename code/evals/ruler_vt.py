"""
ruler_vt.py -- RULER Variable Tracking, single-target scoring variant.

Synthetic, knowledge-free multi-hop tracing (Hsieh et al., RULER, COLM 2024,
arXiv:2404.06654). A value is bound to a variable, then re-bound through a chain
of intermediate variables; assignment statements are scattered among distractor
bindings; the model must resolve the final variable to the original value.

We score the single-target variant (LAMBADA-style): present all statements, then
"<final_var> =" and check whether the model's top-1 next token is the original
value. Knobs: hop count (chain length) and number of distractor statements
(controls how far apart the chain links sit). NOT the verbatim RULER extraction
metric -- labeled accordingly.

Returns {acc_overall, acc_h<h>..., n} via ruler_vt_eval(); eval_name = "ruler_vt".
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


def ruler_vt_eval(model: nn.Module, device, dtype,
                  hops=(2, 3, 4), n_per_hop=120, n_distract=8, seed=4242) -> dict:
    tok = _get_tokenizer()
    vocab = model.cfg.vocab_size
    # disjoint safe token alphabets for variable names and values
    var_alph = list(range(1000, min(1000 + 200, vocab)))
    val_alph = list(range(1300, min(1300 + 200, vocab)))
    eq  = tok.encode(" =", add_special_tokens=False)
    sep = tok.encode(" ;", add_special_tokens=False)

    rng = random.Random(seed)
    model.eval()
    per_hop_correct = {h: 0 for h in hops}
    per_hop_total = {h: 0 for h in hops}

    for h in hops:
        for _ in range(n_per_hop):
            v = rng.choice(val_alph)
            chain_vars = rng.sample(var_alph, h)
            # chain statements: var0 = v ; var_i = var_{i-1}
            stmts = [[chain_vars[0]] + eq + [v]]
            for i in range(1, h):
                stmts.append([chain_vars[i]] + eq + [chain_vars[i - 1]])
            # distractor statements: independent var = val using disjoint tokens
            used = set(chain_vars) | {v}
            distractor_vals = []
            for _ in range(n_distract):
                dv = rng.choice([t for t in var_alph if t not in used]); used.add(dv)
                dval = rng.choice([t for t in val_alph if t not in used]); used.add(dval)
                distractor_vals.append(dval)
                stmts.append([dv] + eq + [dval])
            rng.shuffle(stmts)  # scatter chain links among distractors -> distance

            seq: list[int] = []
            for s in stmts:
                seq += s + sep
            # query: "<final_var> =" -> choose the value bound to it, among present values
            prefix = seq + [chain_vars[-1]] + eq
            if len(prefix) > model.cfg.seq_len:
                continue
            chosen = choose_among(model, prefix, [v] + distractor_vals, device, dtype)
            per_hop_correct[h] += int(chosen == v)
            per_hop_total[h] += 1

    out = {}
    tot_c = tot_n = 0
    for h in hops:
        n = per_hop_total[h]
        if n:
            out[f"acc_h{h}"] = round(per_hop_correct[h] / n, 4)
            tot_c += per_hop_correct[h]; tot_n += n
    out["acc_overall"] = round(tot_c / tot_n, 4) if tot_n else 0.0
    out["n"] = tot_n
    return out
