"""
eval_cheap.py — Block A cheap eval suite for Loop_Dev_AI.

Five eval functions + eval_all_cheap aggregator + checkpoint loader + CLI.

All evals run in eval-only mode (no generation, no probe training).
They return dicts of {metric_name: float} usable with add_eval_results().

Dataset status
--------------
LAMBADA  : REAL — EleutherAI/lambada_openai, cached in HF_HOME on demand.
BLiMP    : REAL — nyu-mll/blimp (8 subsets), cached in HF_HOME on demand.
LAMA     : FALLBACK — LAMA T-REx/Google-RE are script-based on Hub (unsupported
           by datasets>=3.0). A 60-item hand-curated fallback is bundled instead.
           Flag: factual_cloze_eval returns {"source": "fallback"} in its result.
           Wire up real data later by providing a pre-cached JSON file at
           data/eval/lama_fallback.json (same schema: list of {prefix, object}).
assoc_recall : SYNTHETIC — generated in-code with fixed seed.
icl_score    : SYNTHETIC — generated in-code with fixed seed.

CLI
---
    python code/eval_cheap.py --run-dir experiments/<dir> [--ckpt <path>] [--all-checkpoints]
                              [--device cpu] [--dry-run]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make code/ importable
_CODE_DIR = Path(__file__).parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from baseline import BaselineConfig, BaselineTransformer, make_config

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------

_DATA_EVAL_DIR = Path(__file__).parent.parent / "data" / "eval"
_LAMBADA_CACHE = _DATA_EVAL_DIR / "lambada_test.jsonl"
_BLIMP_CACHE   = _DATA_EVAL_DIR / "blimp_subset.jsonl"
_LAMA_CACHE    = _DATA_EVAL_DIR / "lama_fallback.json"

# ---------------------------------------------------------------------------
# LAMA fallback dataset (60 T-REx / Google-RE style cloze items)
# subject + relation expressed as a natural-language prefix → gold object token
# ---------------------------------------------------------------------------

_LAMA_FALLBACK = [
    # P17  country
    {"prefix": "France is a country located in",          "object": "Europe"},
    {"prefix": "Germany is a country located in",         "object": "Europe"},
    {"prefix": "Japan is a country located in",           "object": "Asia"},
    {"prefix": "Brazil is a country located in",          "object": "South"},
    {"prefix": "Canada is a country located in",          "object": "North"},
    {"prefix": "Egypt is a country located in",           "object": "Africa"},
    {"prefix": "Australia is a country located in",       "object": "Oceania"},
    {"prefix": "Argentina is a country located in",       "object": "South"},
    # P36  capital
    {"prefix": "The capital of France is",                "object": "Paris"},
    {"prefix": "The capital of Germany is",               "object": "Berlin"},
    {"prefix": "The capital of Japan is",                 "object": "Tokyo"},
    {"prefix": "The capital of Italy is",                 "object": "Rome"},
    {"prefix": "The capital of Spain is",                 "object": "Madrid"},
    {"prefix": "The capital of China is",                 "object": "Beijing"},
    {"prefix": "The capital of Brazil is",                "object": "Bras"},
    {"prefix": "The capital of Australia is",             "object": "Canberra"},
    {"prefix": "The capital of Canada is",                "object": "Ottawa"},
    {"prefix": "The capital of Russia is",                "object": "Moscow"},
    # P131 located in
    {"prefix": "The Eiffel Tower is located in",          "object": "Paris"},
    {"prefix": "The Colosseum is located in",             "object": "Rome"},
    {"prefix": "The Statue of Liberty is located in",     "object": "New"},
    {"prefix": "Big Ben is located in",                   "object": "London"},
    {"prefix": "The Acropolis is located in",             "object": "Athens"},
    # P27  nationality / citizenship
    {"prefix": "Albert Einstein was born in",             "object": "Germany"},
    {"prefix": "Marie Curie was born in",                 "object": "Poland"},
    {"prefix": "Isaac Newton was born in",                "object": "England"},
    {"prefix": "Leonardo da Vinci was born in",           "object": "Italy"},
    # P106 occupation
    {"prefix": "Albert Einstein was a famous",            "object": "physicist"},
    {"prefix": "William Shakespeare was a famous",        "object": "playwright"},
    {"prefix": "Ludwig van Beethoven was a famous",       "object": "composer"},
    {"prefix": "Pablo Picasso was a famous",              "object": "painter"},
    # P407 language
    {"prefix": "In France, people speak",                 "object": "French"},
    {"prefix": "In Germany, people speak",                "object": "German"},
    {"prefix": "In Japan, people speak",                  "object": "Japanese"},
    {"prefix": "In Brazil, people speak",                 "object": "Portuguese"},
    {"prefix": "In China, people speak",                  "object": "Chinese"},
    {"prefix": "In Russia, people speak",                 "object": "Russian"},
    {"prefix": "In Italy, people speak",                  "object": "Italian"},
    {"prefix": "In Spain, people speak",                  "object": "Spanish"},
    # P30  continent
    {"prefix": "Africa is a",                             "object": "continent"},
    {"prefix": "Asia is a",                               "object": "continent"},
    {"prefix": "Europe is a",                             "object": "continent"},
    # P571 inception / founded
    {"prefix": "The company Apple was founded by Steve",  "object": "Jobs"},
    {"prefix": "Microsoft was founded by Bill",           "object": "Gates"},
    {"prefix": "Amazon was founded by Jeff",              "object": "Bezos"},
    # animals
    {"prefix": "A dog is a domestic",                     "object": "animal"},
    {"prefix": "A cat is a domestic",                     "object": "animal"},
    {"prefix": "A lion is a wild",                        "object": "animal"},
    {"prefix": "A whale is a marine",                     "object": "mammal"},
    # science
    {"prefix": "Water is made of hydrogen and",           "object": "oxygen"},
    {"prefix": "The Earth orbits around the",             "object": "Sun"},
    {"prefix": "The Moon orbits around the",              "object": "Earth"},
    {"prefix": "Humans breathe",                          "object": "air"},
    # colours
    {"prefix": "The sky is typically",                    "object": "blue"},
    {"prefix": "Grass is typically",                      "object": "green"},
    {"prefix": "Snow is typically",                       "object": "white"},
    # food
    {"prefix": "Pizza originated in",                     "object": "Italy"},
    {"prefix": "Sushi originated in",                     "object": "Japan"},
    {"prefix": "Croissants originated in",                "object": "France"},
    # misc
    {"prefix": "The currency of Japan is the",            "object": "yen"},
    {"prefix": "The currency of the UK is the",           "object": "pound"},
    {"prefix": "The currency of the US is the",           "object": "dollar"},
    {"prefix": "The language of programming Python was created by",  "object": "Guido"},
]


# ---------------------------------------------------------------------------
# Dataset prep helpers
# ---------------------------------------------------------------------------

def _ensure_lambada(max_items: int = 5000) -> list[dict]:
    """
    Load LAMBADA test set. Returns list of {"text": str}.
    Caches to _LAMBADA_CACHE on first call; reads cache on subsequent calls.
    """
    _DATA_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if _LAMBADA_CACHE.exists():
        items = []
        with open(_LAMBADA_CACHE, encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))
        return items[:max_items]

    from datasets import load_dataset
    ds = load_dataset("EleutherAI/lambada_openai", split="test")
    items = [{"text": row["text"]} for row in ds][:max_items]
    with open(_LAMBADA_CACHE, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    return items


_BLIMP_SUBSETS = [
    "anaphor_gender_agreement",
    "anaphor_number_agreement",
    "determiner_noun_agreement_1",
    "determiner_noun_agreement_2",
    "regular_plural_subject_verb_agreement_1",
    "regular_plural_subject_verb_agreement_2",
    "wh_questions_object_gap",
    "wh_questions_subject_gap",
]


def _ensure_blimp(items_per_subset: int = 200) -> list[dict]:
    """
    Load a subset of BLiMP minimal pairs.
    Returns list of {"sentence_good": str, "sentence_bad": str}.
    Caches to _BLIMP_CACHE; reads cache on subsequent calls.
    """
    _DATA_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if _BLIMP_CACHE.exists():
        items = []
        with open(_BLIMP_CACHE, encoding="utf-8") as f:
            for line in f:
                items.append(json.loads(line))
        return items

    from datasets import load_dataset
    items = []
    for sub in _BLIMP_SUBSETS:
        ds = load_dataset("nyu-mll/blimp", sub, split="train")
        for row in list(ds)[:items_per_subset]:
            items.append({
                "sentence_good": row["sentence_good"],
                "sentence_bad":  row["sentence_bad"],
            })

    with open(_BLIMP_CACHE, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    return items


def _load_lama() -> tuple[list[dict], bool]:
    """
    Load LAMA cloze items. Returns (items, is_real).
    Tries user-provided JSON at _LAMA_CACHE first (real data), else uses fallback.
    """
    if _LAMA_CACHE.exists():
        with open(_LAMA_CACHE, encoding="utf-8") as f:
            items = json.load(f)
        return items, True
    return _LAMA_FALLBACK, False


# ---------------------------------------------------------------------------
# Tokenizer helper
# ---------------------------------------------------------------------------

_TOK_CACHE: Any = None


def _get_tokenizer():
    global _TOK_CACHE
    if _TOK_CACHE is None:
        from transformers import AutoTokenizer
        _TOK_CACHE = AutoTokenizer.from_pretrained(
            "NousResearch/Llama-2-7b-hf",
            use_fast=True,
        )
    return _TOK_CACHE


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------

@torch.no_grad()
def _sequence_log_prob(
    model: nn.Module,
    token_ids: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """
    Compute sum of log-probs of all tokens in token_ids (treating position 0
    as the first prediction target given an empty prefix — i.e., joint prob).
    Actually returns avg log-prob per token (for ppl consistency).

    In practice we use this to score a FULL sequence and the prefix context
    is included. For LAMBADA/BLiMP we score the target suffix only.
    """
    if len(token_ids) < 2:
        return 0.0
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    x = ids[:, :-1]
    y = ids[:, 1:]
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
        logits, _ = model(x)
    log_probs = F.log_softmax(logits, dim=-1)  # (1, T-1, vocab)
    # gather log-probs of the actual tokens
    gathered = log_probs[0, torch.arange(y.shape[1]), y[0]]  # (T-1,)
    return gathered.sum().item()


@torch.no_grad()
def _token_log_prob_at_position(
    model: nn.Module,
    prefix_ids: list[int],
    target_id: int,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """
    Given a prefix, return log P(target_id | prefix) from the model.
    Also returns top-1 predicted token id.
    Returns (log_prob, top1_id).
    """
    if not prefix_ids:
        # degenerate: return uniform
        return -math.log(model.cfg.vocab_size), 0
    ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    # clamp to seq_len
    if ids.shape[1] > model.cfg.seq_len:
        ids = ids[:, -model.cfg.seq_len:]
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
        logits, _ = model(ids)
    last_logits = logits[0, -1, :]  # (vocab,)
    log_probs = F.log_softmax(last_logits, dim=-1)
    top1 = int(last_logits.argmax().item())
    return float(log_probs[target_id].item()), top1


@torch.no_grad()
def _full_sequence_nll(
    model: nn.Module,
    token_ids: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """
    Return mean negative log-likelihood per token for the sequence.
    Used by BLiMP: lower NLL = more probable = grammatical.
    """
    if len(token_ids) < 2:
        return float("inf")
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    if ids.shape[1] > model.cfg.seq_len + 1:
        ids = ids[:, -(model.cfg.seq_len + 1):]
    x = ids[:, :-1]
    y = ids[:, 1:]
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
        _, loss = model(x, y)
    return float(loss.item())


# ---------------------------------------------------------------------------
# 1. LAMBADA eval
# ---------------------------------------------------------------------------

def lambada_eval(
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    max_items: int = 300,
) -> dict[str, float]:
    """
    LAMBADA last-word top-1 accuracy and mean per-token ppl of the target word.

    For each passage:
      - prefix  = all tokens except those of the final word
      - target  = the tokenized final word (with leading space)
      - top-1 correct if the model's argmax at end of prefix == first target token
      - target_ppl: exp(-mean log P(target_tok | prefix so far)) over target tokens

    Returns: {"top1_acc": float, "target_ppl": float, "n_items": int}
    """
    tok = _get_tokenizer()
    items = _ensure_lambada(max_items)

    model.eval()
    n_correct = 0
    n_total = 0
    log_probs_sum = 0.0
    log_probs_n   = 0

    for item in items:
        text = item["text"].strip()
        # Split off last word (by whitespace boundary)
        parts = text.rsplit(None, 1)
        if len(parts) < 2:
            continue
        prefix_text, last_word = parts
        # Encode prefix (BOS included by tokenizer default)
        prefix_ids = tok.encode(prefix_text)
        # Encode last word with a leading space (standard sub-word convention)
        target_ids = tok.encode(" " + last_word, add_special_tokens=False)
        if not target_ids:
            continue

        # Score each target token conditioned on all previous tokens
        running_prefix = list(prefix_ids)
        first_tok = True
        item_log_probs = 0.0
        correct_top1   = False

        for tid in target_ids:
            lp, top1 = _token_log_prob_at_position(
                model, running_prefix, tid, device, dtype
            )
            if first_tok and top1 == tid:
                correct_top1 = True
            item_log_probs += lp
            running_prefix.append(tid)
            first_tok = False

        n_correct += int(correct_top1)
        n_total   += 1
        log_probs_sum += item_log_probs
        log_probs_n   += len(target_ids)

    if n_total == 0:
        return {"top1_acc": 0.0, "target_ppl": float("inf"), "n_items": 0}

    top1_acc   = n_correct / n_total
    target_ppl = math.exp(-log_probs_sum / max(log_probs_n, 1))
    return {"top1_acc": round(top1_acc, 4), "target_ppl": round(target_ppl, 3), "n_items": n_total}


# ---------------------------------------------------------------------------
# 2. Factual cloze eval (LAMA T-REx / Google-RE style)
# ---------------------------------------------------------------------------

def factual_cloze_eval(
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    """
    LAMA-style factual cloze: given a prefix (subject + relation), score the
    gold object token and compute top-1 accuracy and mean reciprocal rank.

    The gold "object" is the FIRST token of the object string (with leading space).
    This is standard practice for single-token evaluation on LAMA.

    Returns: {"top1_acc": float, "mrr": float, "n_items": int, "source": str}
    where source is "real" if data/eval/lama_fallback.json exists, else "fallback".
    """
    tok = _get_tokenizer()
    items, is_real = _load_lama()

    model.eval()
    n_correct = 0
    mrr_sum   = 0.0
    n_total   = 0

    for item in items:
        prefix_text = item["prefix"].strip()
        object_text = item["object"].strip()

        prefix_ids = tok.encode(prefix_text)
        # Gold object: first token with leading space
        obj_ids = tok.encode(" " + object_text, add_special_tokens=False)
        if not obj_ids:
            continue
        gold_id = obj_ids[0]

        # Get full distribution at the last position
        ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        if ids.shape[1] > model.cfg.seq_len:
            ids = ids[:, -model.cfg.seq_len:]
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                logits, _ = model(ids)
        last_logits = logits[0, -1, :]  # (vocab,)

        # Rank: argsort descending; rank is 1-based position of gold token
        sorted_ids = torch.argsort(last_logits, descending=True)
        # Find rank of gold_id (efficiently with searchsorted on CPU)
        rank_tensor = (sorted_ids == gold_id).nonzero(as_tuple=True)[0]
        if len(rank_tensor) == 0:
            rank = model.cfg.vocab_size  # shouldn't happen
        else:
            rank = int(rank_tensor[0].item()) + 1  # 1-based

        n_correct += int(rank == 1)
        mrr_sum   += 1.0 / rank
        n_total   += 1

    if n_total == 0:
        return {"top1_acc": 0.0, "mrr": 0.0, "n_items": 0, "source": "fallback"}

    return {
        "top1_acc": round(n_correct / n_total, 4),
        "mrr":      round(mrr_sum / n_total, 4),
        "n_items":  n_total,
        "source":   "real" if is_real else "fallback",
    }


# ---------------------------------------------------------------------------
# 3. BLiMP minimal pairs eval
# ---------------------------------------------------------------------------

def blimp_eval(
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    max_items: int = 800,
) -> dict[str, float]:
    """
    BLiMP: fraction of minimal pairs where model assigns lower per-token NLL
    (= higher likelihood) to the grammatical sentence than the ungrammatical one.

    Returns: {"accuracy": float, "n_pairs": int}
    """
    items = _ensure_blimp()[:max_items]
    tok = _get_tokenizer()

    model.eval()
    n_correct = 0
    n_total   = 0

    for item in items:
        good_ids = tok.encode(item["sentence_good"])
        bad_ids  = tok.encode(item["sentence_bad"])
        if len(good_ids) < 2 or len(bad_ids) < 2:
            continue

        nll_good = _full_sequence_nll(model, good_ids, device, dtype)
        nll_bad  = _full_sequence_nll(model, bad_ids,  device, dtype)

        # Grammatical wins if it has strictly lower NLL
        n_correct += int(nll_good < nll_bad)
        n_total   += 1

    if n_total == 0:
        return {"accuracy": 0.0, "n_pairs": 0}

    return {"accuracy": round(n_correct / n_total, 4), "n_pairs": n_total}


# ---------------------------------------------------------------------------
# 4. Associative recall / induction (synthetic)
# ---------------------------------------------------------------------------

def assoc_recall_eval(
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    n_sequences: int = 200,
    n_pairs: int = 8,
    seed: int = 7777,
) -> dict[str, float]:
    """
    Synthetic in-context associative recall.

    Each sequence:  k1->v1 k2->v2 ... kN->vN [query: ki] [predict: vi]

    Keys and values are random tokens from a small alphabet of 256 tokens (chosen
    to avoid UNK / special tokens). The model must have seen the binding in-context
    and recall the value at query time.

    This is a clean ICL recall test: it rewards induction heads and associative
    memory, not world knowledge.

    Returns: {"exact_acc": float, "n_sequences": int}
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    tok = _get_tokenizer()
    vocab_size = model.cfg.vocab_size

    # Pick a fixed alphabet of safe tokens (indices 1000-1256, avoiding specials)
    alphabet = list(range(1000, min(1000 + 256, vocab_size)))

    # Arrow token: encode " ->" and take the first real token
    arrow_ids = tok.encode(" ->", add_special_tokens=False)
    sep_ids   = tok.encode(" ;",  add_special_tokens=False)
    query_ids = tok.encode(" Q:", add_special_tokens=False)
    ans_ids   = tok.encode(" A:", add_special_tokens=False)

    model.eval()
    n_correct = 0
    n_total   = 0

    for _ in range(n_sequences):
        # Sample n_pairs distinct key tokens and value tokens
        keys   = rng.sample(alphabet, n_pairs)
        values = rng.sample(alphabet, n_pairs)
        # Shuffle the order in which k->v pairs appear
        pair_order = list(range(n_pairs))
        rng.shuffle(pair_order)
        # Query: one of the pairs
        query_idx = rng.randrange(n_pairs)
        query_key = keys[query_idx]
        gold_val  = values[query_idx]

        # Build token sequence
        seq: list[int] = []
        for pi in pair_order:
            seq.append(keys[pi])
            seq += arrow_ids
            seq.append(values[pi])
            seq += sep_ids
        seq += query_ids
        seq.append(query_key)
        seq += ans_ids
        # Now the model should predict gold_val at this position

        if len(seq) > model.cfg.seq_len:
            # Won't fit; skip (shouldn't happen for n_pairs=8)
            continue

        _, top1 = _token_log_prob_at_position(model, seq, gold_val, device, dtype)
        n_correct += int(top1 == gold_val)
        n_total   += 1

    if n_total == 0:
        return {"exact_acc": 0.0, "n_sequences": 0}

    return {"exact_acc": round(n_correct / n_total, 4), "n_sequences": n_total}


# ---------------------------------------------------------------------------
# 5. ICL score eval (synthetic)
# ---------------------------------------------------------------------------

def icl_score_eval(
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    n_trials: int = 100,
    n_shots: int = 8,
    seed: int = 8888,
) -> dict[str, float]:
    """
    Synthetic few-shot in-context learning score.

    Each trial: build a prompt with n_shots demonstrations of a simple mapping
    (tokens from a random permutation of an alphabet). The mapping is a fixed
    offset (mod alphabet_size) chosen per trial.

    Metric: loss(first demo) - loss(last demo), where loss is next-token NLL
    for the value token in each demonstration.

    Positive = model learns from context (uses later shots better than earlier).
    Random/untrained models should be near 0. Strong ICL yields positive values.

    Returns: {"icl_score": float, "loss_early": float, "loss_late": float, "n_trials": int}
    """
    rng = random.Random(seed)
    tok = _get_tokenizer()
    vocab_size = model.cfg.vocab_size

    # Use a small alphabet of 64 tokens
    alphabet = list(range(2000, min(2000 + 64, vocab_size)))
    arrow_ids = tok.encode(" ->", add_special_tokens=False)
    sep_ids   = tok.encode(" ;",  add_special_tokens=False)

    model.eval()
    early_losses: list[float] = []
    late_losses:  list[float] = []

    for _ in range(n_trials):
        # Random permutation mapping key->value (consistent within this trial)
        keys   = rng.sample(alphabet, n_shots)
        values = rng.sample(alphabet, n_shots)
        # Shuffle key-value pairs so model can't exploit positional identity
        order = list(range(n_shots))
        rng.shuffle(order)

        # Build the full prompt, recording which token positions are value tokens
        seq: list[int] = []
        value_positions: list[tuple[int, int]] = []  # (position_of_value_token, value_token_id)

        for shot_idx, pi in enumerate(order):
            key_tok = keys[pi]
            val_tok = values[pi]
            seq.append(key_tok)
            seq += arrow_ids
            val_pos = len(seq)   # index in seq where value token will be placed
            seq.append(val_tok)
            seq += sep_ids
            # We record the position of the PRECEDING token (from which we predict val_tok)
            value_positions.append((val_pos, val_tok))  # predict val_tok at val_pos

        if len(seq) > model.cfg.seq_len + 1:
            continue

        # Forward pass once for the full sequence
        ids = torch.tensor([seq], dtype=torch.long, device=device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
                logits, _ = model(ids)
        log_probs = F.log_softmax(logits[0], dim=-1)  # (T, vocab)

        # Collect log-probs for value tokens at each shot
        # log_probs[t] = distribution over next token given seq[:t+1]
        # value token at position val_pos is predicted by log_probs[val_pos - 1]
        shot_lp: list[float] = []
        for (val_pos, val_tok) in value_positions:
            pred_pos = val_pos - 1  # logit position that predicts val_pos
            if pred_pos < 0 or pred_pos >= log_probs.shape[0]:
                continue
            lp = float(log_probs[pred_pos, val_tok].item())
            shot_lp.append(-lp)  # NLL (positive = worse)

        if len(shot_lp) < 2:
            continue

        # "Early" = first shot NLL, "Late" = last shot NLL (after most context)
        early_losses.append(shot_lp[0])
        late_losses.append(shot_lp[-1])

    if not early_losses:
        return {"icl_score": 0.0, "loss_early": 0.0, "loss_late": 0.0, "n_trials": 0}

    loss_early = float(np.mean(early_losses))
    loss_late  = float(np.mean(late_losses))
    icl_score  = loss_early - loss_late  # positive = model learned from context

    return {
        "icl_score":  round(icl_score, 4),
        "loss_early": round(loss_early, 4),
        "loss_late":  round(loss_late,  4),
        "n_trials":   len(early_losses),
    }


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def eval_all_cheap(
    model: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    datasets: dict | None = None,   # reserved for future use (e.g. cached dataset objects)
) -> dict[str, dict[str, float]]:
    """
    Run all five block-A evals and return a nested dict:
        {eval_name: {metric_name: value, ...}}
    """
    results: dict[str, dict] = {}

    t0 = time.monotonic()
    print("  [eval_cheap] lambada ...", flush=True)
    results["lambada"] = lambada_eval(model, device, dtype)
    print(f"    -> {results['lambada']}  ({time.monotonic()-t0:.1f}s)")

    t1 = time.monotonic()
    print("  [eval_cheap] factual_cloze ...", flush=True)
    results["factual_cloze"] = factual_cloze_eval(model, device, dtype)
    print(f"    -> {results['factual_cloze']}  ({time.monotonic()-t1:.1f}s)")

    t2 = time.monotonic()
    print("  [eval_cheap] blimp ...", flush=True)
    results["blimp"] = blimp_eval(model, device, dtype)
    print(f"    -> {results['blimp']}  ({time.monotonic()-t2:.1f}s)")

    t3 = time.monotonic()
    print("  [eval_cheap] assoc_recall ...", flush=True)
    results["assoc_recall"] = assoc_recall_eval(model, device, dtype)
    print(f"    -> {results['assoc_recall']}  ({time.monotonic()-t3:.1f}s)")

    t4 = time.monotonic()
    print("  [eval_cheap] icl_score ...", flush=True)
    results["icl_score"] = icl_score_eval(model, device, dtype)
    print(f"    -> {results['icl_score']}  ({time.monotonic()-t4:.1f}s)")

    print(f"  [eval_cheap] done  total {time.monotonic()-t0:.1f}s", flush=True)
    return results


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def load_checkpoint(
    ckpt_path: str,
    device: str | torch.device = "cpu",
) -> tuple[BaselineTransformer, int]:
    """
    Load a BaselineTransformer from a checkpoint saved by train._save_checkpoint().

    Checkpoint format (saved via torch.save):
        {
            "state_dict":   OrderedDict,   # model weights
            "train_config": dict,          # TrainConfig fields
            "model_config": dict,          # BaselineConfig fields (callables as strings)
        }

    Callable fields (attn_factory, ffn_factory, etc.) are stored as strings, so the
    real factories are re-derived via load_novelty(candidate_id) — variant checkpoints
    (toroidal MoE, MCT, etc.) rebuild with their actual architecture before loading.

    Returns (model, step) where step is inferred from the filename
    (e.g. ckpt_step2500.pt -> 2500, ckpt_final.pt -> -1).
    """
    device = torch.device(device) if isinstance(device, str) else device
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    mc = ckpt["model_config"]
    tc = ckpt.get("train_config", {}) or {}
    # Callable fields are saved as strings, so re-derive the REAL factories via
    # load_novelty using the saved candidate_id (same path train_entry uses).
    # Without this a variant checkpoint (e.g. toroidal MoE in the last layer)
    # rebuilds as plain dense and load_state_dict fails on the shape mismatch.
    _CALLABLE_FIELDS = {"attn_factory", "ffn_factory", "norm_factory",
                        "pre_layer_hook", "post_layer_hook"}
    cfg_kwargs = {k: v for k, v in mc.items() if k not in _CALLABLE_FIELDS}

    variant_or_baseline = tc.get("variant_or_baseline", "baseline")
    candidate_id = tc.get("candidate_id")
    novelty_id = candidate_id if (variant_or_baseline != "baseline" and candidate_id) else "baseline"
    from baseline import make_config           # local import to avoid import cycles
    from novelty import load_novelty
    factories = load_novelty(novelty_id, make_config(d_model=cfg_kwargs["d_model"]))
    for fld in _CALLABLE_FIELDS:
        cfg_kwargs[fld] = factories.get(fld)

    cfg = BaselineConfig(**cfg_kwargs)
    model = BaselineTransformer(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    # Infer step from filename
    fname = Path(ckpt_path).stem  # e.g. "ckpt_step2500" or "ckpt_final"
    step = -1
    if "step" in fname:
        try:
            step = int(fname.split("step")[-1])
        except ValueError:
            pass

    return model, step


def find_checkpoints(run_dir: str) -> list[str]:
    """Return sorted list of checkpoint paths in run_dir (by step number)."""
    run_path = Path(run_dir)
    ckpts = sorted(
        run_path.glob("ckpt_*.pt"),
        key=lambda p: (
            # final last
            1 if p.stem == "ckpt_final" else 0,
            int(p.stem.split("step")[-1]) if "step" in p.stem else 0,
        ),
    )
    return [str(p) for p in ckpts]


# ---------------------------------------------------------------------------
# DB writer (thin wrapper — called only when not --dry-run)
# ---------------------------------------------------------------------------

def _write_to_db(
    db_path: str,
    run_dir: str,
    step: int,
    d_model: int,
    results: dict[str, dict],
) -> None:
    """
    Write eval_cheap results to the eval_results table via db.add_eval_results().
    Imports db lazily so this file can be imported without db on PYTHONPATH.
    """
    import db as _db
    conn = _db.connect(db_path)

    # Look up run_id by run_dir
    row = conn.execute(
        "SELECT id FROM runs WHERE run_dir=? ORDER BY id DESC LIMIT 1",
        (str(Path(run_dir).resolve()),),
    ).fetchone()
    if row is None:
        # Try matching the basename (run_dir may be relative in DB)
        basename = Path(run_dir).name
        row = conn.execute(
            "SELECT id FROM runs WHERE run_dir LIKE ? ORDER BY id DESC LIMIT 1",
            (f"%{basename}",),
        ).fetchone()

    run_id = row["id"] if row else None

    _db.add_eval_results(conn, run_id, step, d_model, results)
    conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Loop_Dev_AI cheap eval suite (block A)"
    )
    parser.add_argument("--run-dir",  required=True,
                        help="Experiment directory, e.g. experiments/e0000__dense_baseline_d512")
    parser.add_argument("--ckpt",     default=None,
                        help="Path to a specific checkpoint .pt file")
    parser.add_argument("--all-checkpoints", action="store_true",
                        help="Eval every ckpt_*.pt in --run-dir")
    parser.add_argument("--device",   default="cpu",
                        help="torch device (default: cpu)")
    parser.add_argument("--dtype",    default="float32",
                        choices=["float32", "bfloat16", "float16"],
                        help="Autocast dtype (default: float32 for CPU; bfloat16 for GPU)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print results only; do NOT write to DB")
    parser.add_argument("--db",       default="db/loop_dev_ai.db",
                        help="Path to SQLite DB (default: db/loop_dev_ai.db)")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    dtype = dtype_map[args.dtype]

    # Collect checkpoint paths
    if args.all_checkpoints:
        ckpt_paths = find_checkpoints(args.run_dir)
        if not ckpt_paths:
            print(f"No checkpoints found in {args.run_dir}", file=sys.stderr)
            sys.exit(1)
    elif args.ckpt:
        ckpt_paths = [args.ckpt]
    else:
        # Default: ckpt_final.pt if it exists, else the latest step checkpoint
        ckpt_paths = find_checkpoints(args.run_dir)
        if not ckpt_paths:
            print(f"No checkpoints found in {args.run_dir}", file=sys.stderr)
            sys.exit(1)
        ckpt_paths = [ckpt_paths[-1]]  # last = final or highest step

    print(f"Checkpoints to evaluate ({len(ckpt_paths)}):")
    for p in ckpt_paths:
        print(f"  {p}")
    print(f"Device: {device}  dtype: {dtype}  dry-run: {args.dry_run}\n")

    for ckpt_path in ckpt_paths:
        print(f"\n=== Evaluating: {ckpt_path} ===")
        model, step = load_checkpoint(ckpt_path, device)
        d_model = model.cfg.d_model
        print(f"    step={step}  d_model={d_model}  n_layers={model.cfg.n_layers}")

        results = eval_all_cheap(model, device, dtype)

        # Pretty-print
        print("\n--- Results ---")
        for eval_name, metrics in results.items():
            print(f"  {eval_name}:")
            for k, v in metrics.items():
                print(f"    {k}: {v}")

        if args.dry_run:
            print("  [dry-run: skipping DB write]")
        else:
            # Ensure DB + eval_results table exist
            import db as _db
            _db.init_db(args.db)
            _write_to_db(args.db, args.run_dir, step, d_model, results)
            print(f"  [wrote to DB: {args.db}]")

        print()


if __name__ == "__main__":
    main()
