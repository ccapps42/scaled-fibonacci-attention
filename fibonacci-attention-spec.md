# Scaled-Fibonacci Attention — Implementation Spec

**Date:** 2026-06-07
**Status:** DRAFT for red-line. No code until approved.
**Author of design:** Chad Capps. Drafted by Claude.
**Companion:** `fibonacci-attention-concept.md` (the original concept; this spec supersedes it on every point of conflict — see §2).

---

## 1. Purpose and scope

Test one mechanism: **sparse attention at Fibonacci-spaced offsets with a per-layer learned scalar α that compresses or expands the spacing** ("the spring"). Two axes:

1. **Perplexity** — held-out language-modeling quality.
2. **Multi-hop reasoning** — RULER Variable Tracking + LEGO.

The study is about **inductive bias, not efficiency.** At seq=1024 / d=512 the attention term is ~16% of per-layer compute and the gather implementation is not faster than fused dense, so FLOPs/speed are reported as **context only**, never as a promotion criterion (§8.4).

This is a **standalone project** in `K:\projects\fibbonacci`. It reuses the `Loop_Dev_AI` harness, config, and data bins as a **read-only dependency**. `Loop_Dev_AI` is never written to.

---

## 2. What changed from the concept doc, and why

| Concept doc | This spec | Reason |
|---|---|---|
| Gumbel-Softmax over a candidate superset + temperature annealing to learn α | **Differentiable interpolated gather** (floor/ceil bracket weighted by fractional part) | At seq=1024 there is no efficiency to harvest, so the heavy superset/temperature machinery buys nothing and adds train/inference mismatch and tuning risk. Interpolation makes α cleanly differentiable with far less surface area. |
| Target scale 1–3B, long context, efficiency framing | d=512, seq=1024, inductive-bias framing | Reuses an existing matched bench and three trained comparators; efficiency is a separate long-context experiment, not this one. |
| α range [0.25, 4.0] | **α ∈ [0.5, 1.5]**, init 1.0 | The base sequence already spans the context at α=1 (fib(15)≈987 ≈ 1024). Expanding (α>1) only clips far rungs and coarsens sampling with no reach gain; α>2 is strictly worse. Useful dynamic range is compression (α<1) plus a small init-stability margin above 1. |
| Self position not attended (local window = `i-W..i-1`, fib = `i-f`) | **Self (distance 0) always attended** | Empty-row safety (position 0, early positions) and parity with the causal convention the baseline uses (query i attends to key i). Deviation from the concept doc, made for correctness. |
| Pure sparse for compute savings | Sparse as a **per-layer distance prior**; full-resolution local window retained | Local syntax needs dense near coverage regardless of where α drifts (the window is the α-independent local guarantee). |

---

## 3. Host and read-only reuse

### 3.1 Exact recipe (inherited verbatim from e0000 / e0030 / e0084)

**Model** (`Loop_Dev_AI/code/baseline.py` `BaselineConfig`): `d_model=512, n_layers=16, n_heads=8, n_kv_heads=2, head_dim=64, vocab_size=32768, seq_len=1024, ffn_ratio=8/3 (hidden=1366), rope_theta=10000, rms_norm_eps=1e-6, tie_embeddings=True, dropout=0.0`.

**Training:** `total_steps=13000, batch_size=8, grad_accum=4, seq_len=1024` → 32,768 tokens/step → **425,984,000 tokens**. `lr=3e-4, weight_decay=0.1, beta1=0.9, beta2=0.95, grad_clip=1.0, warmup_steps=300`, cosine schedule, bf16 autocast, AdamW8bit. `eval_interval=500, save_interval=2500`. **seed=42.**

**Data:** `screen_train.bin` (the d=512 screen corpus) + val bins `fineweb_edu_val`, `wikipedia_val`, `tinystories_val`, `math_val`. Accessed read-only (hard-link into this project's `data/`, or direct absolute path). **The training bin and read offset must be byte-identical to the comparators** or the Δ-comparison is invalid.

### 3.2 What this project imports vs. owns

- **Imports read-only** from `K:\projects\Loop_Dev_AI\code\` via an explicit `sys.path` entry: `baseline.py` (model + `attn_factory` hook), the training loop (`train.py` / `train_entry.py`), and the existing cheap-block eval code. The runner passes **this project's** `db_path` and `experiments_dir` so nothing is written into `Loop_Dev_AI`.
- **Owns (new code, lives here):** the Fibonacci attention module, the RULER-VT and LEGO evals, the FLOPs-context reporter, this project's own SQLite DB, and `experiments/`.

### 3.3 Project layout

```
K:\projects\fibbonacci\
  fibonacci-attention-concept.md      (existing)
  fibonacci-attention-spec.md         (this doc)
  code/
    fib_attention.py                  FibonacciAttention + make_factories
    run.py                            thin runner (imports LDA baseline/train, writes here)
    evals/ruler_vt.py  lego.py  flops_context.py
    db.py                             this project's DB layer (schema mirrors LDA)
  db/fib.db
  experiments/  f01__... f10__...
  data/         hard-links to LDA screen_train.bin + val bins (read-only)
```

---

## 4. The mechanism

For a query at position `i`, the attended key set is the union of:

- **Self + local window:** distances `{0, 1, 2, …, W}` (dense, integer, always present).
- **Fibonacci offsets:** for each base rung `f_k`, a target distance `d_k = α · f_k` (continuous), gathered by interpolation.

Base sequence (K=15): `fib = [1,2,3,5,8,13,21,34,55,89,144,233,377,610,987]`.

α is per-layer, learned: `α = 0.5 + 1.0 · sigmoid(θ)`, θ a learnable parameter, **init θ=0 → α=1.0**. Range [0.5, 1.5].

### 4.1 Variant parameters (one module, switched by config)

| Param | Values | Notes |
|---|---|---|
| `offset_base` | `fib` \| `pow2` | `pow2` = log-sparse control, reach-matched (§7) |
| `alpha_mode` | `learned` \| `fixed` | `fixed` freezes α=1.0 (θ not trained) |
| `alpha_scope` | `per_layer` \| `per_head` | per_head → θ shape `[n_heads]` |
| `W` | 6, 8, 10, 12 | local window (swept) |
| `K` | 15 | number of rungs |

All variants run through the **same** `FibonacciAttention` module so there is no numerical-path confound between the learned-α candidate and its controls. Fixed-α and pow2 controls have integer offsets, so their interpolation fractional part is 0 (degenerate, exact gather) — identical code path, no special-casing.

---

## 5. Implementation detail

### 5.1 Module contract

`FibonacciAttention(cfg, layer_idx, **variant_params)` is an `nn.Module` with
`forward(x, cos, sin, mask=None) -> Tensor`, matching the `attn_factory` contract in `baseline.py`. It is supplied via `make_factories(...) -> {"attn_factory": ...}`, the same pattern as `c084_nsa_all16.py`. Projections (`q/k/v/o_proj`), GQA shapes, RoPE application, and the `o_proj.is_residual_proj` init flag are identical to `GQAAttention`. The incoming `mask` arg is ignored (custom attention, like NSA).

### 5.2 Why not the boolean-mask SDPA path

The stride family builds a bool `[T,T]` mask and calls `scaled_dot_product_attention(attn_mask=bool, is_causal=False)`. That path is NaN-safe and fast, **but a hard 0/1 mask carries no gradient to α** — `round(α·f_k)` is non-differentiable and a binary mask is flat in α. So the learned-α candidate must use a custom gather forward. The controls *could* use a bool mask, but must not, per §4.1 (shared path).

### 5.3 Gather + interpolation (the core)

Offsets are shared across all query positions (a function of α and the base), so this is a set of **shifted gathers**, not per-position index lists.

For each rung k with target `d_k = α·f_k`: `lo = floor(d_k)`, `hi = ceil(d_k)`, `frac = d_k − lo`. The gathered key for query i is
`K_k[i] = (1−frac)·K[i−lo] + frac·K[i−hi]`, and likewise for V. Gradient flows through `frac` to θ to α. (RoPE is applied to q,k at their true positions before gathering, as in baseline.)

Local window + self gather exact integer-shifted keys for distances `{0..W}`.

The query then attends (softmax) over its concatenated set of `(W+1) + K` keys/values. Dedup where a window distance coincides with a rounded rung is handled by masking duplicates (or accepted as a benign double-count — to be decided in §12).

### 5.4 Causality and boundaries

- A gathered key at position `i − d < 0` is **invalid** → its attention logit is set to a large negative finite value (not `−inf`, to stay NaN-safe) so its softmax weight is ~0.
- Self (distance 0) is always valid, so **no query ever has an empty key set** (fixes position 0 and early positions).
- Inference uses the standard KV cache: attended positions are a deterministic function of `i` and the (now fixed) α, so cache-and-index works unchanged.

### 5.5 Inference-time α

Train and inference both use the interpolated (soft) gather with the learned α. No hard-snapping step (snapping is only relevant to the efficiency case, which this study excludes). This keeps train/inference identical and removes a mismatch bug class.

---

## 6. Experiment matrix (Option 2 — 10 new runs, seed 42)

| # | run id | offset_base | alpha_mode | scope | W |
|---|---|---|---|---|---|
| 1 | f01__fib_learned_w6  | fib  | learned | per_layer | 6  |
| 2 | f02__fib_learned_w8  | fib  | learned | per_layer | 8  |
| 3 | f03__fib_learned_w10 | fib  | learned | per_layer | 10 |
| 4 | f04__fib_learned_w12 | fib  | learned | per_layer | 12 |
| 5 | f05__fib_fixed_w6    | fib  | fixed   | per_layer | 6  |
| 6 | f06__fib_fixed_w8    | fib  | fixed   | per_layer | 8  |
| 7 | f07__fib_fixed_w10   | fib  | fixed   | per_layer | 10 |
| 8 | f08__fib_fixed_w12   | fib  | fixed   | per_layer | 12 |
| 9 | f09__logsparse_w8    | pow2 | fixed   | per_layer | 8  |
| 10 | f10__fib_learned_ph_w8 | fib | learned | per_head | 8 |

**Headline claim:** learned-α (1–4) vs fixed-α=1 (5–8) **as a function of W** — the value of the spring. Dense (e0000, reused), log-sparse (f09), and learned-α (f02) anchor the four-way at W=8. Per-head (f10) is the secondary variation.

**Measured feasibility (RTX 3090, idle, 2026-06-08).** Dense = 7.9 GB / ~2.1 h.
- **Learned-α runs (f01–f04, f10):** materialized gather at **B=4 ×accum8** — 12.2–12.8 GB, **~10 h/run**, flat across W={6,8,10,12}. This is the fastest *and* safe config (the naive gather at B=8 hit 23.5–24.9 GB and spilled to host RAM on Windows WDDM, ~12–20 h).
- **Fixed-α + pow2 controls (f05–f09):** bool-mask path (option B) at **B=8 ×accum4** — 8.0 GB, **~2.3 h/run** (≈ dense).
- **Streaming + checkpoint (option C):** built and smoke-validated; cuts memory to 8.2 GB even at B=8/W=12, but ~30% slower than materialized-B=4 due to the recompute, so it is **dominated for this matrix** and not used here. Retained as the correct tool for the long-context case (where materialized OOMs at any batch).
- **Core matrix (f01–f09) ≈ 52 GPU-h**, plus f10. Microbatch sizes differ (B=4 learned, B=8 controls) but both give 32,768 tok/step / 426M tokens — same effective batch, matched comparison. GPU is shared; check `nvidia-smi` first.

---

## 7. Control fairness — reach-matched log-sparse

`pow2` reaches 1024 in ~11 rungs (1,2,4,…,1024); fib uses 15 to reach ~987. Fibonacci is intrinsically denser at equal reach — that is part of what "Fibonacci spacing" means. The control **matches reach, not rung count**: pow2 rungs up to the largest ≤ seq, same W, same α=1. Reported explicitly so the comparison is "same reach, different spacing density." Expectation (per literature, §ref): log-sparse is an efficiency mechanism and should roughly match or trail dense on quality; its role is to isolate the spacing choice, not to be a strong competitor.

---

## 8. Evaluation

### 8.1 Cheap block (reuse LDA, every checkpoint)
`val_ppl` (fineweb_edu primary + wiki/tinystories/math), `lambada_full` (target_ppl + top1_acc), `blimp`, `factual_cloze`, `icl_score`, `assoc_recall`. Same code, same cadence as the comparators.

### 8.2 RULER Variable Tracking — primary multi-hop (NEW, this project)
Synthetic, knowledge-free. Chains of variable re-bindings (`X1 = <num>`, `X2 = X1`, …) inserted into filler text; query one target variable's value. **Single-target scoring variant:** rank candidate numeric answers by LM likelihood (LAMBADA-style), report top-1 accuracy. Knobs: **hop count** (2, 3, 4) and **inter-assignment distance** (controls how far apart the chain links sit). Report accuracy vs distance — the discriminating curve for long-range sparse attention. Labeled `ruler_vt` in the DB; flagged as the single-target scoring variant, not verbatim RULER. (Ref: Hsieh et al., RULER, COLM 2024, arXiv:2404.06654.)

### 8.3 LEGO — second reasoning task (NEW, this project)
Chained variable assignments + fixed-group operations; output the resolved value(s). Mechanistically aligned with the per-layer α hypothesis (LEGO transformers develop long-range "association" vs short-range "manipulation" heads). Synthetic, controllable chain length. Labeled `lego`. (Ref: Zhang, Backurs et al., arXiv:2206.04301.)

### 8.4 FLOPs / speed — CONTEXT ONLY (NEW reporter, this project)
Report per model: `flops_per_token` (analytic, with the attention-term breakdown that shows it is ~16% of per-layer compute at this shape), `tokens_per_sec` (train and inference, **labeled which**), `params_total`, `params_nonembed`. **Not a promotion criterion.** Measured on an idle GPU. Surfaces the expected result that the gather variant is not faster than fused dense at seq=1024 — the point of the study is representation quality, not speed.

### 8.5 Backfill on comparators
RULER-VT, LEGO, and the FLOPs reporter are **new** evals not present in the comparators. Run them on the saved `ckpt_final.pt` (and optionally the 5 intermediate checkpoints) of **e0000, e0030, e0084** — read-only — and store results **in this project's DB**. This makes the multi-hop and FLOPs comparisons apples-to-apples without touching `Loop_Dev_AI`.

---

## 9. Comparators

| run | role | params note |
|---|---|---|
| e0000 dense | matched control | the clean param-matched comparison |
| e0030 stride_early {1,3,5} | sibling sparse (fixed integer stride) | param-matched |
| e0084 nsa_all16 | sibling sparse (content selection) | **+~3.6% params, NOT param-matched** — note in every comparison |

The Fibonacci variants add only θ (16 scalars per_layer, or n_heads·16 per_head) over dense, so they are essentially param-matched to e0000. Checkpoints for all three verified present (`ckpt_final.pt` + steps 2500–12500).

---

## 10. DB and provenance

This project's `db/fib.db` mirrors the LDA schema (`runs`, `eval_results`, `metrics`, …) so the existing `compare_to_baseline`-style queries work. Each run logs full `config_json`. Comparator backfill rows are tagged with their source run id and `source_db='Loop_Dev_AI'` so provenance is explicit. The DB is the source of truth; any markdown views are derived.

---

## 11. Methodology and honesty notes

- **Single seed (42)** for all 10 runs, matching the comparators. If the learned-vs-fixed gap holds, a replication seed is required before calling it real (follow-up, not in these 10).
- **Matched comparisons only.** Learned-α vs fixed-α at the **same W**; four-way at **W=8**.
- **No cherry-picking:** because fixed-α is swept across the same four W as learned-α, the spring's value is read off the full learned-vs-fixed curve, not a single chosen W.
- **Expectations stated up front:** fixed-Fibonacci and log-sparse are not expected to beat dense on raw quality at this scale; the live question is whether the *learned spring* moves the result and whether Fibonacci's denser sampling helps *multi-hop* specifically.

---

## 12. Open implementation risks — required before the full matrix

1. **Gather + interpolation correctness.** Smoke test: a tiny model (2 layers, d=64) where `FibonacciAttention` at α=1, W large, K covering all distances reproduces full causal attention within float tolerance; and gradient-checks that ∂loss/∂θ is nonzero and finite.
2. **Duplicate handling** when a window distance equals a rounded rung — mask the duplicate vs. accept the double-count. Decide in the smoke test; pick whichever is simpler and document it.
3. **Boundary masking** uses a large finite negative, never `−inf` (NaN-safe, per the stride family's hard-won convention).
4. **Wall-clock** of the gather path — measure on the smoke test and the first real run; retighten the GPU-hour budget in §6.
5. **Data identity** — confirm `screen_train.bin` read offset and order match the comparators exactly before trusting any Δ.

---

## 13. Locked decisions (recap)

Host = dense, standalone project, read-only LDA reuse · mechanism = interpolated-gather scaled-Fibonacci, self+window always attended · per-layer α primary, α∈[0.5,1.5] init 1.0, K=15 · W swept {6,8,10,12} on both learned and fixed α (Option 2) · log-sparse reach-matched control + per-head variation at W=8 · evals = cheap block + RULER-VT (single-target) + LEGO + FLOPs-as-context · comparators = e0000, e0030, e0084, new evals backfilled into this project's DB · single seed 42, replication is a follow-up.

### Red-line resolutions (Chad, 2026-06-07)
- α range **[0.5, 1.5]**, init 1.0. ✓
- Run-id scheme **`f01…f10`**. ✓
- RULER-VT hop counts **{2,3,4}**, LEGO default chain lengths. ✓
- **Self (distance 0) always attended** (deviation from concept doc, for empty-row safety + causal-convention parity). ✓

### Implementation note added during build (interpolation bracket)
Use the **floor / floor+1** bracket (not floor/ceil) with **clamped** gather of the
further index. At integer α this keeps the gradient to θ alive (the floor/ceil bracket
collapses to zero gradient when every rung lands on an integer simultaneously, e.g. exactly
α=1.0). Clamping the further index to ≥0 keeps boundary positions a convex blend of valid
keys, so the forward value at frac=0 is exactly `K[t−lo]` (matches a boolean-mask reference)
while the right-derivative stays nonzero. Validated by the smoke test.
