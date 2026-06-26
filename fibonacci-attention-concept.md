# Fibonacci Sparse Attention — Concept Document

**Date:** 2026-06-03  
**Status:** Exploratory — for implementation by a separate instance  
**Target hardware:** Single NVIDIA RTX 3090 (24GB VRAM), Windows — no Triton kernels  
**Target scale:** 1–3B parameters  
**Training stack:** PyTorch from scratch, HuggingFace datasets for token data  

---

## Core Idea

Standard attention is quadratic in sequence length — every position attends to every other position. Sparse attention schemes reduce this cost by only computing a subset of the full attention matrix. Most existing sparse schemes use local windows (attend to nearby tokens only) which limits long-range reasoning.

This mechanism proposes **Fibonacci-spaced sparse attention**: each position attends to tokens at distances following a Fibonacci-like sequence (1, 2, 3, 5, 8, 13, 21, 34, ...). This allows reaching far into the sequence without computing the full quadratic matrix, because the number of attended positions grows logarithmically relative to sequence length.

**Key properties:**
- Long-range reach at sub-quadratic cost
- Natural bias toward recency (dense at close range, sparse at distance)
- Per-layer α parameter controls compression/expansion of the Fibonacci spacing — learned during training via Gumbel-Softmax
- True sparsity at inference — training uses soft selection, inference uses hard discrete indices

---

## Attention Pattern Detail

For a token at position `i`, the set of attended positions is:

```
attend_to = {i - f : f in fib_sequence, i - f >= 0}
             union
             {i - k : k in 1..local_window}  # local dense component
```

Where `fib_sequence = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ...]`

### Example: position 100, local window = 4, fib up to 89

```
Dense local:  99, 98, 97, 96
Fib sparse:   99, 98, 97, 95, 92, 87, 79, 66, 45, 11
```

(overlap between local and fib at 99, 98, 97 is fine — deduplicate)

Total attended positions: ~14 instead of 100. Cost savings scale with sequence length.

### Causal vs bidirectional

For autoregressive (causal) language modeling: attend only to positions `< i` (past tokens). The Fibonacci offsets are applied backwards from current position.

For encoder/bidirectional: apply offsets in both directions.

---

## Hybrid: Local Dense + Fibonacci Sparse

Pure Fibonacci sparse misses the fine-grained local context that dense attention captures well. The recommended hybrid:

1. **Local dense window**: attend to all tokens within distance `W` (e.g., W=8 or W=16). Captures phrase-level and syntactic relationships.
2. **Fibonacci sparse**: attend to Fibonacci-spaced positions beyond the local window. Captures long-range dependencies.

The two sets are unioned and deduped. A single attention computation runs over this combined index set.

**Why this matters:** Local patterns (subject-verb agreement, punctuation, short phrases) need dense coverage. Long-range patterns (coreference, topic consistency, argument structure) need reach. Neither alone is sufficient.

---

## Scaled Fibonacci — The Spring Model

The Fibonacci sequence is treated as a spring. A scalar parameter α per layer compresses or expands the spacing:

```
scaled_distances = [α × f for f in fibonacci_sequence]
```

- α = 1.0 → standard: 1, 2, 3, 5, 8, 13, 21...
- α = 0.5 → compressed: 0.5, 1, 1.5, 2.5, 4, 6.5... (denser, shorter reach)
- α = 2.0 → expanded: 2, 4, 6, 10, 16, 26, 42... (sparser, longer reach)

Each layer has its own α, learned during training. The result is that layers naturally specialize: some converge to tight α (local-biased), others to large α (long-range-biased), without this being hand-designed.

**The discreteness problem:** Token positions are integers. α×fib values are continuous. Rounding to the nearest integer is non-differentiable, which breaks standard gradient descent. This is solved via Gumbel-Softmax (see below).

---

## Learned α via Gumbel-Softmax

### The Problem

To learn α via gradient descent, we need gradients to flow through the selection of discrete token positions. Rounding has zero gradient almost everywhere — standard backprop fails.

### The Solution: Gumbel-Softmax with Candidate Superset

**Step 1 — Build a candidate superset at init:**

For each layer, precompute all positions that any α in a reasonable range (e.g., 0.25 to 4.0) would ever select:

```python
candidate_distances = sorted(set(
    round(a * f)
    for a in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]
    for f in base_fibonacci
    if round(a * f) > 0
))
```

This produces a fixed superset of ~2-3x the base Fibonacci size. It is computed once and never changes.

**Step 2 — Score each candidate via α:**

For each candidate distance `d` in the superset, compute how well it matches the current α:

```python
# ideal distances at current alpha
ideal = [alpha * f for f in base_fibonacci]
# score each candidate by proximity to nearest ideal distance
scores = [-min(abs(d - ideal_d) for ideal_d in ideal) for d in candidates]
```

**Step 3 — Gumbel-Softmax selection:**

Apply Gumbel-Softmax over the scores to select K positions (one per Fibonacci step):

```python
# during training: soft, differentiable selection
gumbel_weights = F.gumbel_softmax(scores, tau=temperature, hard=False)
output = sum(w * attend(position - d) for w, d in zip(gumbel_weights, candidates))

# during inference: hard discrete selection
selected = candidates[scores.argmax()]
```

**Step 4 — Temperature annealing:**

Start with high temperature (soft, exploratory) and anneal toward zero (hard, sparse):

```python
temperature = max(temp_min, temp_start * (temp_decay ** training_step))
```

Recommended schedule: start τ=1.0, decay to τ=0.1 over first 20% of training, hold at 0.1 for remainder. At inference, τ→0 collapses to hard argmax — true sparse selection.

### Training vs Inference Behavior

| Phase | Selection | Sparsity | Gradient |
| --- | --- | --- | --- |
| Early training | Soft over superset | None — attends to all candidates | Flows through Gumbel weights |
| Late training | Near-hard, peaked | Near-sparse | Flows but peaked |
| Inference | Hard argmax | True sparse | N/A |

### Complexity During Training

Training attends over the candidate superset (~2-3x base Fibonacci size), not the full sequence. Still O(n × superset_size) — sub-quadratic. Roughly 1.5-2x slower than fixed-α training, much faster than full attention.

### What α Actually Learns

The gradient signal on α comes from which attended positions produce useful attention weights. If attending to tokens ~10 steps back consistently reduces loss more than tokens ~5 steps back, α drifts upward. Over training, each layer's α converges to the distance regime most useful for its depth in the network — without any hand-design of which layer gets which range.

---

## Complexity Analysis

Let `n` = sequence length, `L` = local window size, `F` = number of Fibonacci steps used.

- Standard full attention: O(n²) per layer
- This mechanism: O(n × (L + F)) per layer
- F grows as log_φ(n) where φ = 1.618 (golden ratio) — so F ≈ 1.44 × log₂(n)

For n=2048, L=8, F=16 (covering up to fib(16)=987):
- Full attention: 2048² = 4.2M operations per position-pair
- This mechanism: 2048 × 24 = ~49K — roughly **85x fewer attention operations**

Real speedup depends on kernel implementation — sparse attention requires custom CUDA or careful use of indexed gather operations. Naive PyTorch sparse will not be faster; a custom kernel or block-sparse implementation is needed for real gains.

---

## Implementation Sketch (PyTorch)

```python
def base_fibonacci(max_dist):
    fibs = [1, 2]
    while fibs[-1] < max_dist:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs

def build_candidate_superset(base_fibs, alpha_range=(0.25, 4.0), n_samples=12):
    alphas = torch.linspace(alpha_range[0], alpha_range[1], n_samples)
    candidates = set()
    for a in alphas:
        for f in base_fibs:
            d = round(a.item() * f)
            if d > 0:
                candidates.add(d)
    return sorted(candidates)

class FibonacciAttention(nn.Module):
    def __init__(self, d_model, n_heads, local_window=8, max_seq=2048,
                 temp_start=1.0, temp_min=0.1, temp_decay=0.99995):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.local_window = local_window
        self.temp_start = temp_start
        self.temp_min = temp_min
        self.temp_decay = temp_decay

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # learned per-layer spring compression/expansion scalar
        self.log_alpha = nn.Parameter(torch.zeros(1))  # init α=1.0 (log scale for positivity)

        base_fibs = base_fibonacci(max_seq)
        self.base_fibs = base_fibs
        candidates = build_candidate_superset(base_fibs)
        self.register_buffer('candidates', torch.tensor(candidates, dtype=torch.long))

    @property
    def alpha(self):
        return self.log_alpha.exp()  # ensures α > 0

    def get_selection_scores(self):
        alpha = self.alpha
        ideal = [alpha * f for f in self.base_fibs]
        # score each candidate: negative min distance to any ideal position
        scores = []
        for d in self.candidates:
            score = -min(abs(d.float() - id_) for id_ in ideal)
            scores.append(score)
        return torch.stack(scores)

    def get_sparse_indices(self, pos, seq_len, training, step=0):
        local = list(range(max(0, pos - self.local_window), pos))

        if training:
            temperature = max(self.temp_min,
                              self.temp_start * (self.temp_decay ** step))
            scores = self.get_selection_scores()
            # soft selection over candidate superset
            weights = F.gumbel_softmax(scores, tau=temperature, hard=False)
            # return (weights, candidate_distances) for soft attention
            valid = self.candidates[self.candidates < pos]
            return local, weights, valid
        else:
            # hard selection at inference
            scores = self.get_selection_scores()
            selected_dist = self.candidates[scores.argmax()].item()
            sparse = []
            d = selected_dist
            while pos - d >= 0:
                sparse.append(pos - d)
                d = round(self.alpha.item() * d)  # next scaled step
            return sorted(set(local + sparse))

    def forward(self, x, training_step=0):
        B, T, C = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # naive correctness implementation — replace inner loop with
        # index-gather batched attention for real training speed
        outputs = []
        for pos in range(T):
            if self.training:
                local_idx, gumbel_weights, cand_dists = \
                    self.get_sparse_indices(pos, T, True, training_step)
                # soft attend over candidates (training path)
                all_idx = torch.cat([
                    torch.tensor(local_idx, device=x.device),
                    cand_dists[cand_dists < pos]
                ]).unique()
                # ... (standard attention over all_idx, weighted by gumbel_weights)
            else:
                all_idx = torch.tensor(
                    self.get_sparse_indices(pos, T, False), device=x.device)

            if len(all_idx) == 0:
                outputs.append(torch.zeros(B, C, device=x.device))
                continue

            q = Q[:, pos:pos+1, :]
            k = K[:, all_idx, :]
            v = V[:, all_idx, :]
            scale = self.d_head ** -0.5
            scores = (q @ k.transpose(-2, -1)) * scale
            attn = F.softmax(scores, dim=-1)
            out = (attn @ v).squeeze(1)
            outputs.append(out)

        out = torch.stack(outputs, dim=1)
        return self.W_o(out)
```

**This is a correctness reference, not a performance implementation.** The inner position loop must be replaced with batched index-gather operations for real training speed. See kernel strategy in Open Questions.

---

## Known Related Work

Understanding these neighbors helps differentiate this mechanism and avoid reinventing solved problems:

| Work | Similarity | Key difference |
| --- | --- | --- |
| **Longformer** (Beltagy 2020) | Local window + sparse global tokens | Global tokens are learned/fixed positions, not Fibonacci-spaced |
| **BigBird** (Zaheer 2020) | Local + global + random sparse | Random sparse, not structured Fibonacci pattern |
| **ALiBi** (Press 2021) | Distance-based attention bias | Dampens weights by distance, doesn't skip positions |
| **Sparse Transformer** (Child 2019) | Strided sparse patterns | Fixed stride, not exponentially growing |
| **Log-sparse attention** | Attends at powers-of-2 distances | Powers of 2 vs Fibonacci — similar intuition, different spacing |

**Closest prior work: Log-sparse attention** uses positions at distances 1, 2, 4, 8, 16... (powers of 2). Fibonacci spacing (1, 2, 3, 5, 8, 13...) is denser at short range and grows more slowly — it's a middle ground between dense local and aggressive log-sparse.

**Key novelty:** Per-layer Fibonacci offset variation is not standard in any of the above. The idea that different layers specialize by distance regime via structured sparse patterns is the differentiating design choice.

---

## Open Questions for Implementation

1. **Kernel strategy**: naive loop is too slow for training. **Triton is not viable on Windows.** Options in order of practicality on Windows + 3090: (a) masked dense attention — compute full QKᵀ, apply a precomputed boolean mask setting non-attended positions to -inf before softmax; this is correct and PyTorch-native but doesn't reduce compute, only improves memory slightly via sparsity in the softmax output. (b) `torch.sparse` COO/CSR tensors for the attention matrix — limited CUDA sparse matmul support in PyTorch, worth benchmarking but often slower than dense due to overhead. (c) precomputed index gather — for each position, gather only the attended K/V rows using `torch.index_select`, compute small dense attention over that subset; this reduces actual compute but adds gather overhead. Option (c) is the most promising for real speedup on Windows without custom kernels.

2. **α initialization**: all layers start at α=1.0 (log_alpha=0). Consider initializing different layers at different values to give them a head start on specialization — e.g., early layers at α=0.5, late layers at α=2.0. Worth ablating.

3. **Local window size**: W=8 is conservative. W=16 or W=32 may be better for language — worth ablating.

4. **Bidirectional Fibonacci**: for causal LM, only past positions are attended. For an encoder, both directions. Decide upfront — it affects mask construction.

5. **KV cache at inference**: standard KV cache works because attended positions are a deterministic function of current position — cache all K/V, index into it with the Fibonacci pattern.

6. **Baseline comparison**: train an equivalent-parameter standard transformer on the same data/compute budget. Without this, quality claims are unverifiable.

---

## Recommended Starting Point

1. Implement the naive loop version above and verify it produces coherent outputs on a tiny model (2 layers, 128 dim, 8 heads)
2. Swap to a masked dense attention implementation (compute full QKᵀ, mask non-attended positions to -inf before softmax) — slower than true sparse but correct and easy to validate
3. Profile on 3090 to find the real bottleneck
4. No Triton on Windows — index-gather is the target path for real sparse speedup

Start with sequence length 512, scale to 2048 once the mechanism is validated. Train on a small HuggingFace text dataset (e.g., `wikitext-103`) for fast iteration before committing to a larger training run.

**Windows-specific note:** FlashAttention 2 has Windows CUDA wheels available and is worth trying as the baseline attention kernel — it won't implement Fibonacci sparsity but gives a fast dense reference to compare against. For the sparse mechanism itself, the index-gather approach (option c above) is the recommended path: precompute attended index sets per position once at model init, then use `torch.index_select` at each forward pass.

