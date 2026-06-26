# Scaled-Fibonacci Attention

Code, paper, and experiment database for **"Depth-Staggered Fibonacci Spacing for
Sparse Attention: Static Schedules Beat Learned Dilation and Extrapolate Where
Dense Attention Fails."**

Each query attends to a dense local window plus a set of Fibonacci-spaced offsets,
with a per-layer scalar that scales the spacing. The study compares four ways of
setting that scalar across depth (fixed, per-layer learned, a static linear
stagger, and a coprime reassignment of the stagger) plus a reach-matched
power-of-2 control, over 21 trained language models.

The paper is in [`paper/main.pdf`](paper/main.pdf).

## Findings

- A static per-layer stagger is the dominant lever. It beats both a fixed and a
  learned scalar, and the gain is base-agnostic: staggering a power-of-2 base
  lifts it above fixed Fibonacci and to parity with learned Fibonacci.
- Learning the scalar per layer is inert and costs about 5x the inference latency.
- All sparse variants extrapolate to 4x training length with little degradation,
  while a recipe-matched dense baseline collapses (+201% perplexity at 4x).

## Layout

```
code/        training, evaluation, and analysis scripts
paper/       LaTeX source (main.tex, references.bib) and the compiled PDF
db/fib.db    SQLite database with all training and evaluation results
*.json       position-resolved loss (posloss) and length-extrapolation (lenextrap) dumps
*.ps1        PowerShell helpers to launch (run_all) and watch (watch) the run queue
```

`code/` includes the shared training stack (`baseline.py`, `train.py`, `db.py`,
`util.py`, `eval_multi.py`, `eval_cheap.py`) so the scripts run without any
external project on the path.

## Running

Requires Python with `torch` and `numpy`. The learned-scalar runs also use
`bitsandbytes` (8-bit AdamW); the cheap-eval suite lazily imports `datasets` and
`transformers`. Trained on a single RTX 3090.

```
python code/run.py --list                  # show the run matrix
python code/run.py --run f13__fib_stagger_w12
python code/run.py --run all               # full matrix, resumable, auto-evals
python code/posloss.py                      # position-resolved loss
python code/lenextrap.py                     # length-extrapolation sweep
```

### Not included (external artifacts)

Large artifacts live outside the repo:

- **Trained checkpoints** (`experiments/`, ~41 GB) are gitignored. `run.py` writes
  them there during training; `posloss.py`, `lenextrap.py`, and `eval_posthoc.py`
  read them, so those evaluations need either a training run first or your own
  checkpoints.
- **Training and validation data** (`*.bin` token files). Set the data path at the
  top of `run.py` / `posloss.py` / `lenextrap.py` to your tokenized corpus.
- **Dense comparator checkpoints** (`e0000` and friends), used only by the
  optional `--comparator` path in `eval_posthoc.py`. The variant-checkpoint
  reconstruction branch in `eval_cheap.load_checkpoint` also expects an external
  `novelty` module; the Fibonacci runs do not use that branch.

The committed `db/fib.db` and the JSON dumps already contain the reported results,
so the paper's tables can be reproduced from this repo without re-running anything.
