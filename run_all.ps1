Set-Location $PSScriptRoot

# Everything not done. Each script is resumable, so finished work is skipped.
# Ordered so the high-value runs come before the slow cheap-sweep new-grid finish.

# Blur (skipped if already done)
python code/retrieval/sweep.py --mechs fib_blur_stag --depths 2 4 8 --seeds 0 1 --distances 233 300 350 512 700

# Coverage benches
python code/retrieval/multitarget.py --depths 2 4 8 --dists 200 350 --seeds 0 1
python code/retrieval/multitarget.py --depths 2 4 8 --dists 200 350 --seeds 0 1 --no-stagger
python code/retrieval/marked.py --depths 2 4 8 --dists 200 350 --seeds 0 1
python code/retrieval/marked.py --depths 2 4 8 --dists 200 350 --seeds 0 1 --no-stagger

# Real-token perplexity matrix (~52 GPU-h) + post-hoc evals + comparator backfill
python code/run.py --run all
python code/eval_posthoc.py --run f01__fib_learned_w6 --device cuda
python code/eval_posthoc.py --run f02__fib_learned_w8 --device cuda
python code/eval_posthoc.py --run f03__fib_learned_w10 --device cuda
python code/eval_posthoc.py --run f04__fib_learned_w12 --device cuda
python code/eval_posthoc.py --run f05__fib_fixed_w6 --device cuda
python code/eval_posthoc.py --run f06__fib_fixed_w8 --device cuda
python code/eval_posthoc.py --run f07__fib_fixed_w10 --device cuda
python code/eval_posthoc.py --run f08__fib_fixed_w12 --device cuda
python code/eval_posthoc.py --run f09__logsparse_w8 --device cuda
python code/eval_posthoc.py --comparator e0000 --device cuda
python code/eval_posthoc.py --comparator e0030 --device cuda
python code/eval_posthoc.py --comparator e0084 --device cuda

# Finish the cheap sweep on the new rung/gap grid (lower priority -- slow gumbel)
python code/retrieval/sweep.py --mechs dense logsparse fib_fixed fib_spread gumbel_fib --depths 2 4 8 --seeds 0 1

# Efficiency benchmark (idle GPU) + final DB sync
python code/efficiency.py | Tee-Object -FilePath experiments/efficiency.txt
python code/retrieval/to_db.py
