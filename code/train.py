"""
train.py — Training loop for Loop_Dev_AI architecture discovery.

Usage:
    from train import TrainConfig, train
    cfg = TrainConfig(...)
    train(cfg, model)

Logs stdout/stderr to experiments/<run_id>/train.log for the duration of the run.
All config fields are serialized to experiments/<run_id>/config.json at startup.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import sys
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Optional bitsandbytes AdamW8bit; fall back gracefully
try:
    import bitsandbytes as bnb
    _HAS_BNB = True
except ImportError:
    _HAS_BNB = False

import db
import eval_multi
from baseline import BaselineTransformer, param_count
from eval_cheap import eval_all_cheap
from util import format_int, gen_run_id, load_memmap, sample_batch, set_seed


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Experiment identity
    run_id: str = ""                # auto-generated if empty (requires short_slug)
    short_slug: str = "run"         # used in auto-generated run_id
    seed: int = 42

    # Data
    data_dir: str = "data"
    train_bin: str = "train.bin"    # relative to data_dir
    val_bins: dict = field(default_factory=lambda: {
        "fineweb_edu_val": "fineweb_edu_val.bin",
        "wikitext_val":    "wikitext_val.bin",
        "tinystories_val": "tinystories_val.bin",
        "code_val":        "code_val.bin",
        "math_val":        "math_val.bin",
    })
    experiments_dir: str = "experiments"

    # --- Run identity for the state DB ---
    candidate_id: str | None = None        # cXXX for a variant arm; None for the shared baseline
    variant_or_baseline: str = "baseline"  # "baseline" or "variant"
    phase: int = 2                         # 2 = screen, 3 = finalist
    title: str = ""                        # human-readable run title; defaults to run_id
    db_path: str = "db/loop_dev_ai.db"
    log_to_db: bool = True

    # Training schedule
    total_steps: int = 10_000
    batch_size: int = 8
    grad_accum: int = 4             # effective batch = batch_size * grad_accum * seq_len tokens

    # Optimizer
    lr: float = 3e-4
    weight_decay: float = 0.1
    moe_no_weight_decay: bool = False  # if True, exclude MoE router/experts ('.moe.' params) from weight decay
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 200

    # Eval & checkpointing
    eval_interval: int = 500        # light eval cadence (val ppl on reduced budget)
    light_eval_tokens: int = 262144 # tokens per dataset for light eval (2^18)
    eval_batch_size: int = 8
    eval_tokens: int = 1 << 20      # tokens per dataset for full eval at save_interval
    save_interval: int = 0          # 0 = only save at end; also triggers full eval + cheap block
    log_interval: int = 50          # per-step metrics print cadence (console)

    # Dtype (used in autocast)
    autocast_dtype: str = "bfloat16"   # string because dataclass must be JSON-serializable

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# LR schedule: cosine with linear warmup
# ---------------------------------------------------------------------------

def _get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
    # Cosine decay to 10% of peak lr (common small-LM convention)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def _fmt_opt(v) -> str:
    """Format an optional per-step scalar (rho/gain); blank dash if the model
    does not expose it (e.g. a plain dense baseline has no LTI gate)."""
    return f"{v:.4f}" if v is not None else "  n/a "


def _fmt_eta(secs: float) -> str:
    secs = int(max(secs, 0.0))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Logging setup: tee stdout/stderr to train.log
# ---------------------------------------------------------------------------

class _Tee:
    """Write to both a file and the original stream."""
    def __init__(self, stream, filepath: str):
        self._stream = stream
        self._file = open(filepath, "w", buffering=1, encoding="utf-8")

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    # Forward everything else (isatty, fileno, etc.) to the original stream
    def __getattr__(self, name):
        return getattr(self._stream, name)


# ---------------------------------------------------------------------------
# Main train function
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig, model: BaselineTransformer) -> None:
    """
    Full training run. Writes all artifacts to experiments/<run_id>/.

    The caller is responsible for constructing model with the desired config
    and moving it to the target device before calling train(). The model must
    already be on the correct device.
    """
    # --- Resolve run_id and output directory ---
    if not cfg.run_id:
        cfg.run_id = gen_run_id(cfg.short_slug, cfg.experiments_dir)
    run_dir = os.path.join(cfg.experiments_dir, cfg.run_id)
    os.makedirs(run_dir, exist_ok=True)

    # --- Redirect stdout/stderr to train.log (tee: still prints to console) ---
    log_path = os.path.join(run_dir, "train.log")
    _tee_out = _Tee(sys.stdout, log_path)
    _tee_err = _Tee(sys.stderr, log_path)
    sys.stdout = _tee_out  # type: ignore[assignment]
    sys.stderr = _tee_err  # type: ignore[assignment]

    try:
        _run_training(cfg, model, run_dir)
    finally:
        # Always restore stdout/stderr even if training crashes
        sys.stdout = _tee_out._stream
        sys.stderr = _tee_err._stream
        _tee_out.close()
        _tee_err.close()


def _run_training(cfg: TrainConfig, model: BaselineTransformer, run_dir: str) -> None:
    # --- Seed ---
    set_seed(cfg.seed)

    # --- Determine device from model ---
    device = next(model.parameters()).device

    # --- Write config.json ---
    # Merge train cfg + model cfg into one record
    model_cfg_dict = dataclasses.asdict(model.cfg) if dataclasses.is_dataclass(model.cfg) else {}
    # Hook callables are not JSON-serializable; replace with their names
    for k, v in list(model_cfg_dict.items()):
        if callable(v):
            model_cfg_dict[k] = getattr(v, "__name__", str(v))
    combined_cfg = {"train": cfg.as_dict(), "model": model_cfg_dict}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(combined_cfg, f, indent=2, default=str)

    # Create empty notes.md as scratch space
    notes_path = os.path.join(run_dir, "notes.md")
    if not os.path.exists(notes_path):
        with open(notes_path, "w") as f:
            f.write("")

    # --- Print header ---
    mcfg = model.cfg
    pcounts = param_count(model)
    total_train_tokens = cfg.total_steps * cfg.batch_size * cfg.grad_accum * mcfg.seq_len
    print(
        f"=== {cfg.run_id} | "
        f"d={mcfg.d_model} L={mcfg.n_layers} H={mcfg.n_heads} KV={mcfg.n_kv_heads} | "
        f"params={format_int(pcounts['total'])} | "
        f"seed={cfg.seed} | "
        f"tokens={format_int(total_train_tokens)} ==="
    )
    print(f"    run_dir: {run_dir}")
    print(f"    device:  {device}  autocast: {cfg.autocast_dtype}")
    print(f"    batch={cfg.batch_size} × grad_accum={cfg.grad_accum} × seq={mcfg.seq_len} = "
          f"{format_int(cfg.batch_size * cfg.grad_accum * mcfg.seq_len)} tok/step")
    print(f"    param breakdown: embed={format_int(pcounts['embed'])} "
          f"attn={format_int(pcounts['attn'])} ffn={format_int(pcounts['ffn'])} "
          f"norm={format_int(pcounts['norm'])}")

    # --- Register run in the state DB (so every run is captured automatically) ---
    db_conn = None
    db_run_id = None
    if cfg.log_to_db:
        db.init_db(cfg.db_path)
        db_conn = db.connect(cfg.db_path)
        db_run_id = db.start_run(
            db_conn,
            title=cfg.title or cfg.run_id,
            run_dir=run_dir,
            candidate_id=cfg.candidate_id,
            variant_or_baseline=cfg.variant_or_baseline,
            phase=cfg.phase,
            config_json=json.dumps(combined_cfg, default=str),
            seed=cfg.seed,
            d_model=mcfg.d_model,
            n_layers=mcfg.n_layers,
            params_effective=pcounts["total"],
        )
        db.mark_run_running(db_conn, db_run_id)
        print(f"    db: {cfg.db_path}  (run id={db_run_id}, phase={cfg.phase})")
    print()
    sys.stdout.flush()

    # --- Data ---
    train_bin_path = os.path.join(cfg.data_dir, cfg.train_bin)
    vocab_size = mcfg.vocab_size
    bin_dtype = np.uint16 if vocab_size <= 65536 else np.uint32
    train_data = load_memmap(train_bin_path, bin_dtype)

    # Build absolute eval dataset paths (skip missing files silently)
    eval_paths = {
        name: os.path.join(cfg.data_dir, fname)
        for name, fname in cfg.val_bins.items()
    }

    # --- Optimizer ---
    # Separate weight-decay and no-decay parameter groups (biases + norms: no decay)
    def _is_no_decay(n: str, p) -> bool:
        # Biases + norms (dim<2) never decay. With moe_no_weight_decay, also exclude
        # the MoE ROUTER only (RTCC-confirmed fix 2026-05-24: weight decay on the
        # router + the DeepSeek aux loss together squeeze top-K routing to ~uniform;
        # dropping decay on the router alone lets its logits regrow). Experts keep decay.
        if p.dim() < 2:
            return True
        return bool(cfg.moe_no_weight_decay and "router" in n)
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and not _is_no_decay(n, p)]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and _is_no_decay(n, p)]
    if cfg.moe_no_weight_decay:
        n_router = sum(1 for n, p in model.named_parameters() if p.requires_grad and "router" in n)
        print(f"    weight decay: {len(decay_params)} decay / {len(nodecay_params)} no-decay "
              f"(MoE router excluded: {n_router} tensor(s))")
    param_groups = [
        {"params": decay_params,   "weight_decay": cfg.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    if _HAS_BNB:
        optimizer = bnb.optim.AdamW8bit(
            param_groups,
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
        )
    else:
        warnings.warn(
            "bitsandbytes not available — falling back to torch.optim.AdamW. "
            "8-bit quantization disabled; VRAM usage will be higher.",
            stacklevel=1,
        )
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
        )

    # --- AMP autocast ---
    _dtype = torch.bfloat16 if cfg.autocast_dtype == "bfloat16" else torch.float16
    autocast_ctx = torch.autocast(device_type=device.type, dtype=_dtype)

    # --- Metrics JSONL ---
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    metrics_file = open(metrics_path, "a", buffering=1, encoding="utf-8")

    # --- Load cheap-block datasets once, before the loop ---
    # These are passed to eval_all_cheap at every full event; loading once avoids
    # repeated HF-cache I/O and keeps dataset objects alive across checkpoints.
    # eval_all_cheap's `datasets` arg is currently a reserved pass-through; we pass
    # None since the function handles its own caching internally.
    _cheap_datasets = None  # placeholder; eval_cheap uses module-level caches

    # --- Training loop ---
    model.train()
    tokens_per_step = cfg.batch_size * cfg.grad_accum * mcfg.seq_len
    recent_secs: deque = deque(maxlen=50)   # rolling window for smoothed tok/s + eta
    t_start = time.monotonic()
    t_step_start = time.monotonic()

    try:
        for step in range(1, cfg.total_steps + 1):
            # LR update
            lr_now = _get_lr(step, cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            # Gradient accumulation
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for micro_step in range(cfg.grad_accum):
                x, y = sample_batch(train_data, cfg.batch_size, mcfg.seq_len, device)
                with autocast_ctx:
                    _, loss = model(x, y)
                    loss = loss / cfg.grad_accum
                loss.backward()
                accum_loss += loss.item()

            # Grad clip + optimizer step
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
            optimizer.step()

            # --- Per-step metrics ---
            t_now = time.monotonic()
            step_secs = t_now - t_step_start
            t_step_start = t_now
            recent_secs.append(step_secs)
            avg_secs = sum(recent_secs) / len(recent_secs)
            tokens_per_sec = tokens_per_step / max(avg_secs, 1e-9)
            wall = t_now - t_start
            vram_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
            eta_secs = (cfg.total_steps - step) * avg_secs

            # Architecture-specific scalars (rho, gain), if the model exposes them.
            # A plain dense baseline has neither -> printed as a blank dash.
            aux = model.aux_metrics() if hasattr(model, "aux_metrics") else {}
            rho = aux.get("rho")
            gain = aux.get("gain")

            row = {
                "step": step,
                "loss": round(accum_loss, 6),
                "lr": lr_now,
                "grad_norm": round(grad_norm, 4),
                "tps": int(tokens_per_sec),
                "vram_gb": round(vram_gb, 2),
                "wall_seconds_elapsed": round(wall, 1),
            }
            if rho is not None:
                row["rho"] = round(rho, 4)
            if gain is not None:
                row["gain"] = round(gain, 4)
            metrics_file.write(json.dumps(row) + "\n")

            # Console print (CART-style line) every log_interval steps
            if step % cfg.log_interval == 0:
                print(
                    f"step {step:5d}/{cfg.total_steps}  "
                    f"rho={_fmt_opt(rho)}  "
                    f"loss={accum_loss:.4f}  "
                    f"gain={_fmt_opt(gain)}  "
                    f"lr={lr_now:.2e}  "
                    f"norm={grad_norm:.3f}  "
                    f"vram={vram_gb:.1f}GB  "
                    f"tok/s={int(tokens_per_sec):,}  "
                    f"eta={_fmt_eta(eta_secs)}"
                )

            # --- Eval dispatch (RNG-isolated) ---
            # Full event at save_interval; light eval at eval_interval only
            # (never both at the same step — saves duplicate metrics rows).
            is_last = step == cfg.total_steps
            _do_full = (cfg.save_interval > 0 and step % cfg.save_interval == 0) or is_last
            _do_light = (not _do_full) and (step % cfg.eval_interval == 0)

            if _do_full or _do_light:
                # --- RNG save ---
                _np_rng = np.random.get_state()
                _torch_rng = torch.get_rng_state()
                _cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None

                if _do_full:
                    # Full event: checkpoint + full val ppl + cheap block
                    if not is_last:
                        ckpt_path = os.path.join(run_dir, f"ckpt_step{step}.pt")
                        _save_checkpoint(model, cfg, ckpt_path)
                        print(f"  [checkpoint saved: {ckpt_path}]")

                    print(f"  [full eval @ step {step}]")
                    sys.stdout.flush()
                    eval_result = eval_multi.evaluate(
                        model=model,
                        step=step,
                        run_dir=run_dir,
                        dataset_paths=eval_paths,
                        batch_size=cfg.eval_batch_size,
                        n_tokens_per_dataset=cfg.eval_tokens,
                        device=device,
                        dtype=_dtype,
                    )
                    for ds_name, ds_metrics in eval_result["datasets"].items():
                        print(
                            f"    {ds_name:25s}  ppl={ds_metrics['ppl']:.2f}  "
                            f"loss={ds_metrics['loss']:.4f}"
                        )
                    print(f"  [full eval done in {eval_result['wall_seconds']:.1f}s]")
                    sys.stdout.flush()
                    if db_conn is not None:
                        db.add_metrics(db_conn, db_run_id, step, {
                            name: {"ppl": m["ppl"], "loss": m["loss"],
                                   "eval_tokens": m.get("eval_tokens", cfg.eval_tokens)}
                            for name, m in eval_result["datasets"].items()
                        })

                    # Cheap block — same device/dtype as training; no_grad via internal decorators
                    model.eval()
                    with torch.no_grad():
                        cheap_results = eval_all_cheap(model, device, _dtype, _cheap_datasets)
                    if db_conn is not None:
                        db.add_eval_results(db_conn, db_run_id, step, mcfg.d_model, cheap_results)

                else:
                    # Light event: reduced-budget val ppl only, no checkpoint, no cheap block
                    print(f"  [light eval @ step {step}]")
                    sys.stdout.flush()
                    eval_result = eval_multi.evaluate(
                        model=model,
                        step=step,
                        run_dir=run_dir,
                        dataset_paths=eval_paths,
                        batch_size=cfg.eval_batch_size,
                        n_tokens_per_dataset=cfg.light_eval_tokens,
                        device=device,
                        dtype=_dtype,
                    )
                    for ds_name, ds_metrics in eval_result["datasets"].items():
                        print(
                            f"    {ds_name:25s}  ppl={ds_metrics['ppl']:.2f}  "
                            f"loss={ds_metrics['loss']:.4f}"
                        )
                    print(f"  [light eval done in {eval_result['wall_seconds']:.1f}s]")
                    sys.stdout.flush()
                    if db_conn is not None:
                        db.add_metrics(db_conn, db_run_id, step, {
                            name: {"ppl": m["ppl"], "loss": m["loss"],
                                   "eval_tokens": m.get("eval_tokens", cfg.light_eval_tokens)}
                            for name, m in eval_result["datasets"].items()
                        })

                model.train()

                # --- RNG restore: ensures training data stream is independent of eval cadence ---
                np.random.set_state(_np_rng)
                torch.set_rng_state(_torch_rng)
                if _cuda_rng is not None:
                    torch.cuda.set_rng_state_all(_cuda_rng)

    except BaseException as exc:
        if db_conn is not None:
            db.mark_run_failed(db_conn, db_run_id, repr(exc))
        raise
    finally:
        metrics_file.close()

    # --- Final checkpoint ---
    final_ckpt = os.path.join(run_dir, "ckpt_final.pt")
    _save_checkpoint(model, cfg, final_ckpt)
    print(f"\nTraining complete. Checkpoint: {final_ckpt}")
    print(f"Total wall time: {time.monotonic() - t_start:.0f}s")

    if db_conn is not None:
        db.mark_run_done(db_conn, db_run_id, time.monotonic() - t_start, total_train_tokens)
        db.log_event(db_conn, "run_done", f"{cfg.run_id} complete",
                     {"run_id": db_run_id, "run_dir": run_dir})
        db_conn.close()


def _save_checkpoint(model: BaselineTransformer, cfg: TrainConfig, path: str) -> None:
    """Save state_dict + serializable config to a .pt file."""
    model_cfg = model.cfg
    # Serialize model config without callables
    try:
        model_cfg_dict = dataclasses.asdict(model_cfg)
        for k, v in list(model_cfg_dict.items()):
            if callable(v):
                model_cfg_dict[k] = getattr(v, "__name__", str(v))
    except Exception:
        model_cfg_dict = {}

    torch.save({
        "state_dict": model.state_dict(),
        "train_config": cfg.as_dict(),
        "model_config": model_cfg_dict,
    }, path)
