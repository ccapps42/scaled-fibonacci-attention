"""
db.py — SQLite access layer for Loop_Dev_AI.

Single-file, stdlib-only. No ORM. All timestamps are UTC ISO-8601 strings.
All SQL uses ? placeholders — never string-interpolated values in queries.
"""

import json
import math
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS candidates (
    id                          TEXT PRIMARY KEY,
    name                        TEXT NOT NULL,
    category                    TEXT NOT NULL,
    novelty_rating              TEXT NOT NULL,
    mechanism                   TEXT,
    why_might_help_at_small_scale TEXT,
    source_paper                TEXT,
    implementation_effort       TEXT,
    risk_notes                  TEXT,
    chinese_origin              INTEGER DEFAULT 0,
    priority                    INTEGER DEFAULT 100,
    tags_json                   TEXT,
    status                      TEXT NOT NULL DEFAULT 'pending',
    notes                       TEXT,
    created_at                  TEXT NOT NULL,
    updated_at                  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    title               TEXT NOT NULL,
    run_dir             TEXT NOT NULL,
    candidate_id        TEXT,
    variant_or_baseline TEXT NOT NULL,
    phase               INTEGER NOT NULL,
    config_json         TEXT NOT NULL,
    seed                INTEGER NOT NULL,
    d_model             INTEGER NOT NULL,
    n_layers            INTEGER NOT NULL,
    params_effective    INTEGER,
    tokens_trained      INTEGER,
    status              TEXT NOT NULL DEFAULT 'pending',
    started_at          TEXT,
    ended_at            TEXT,
    wall_seconds        REAL,
    failure_reason      TEXT,
    created_at          TEXT NOT NULL,
    FOREIGN KEY (candidate_id) REFERENCES candidates(id)
);

CREATE TABLE IF NOT EXISTS metrics (
    run_id      INTEGER NOT NULL,
    step        INTEGER NOT NULL,
    dataset     TEXT NOT NULL,
    ppl         REAL,
    loss        REAL,
    eval_tokens INTEGER,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (run_id, step, dataset),
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE TABLE IF NOT EXISTS noise_floor (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    d_model         INTEGER NOT NULL,
    dataset         TEXT NOT NULL,
    primary_metric  TEXT NOT NULL,
    n_seeds         INTEGER NOT NULL,
    mean            REAL NOT NULL,
    std             REAL NOT NULL,
    threshold_4x    REAL NOT NULL,
    run_ids_json    TEXT,
    computed_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS comparisons (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id            TEXT NOT NULL,
    phase                   INTEGER NOT NULL,
    baseline_run_ids_json   TEXT NOT NULL,
    variant_run_ids_json    TEXT NOT NULL,
    primary_metric          TEXT NOT NULL,
    dataset                 TEXT NOT NULL,
    baseline_mean           REAL NOT NULL,
    variant_mean            REAL NOT NULL,
    effect_size             REAL NOT NULL,
    noise_floor_4x          REAL NOT NULL,
    effect_in_4x_units      REAL NOT NULL,
    gate_passed             INTEGER NOT NULL,
    notes                   TEXT,
    computed_at             TEXT NOT NULL,
    FOREIGN KEY (candidate_id) REFERENCES candidates(id)
);

CREATE TABLE IF NOT EXISTS findings (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id            TEXT NOT NULL UNIQUE,
    summary                 TEXT NOT NULL,
    mechanism_explanation   TEXT,
    effect_size_phase2      REAL,
    effect_size_phase3      REAL,
    status                  TEXT NOT NULL,
    validated_at            TEXT,
    FOREIGN KEY (candidate_id) REFERENCES candidates(id)
);

CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,
    kind         TEXT NOT NULL,
    message      TEXT NOT NULL,
    payload_json TEXT
);

CREATE TABLE IF NOT EXISTS eval_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER,           -- NULL if run not yet registered in DB
    step        INTEGER NOT NULL,  -- training step the checkpoint came from (-1 = final)
    model_size  INTEGER NOT NULL,  -- d_model (for cross-size ladder queries)
    eval_name   TEXT    NOT NULL,  -- e.g. "lambada", "blimp", "factual_cloze"
    metric      TEXT    NOT NULL,  -- e.g. "top1_acc", "accuracy", "mrr"
    value       REAL    NOT NULL,
    recorded_at TEXT    NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

CREATE INDEX IF NOT EXISTS idx_runs_status      ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_candidate   ON runs(candidate_id);
CREATE INDEX IF NOT EXISTS idx_runs_phase       ON runs(phase);
CREATE INDEX IF NOT EXISTS idx_metrics_run      ON metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_comparisons_candidate ON comparisons(candidate_id);
CREATE INDEX IF NOT EXISTS idx_findings_status  ON findings(status);
CREATE INDEX IF NOT EXISTS idx_events_ts        ON events(ts);
CREATE INDEX IF NOT EXISTS idx_eval_results_run ON eval_results(run_id, step);
CREATE INDEX IF NOT EXISTS idx_eval_results_eval ON eval_results(eval_name, metric);
"""


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Transient-write resilience
# ---------------------------------------------------------------------------
# On Windows, an external process (Defender/AV, backup, sync) can momentarily
# lock the WAL/-shm sidecar files, making SQLite return "attempt to write a
# readonly database" (SQLITE_READONLY) or "database is locked" (SQLITE_BUSY)
# for the duration of that lock. A single failed write was enough to abort a
# whole run at start_run (lost the e0061-e0065 replication batch, 2026-06-01).
# We ride out these transients with exponential backoff instead of failing.

# Substrings of OperationalError messages that indicate a TRANSIENT write
# failure worth retrying (vs a real schema/SQL error, which must surface).
_RETRYABLE_WRITE_ERRORS = ("readonly", "locked", "busy", "disk i/o error")

# Backoff schedule: 0.1, 0.2, 0.4, ... capped at 30s, 14 attempts.
# Total worst-case wait ~3.4 min — enough to outlast an AV/backup lock window
# (the observed 2026-06-01 outage spanned ~3 min) before giving up loudly.
_RETRY_ATTEMPTS = 14
_RETRY_BASE_SECONDS = 0.1
_RETRY_CAP_SECONDS = 30.0


def _is_retryable(exc: sqlite3.OperationalError) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in _RETRYABLE_WRITE_ERRORS)


def _retry_write(op):
    """Run a DB write op() with exponential backoff on transient lock/readonly
    errors. Re-raises immediately on non-transient errors and after the final
    attempt. Logs each backoff to stderr so unattended runs show what happened."""
    delay = _RETRY_BASE_SECONDS
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return op()
        except sqlite3.OperationalError as exc:
            if not _is_retryable(exc) or attempt == _RETRY_ATTEMPTS - 1:
                raise
            wait = min(delay, _RETRY_CAP_SECONDS)
            print(
                f"[db] transient write error ({exc}); retry "
                f"{attempt + 1}/{_RETRY_ATTEMPTS - 1} in {wait:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
            delay *= 2


class _RetryConnection(sqlite3.Connection):
    """sqlite3.Connection whose execute/executemany/commit transparently retry
    transient write failures. All db.py writes go through these methods on a
    connection from connect(), so wrapping them covers every write path."""

    def execute(self, *args, **kwargs):
        return _retry_write(lambda: sqlite3.Connection.execute(self, *args, **kwargs))

    def executemany(self, *args, **kwargs):
        return _retry_write(lambda: sqlite3.Connection.executemany(self, *args, **kwargs))

    def commit(self):
        return _retry_write(lambda: sqlite3.Connection.commit(self))


def connect(path: str = "db/loop_dev_ai.db") -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False, factory=_RetryConnection)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(path: str = "db/loop_dev_ai.db") -> None:
    """Create all tables and indexes if they do not exist. Idempotent."""
    conn = connect(path)
    conn.executescript(_DDL)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    return dict(row)


def _rows_to_list(rows) -> list[dict]:
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def add_candidate(conn: sqlite3.Connection, **fields) -> str:
    """Upsert a candidate row by id. Returns the id."""
    now = _now()
    fields.setdefault("created_at", now)
    fields["updated_at"] = now

    cols = list(fields.keys())
    placeholders = ", ".join(["?"] * len(cols))
    col_str = ", ".join(cols)
    update_str = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "id")

    sql = (
        f"INSERT INTO candidates ({col_str}) VALUES ({placeholders})"
        f" ON CONFLICT(id) DO UPDATE SET {update_str}"
    )
    conn.execute(sql, [fields[c] for c in cols])
    conn.commit()
    return fields["id"]


def update_candidate_status(conn: sqlite3.Connection, id: str, status: str) -> None:
    conn.execute(
        "UPDATE candidates SET status=?, updated_at=? WHERE id=?",
        (status, _now(), id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def start_run(conn: sqlite3.Connection, **fields) -> int:
    """Insert a new run with status='pending'. Returns run_id."""
    fields.setdefault("status", "pending")
    fields.setdefault("created_at", _now())

    cols = list(fields.keys())
    placeholders = ", ".join(["?"] * len(cols))
    col_str = ", ".join(cols)

    cur = conn.execute(
        f"INSERT INTO runs ({col_str}) VALUES ({placeholders})",
        [fields[c] for c in cols],
    )
    conn.commit()
    return cur.lastrowid


def mark_run_running(conn: sqlite3.Connection, run_id: int) -> None:
    conn.execute(
        "UPDATE runs SET status='running', started_at=? WHERE id=?",
        (_now(), run_id),
    )
    conn.commit()


def mark_run_done(
    conn: sqlite3.Connection,
    run_id: int,
    wall_seconds: float,
    tokens_trained: int,
) -> None:
    now = _now()
    conn.execute(
        """UPDATE runs
           SET status='done', ended_at=?, wall_seconds=?, tokens_trained=?
           WHERE id=?""",
        (now, wall_seconds, tokens_trained, run_id),
    )
    conn.commit()


def mark_run_failed(conn: sqlite3.Connection, run_id: int, reason: str) -> None:
    conn.execute(
        "UPDATE runs SET status='failed', ended_at=?, failure_reason=? WHERE id=?",
        (_now(), reason, run_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def add_metrics(
    conn: sqlite3.Connection,
    run_id: int,
    step: int,
    dataset_dict: dict[str, dict],
) -> None:
    """
    Bulk-insert one metrics row per (step, dataset).
    dataset_dict: {dataset_name: {ppl, loss, eval_tokens}, ...}
    Uses INSERT OR REPLACE so re-running an eval checkpoint is idempotent.
    """
    now = _now()
    rows = []
    for dataset, vals in dataset_dict.items():
        rows.append((
            run_id,
            step,
            dataset,
            vals.get("ppl"),
            vals.get("loss"),
            vals.get("eval_tokens"),
            now,
        ))
    conn.executemany(
        """INSERT OR REPLACE INTO metrics
           (run_id, step, dataset, ppl, loss, eval_tokens, recorded_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Noise floor
# ---------------------------------------------------------------------------

def record_noise_floor(
    conn: sqlite3.Connection,
    d_model: int,
    dataset: str,
    primary_metric: str,
    run_ids: list[int],
    ppl_values: list[float],
) -> int:
    """Compute mean/std/threshold from ppl_values and insert a noise_floor row."""
    n = len(ppl_values)
    if n == 0:
        raise ValueError("ppl_values must not be empty")
    mean = sum(ppl_values) / n
    variance = sum((v - mean) ** 2 for v in ppl_values) / n if n > 1 else 0.0
    std = math.sqrt(variance)
    threshold_4x = 4.0 * std

    cur = conn.execute(
        """INSERT INTO noise_floor
           (d_model, dataset, primary_metric, n_seeds, mean, std, threshold_4x,
            run_ids_json, computed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            d_model, dataset, primary_metric, n,
            mean, std, threshold_4x,
            json.dumps(run_ids),
            _now(),
        ),
    )
    conn.commit()
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

def _get_best_metric(
    conn: sqlite3.Connection,
    run_ids: list[int],
    dataset: str,
    metric: str,
) -> float:
    """
    For each run_id get the final (highest-step) metric value, then average.
    'Best' = lowest ppl, so we take the min-step value at the last recorded step.
    We actually want the last checkpoint value — use MAX(step) per run.
    """
    values: list[float] = []
    for rid in run_ids:
        row = conn.execute(
            f"""SELECT {metric} FROM metrics
                WHERE run_id=? AND dataset=? AND step=(
                    SELECT MAX(step) FROM metrics WHERE run_id=? AND dataset=?
                )""",
            (rid, dataset, rid, dataset),
        ).fetchone()
        if row is None or row[0] is None:
            raise ValueError(
                f"No {metric} metric found for run_id={rid}, dataset={dataset}"
            )
        values.append(row[0])
    return sum(values) / len(values)


def record_comparison(
    conn: sqlite3.Connection,
    candidate_id: str,
    phase: int,
    baseline_run_ids: list[int],
    variant_run_ids: list[int],
    primary_metric: str,
    dataset: str,
    notes: str | None = None,
) -> int:
    """
    Pull last-checkpoint metrics for each run set, compute effect_size and
    effect_in_4x_units, evaluate gate, insert comparison row. Returns comparison_id.
    """
    baseline_mean = _get_best_metric(conn, baseline_run_ids, dataset, primary_metric)
    variant_mean = _get_best_metric(conn, variant_run_ids, dataset, primary_metric)

    # Positive effect_size means variant is better (lower ppl)
    effect_size = baseline_mean - variant_mean

    nf = get_noise_floor(conn, _get_d_model_for_runs(conn, baseline_run_ids), dataset)
    if nf is None:
        raise ValueError(
            f"No noise_floor row found for dataset={dataset}. "
            "Run record_noise_floor first."
        )
    threshold_4x = nf["threshold_4x"]
    effect_in_4x_units = effect_size / threshold_4x if threshold_4x > 0 else 0.0
    gate_passed = 1 if effect_in_4x_units >= 1.0 else 0

    cur = conn.execute(
        """INSERT INTO comparisons
           (candidate_id, phase, baseline_run_ids_json, variant_run_ids_json,
            primary_metric, dataset, baseline_mean, variant_mean,
            effect_size, noise_floor_4x, effect_in_4x_units, gate_passed,
            notes, computed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            candidate_id, phase,
            json.dumps(baseline_run_ids), json.dumps(variant_run_ids),
            primary_metric, dataset,
            baseline_mean, variant_mean,
            effect_size, threshold_4x, effect_in_4x_units, gate_passed,
            notes, _now(),
        ),
    )
    conn.commit()
    return cur.lastrowid


def _get_d_model_for_runs(conn: sqlite3.Connection, run_ids: list[int]) -> int:
    """Return d_model from the first run_id (all runs in a comparison share d_model)."""
    row = conn.execute(
        "SELECT d_model FROM runs WHERE id=?", (run_ids[0],)
    ).fetchone()
    if row is None:
        raise ValueError(f"run_id={run_ids[0]} not found")
    return row["d_model"]


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

def record_finding(
    conn: sqlite3.Connection,
    candidate_id: str,
    summary: str,
    mechanism_explanation: str,
    status: str,
    **effects: Any,
) -> None:
    """Upsert a finding row by candidate_id. Pass effect_size_phase2/3 as kwargs."""
    now = _now()
    validated_at = now if status in ("phase3_validated", "published") else None

    conn.execute(
        """INSERT INTO findings
           (candidate_id, summary, mechanism_explanation, effect_size_phase2,
            effect_size_phase3, status, validated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(candidate_id) DO UPDATE SET
               summary=excluded.summary,
               mechanism_explanation=excluded.mechanism_explanation,
               effect_size_phase2=excluded.effect_size_phase2,
               effect_size_phase3=excluded.effect_size_phase3,
               status=excluded.status,
               validated_at=excluded.validated_at""",
        (
            candidate_id, summary, mechanism_explanation,
            effects.get("effect_size_phase2"),
            effects.get("effect_size_phase3"),
            status, validated_at,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_next_candidate(conn: sqlite3.Connection) -> dict | None:
    """Return the highest-priority pending candidate (lowest priority number = first)."""
    row = conn.execute(
        "SELECT * FROM candidates WHERE status='pending' ORDER BY priority ASC LIMIT 1"
    ).fetchone()
    return _row_to_dict(row)


def get_noise_floor(
    conn: sqlite3.Connection, d_model: int, dataset: str
) -> dict | None:
    """Return the most-recently computed noise_floor row for (d_model, dataset)."""
    row = conn.execute(
        """SELECT * FROM noise_floor
           WHERE d_model=? AND dataset=?
           ORDER BY computed_at DESC LIMIT 1""",
        (d_model, dataset),
    ).fetchone()
    return _row_to_dict(row)


def log_event(
    conn: sqlite3.Connection,
    kind: str,
    message: str,
    payload: Any = None,
) -> None:
    conn.execute(
        "INSERT INTO events (ts, kind, message, payload_json) VALUES (?, ?, ?, ?)",
        (_now(), kind, message, json.dumps(payload) if payload is not None else None),
    )
    conn.commit()


def add_eval_results(
    conn: sqlite3.Connection,
    run_id: int | None,
    step: int,
    model_size: int,
    results: dict[str, dict],
) -> None:
    """
    Bulk-insert eval_cheap results into eval_results.

    results: {eval_name: {metric_name: value, ...}}
    Non-numeric values (e.g. source="fallback") are silently skipped.
    Non-finite values (nan/inf) are also skipped: sqlite binds float('nan') as
    SQL NULL, which would otherwise trip the value NOT NULL constraint and abort
    the whole insert — taking down an otherwise-healthy run at checkpoint time
    just because one eval item produced a degenerate (e.g. -inf logprob) value.
    Uses INSERT OR REPLACE so re-running an eval is idempotent.
    """
    now = _now()
    rows = []
    for eval_name, metrics in results.items():
        for metric, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue  # skip non-numeric fields like "source"
            if not math.isfinite(value):
                continue  # skip nan/inf (sqlite stores nan as NULL -> NOT NULL error)
            rows.append((run_id, step, model_size, eval_name, metric, float(value), now))

    conn.executemany(
        """INSERT OR REPLACE INTO eval_results
           (run_id, step, model_size, eval_name, metric, value, recorded_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()


def leaderboard(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    """
    All done runs with their best (lowest) fineweb_edu_val ppl, sorted ascending.
    Joined with comparisons to surface Δ vs matched baseline where available.
    """
    rows = conn.execute(
        """SELECT
               r.id        AS run_id,
               r.title,
               r.candidate_id,
               r.variant_or_baseline,
               r.phase,
               r.d_model,
               r.seed,
               r.wall_seconds,
               r.tokens_trained,
               m.ppl       AS best_ppl,
               m.step      AS best_step
           FROM runs r
           JOIN metrics m ON m.run_id = r.id
           WHERE r.status = 'done'
             AND m.dataset = 'fineweb_edu_val'
             AND m.step = (
                 SELECT MAX(step) FROM metrics
                 WHERE run_id = r.id AND dataset = 'fineweb_edu_val'
             )
           ORDER BY m.ppl ASC
           LIMIT ?""",
        (limit,),
    ).fetchall()
    return _rows_to_list(rows)


def progress_summary(conn: sqlite3.Connection) -> dict:
    """Aggregate counts used by write_progress_md."""
    def scalar(sql, *args):
        row = conn.execute(sql, args).fetchone()
        return row[0] if row else 0

    return {
        "total_candidates": scalar("SELECT COUNT(*) FROM candidates"),
        "pending_candidates": scalar(
            "SELECT COUNT(*) FROM candidates WHERE status='pending'"
        ),
        "active_candidates": scalar(
            "SELECT COUNT(*) FROM candidates WHERE status='active'"
        ),
        "killed_candidates": scalar(
            "SELECT COUNT(*) FROM candidates WHERE status='killed'"
        ),
        "phase2_promising": scalar(
            "SELECT COUNT(*) FROM candidates WHERE status='phase2_promising'"
        ),
        "phase3_validated": scalar(
            "SELECT COUNT(*) FROM candidates WHERE status='phase3_validated'"
        ),
        "total_runs": scalar("SELECT COUNT(*) FROM runs"),
        "done_runs": scalar("SELECT COUNT(*) FROM runs WHERE status='done'"),
        "running_runs": scalar("SELECT COUNT(*) FROM runs WHERE status='running'"),
        "failed_runs": scalar("SELECT COUNT(*) FROM runs WHERE status='failed'"),
        "total_findings": scalar("SELECT COUNT(*) FROM findings"),
        "validated_findings": scalar(
            "SELECT COUNT(*) FROM findings WHERE status='phase3_validated'"
        ),
        "noise_floor_rows": scalar("SELECT COUNT(*) FROM noise_floor"),
    }


def findings_table(conn: sqlite3.Connection) -> list[dict]:
    """All findings joined with candidate name."""
    rows = conn.execute(
        """SELECT f.*, c.name AS candidate_name, c.category, c.novelty_rating
           FROM findings f
           JOIN candidates c ON c.id = f.candidate_id
           ORDER BY f.status, f.effect_size_phase3 ASC""",
    ).fetchall()
    return _rows_to_list(rows)
