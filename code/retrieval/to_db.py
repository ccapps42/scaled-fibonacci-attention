"""
to_db.py -- land the retrieval-coverage json results into this project's db/fib.db.

The retrieval line (sweep.py / multitarget.py / marked.py) writes flat json for cheap
incremental/resumable saves during the GPU runs. This loader syncs those json files into
a `retrieval_results` table in fib.db so the coverage numbers are queryable alongside the
perplexity runs and comparators. Idempotent (INSERT OR REPLACE on a natural key) -- safe
to re-run after each sweep to pick up new cells. Does NOT touch the read-only LDA db.py;
it only creates its own table via CREATE TABLE IF NOT EXISTS.

Normalized to LONG format: one row per (source, mech, depth, seed, staggered, distance,
kind). acc is the copy accuracy at that distance; per-layer alphas/report kept as json.

  python code/retrieval/to_db.py                 # sync all three default json files
  python code/retrieval/to_db.py --files a.json  # sync specific files

Example queries afterwards:
  SELECT mech, distance, acc FROM retrieval_results
    WHERE source='sweep' AND kind='trained' AND depth=8 ORDER BY distance;
  SELECT depth, staggered, distance, acc FROM retrieval_results
    WHERE source='marked' ORDER BY depth, distance;
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).parent
_CODE = _HERE.parent
_LDA = Path(r"K:\projects\Loop_Dev_AI\code")
for p in (str(_CODE), str(_LDA)):
    if p not in sys.path:
        sys.path.insert(0, p)

import db as _db   # noqa: E402  (LDA helper: connect() gives WAL + retry connection)

PROJECT_ROOT = str(_CODE.parent)
DB_PATH = os.path.join(PROJECT_ROOT, "db", "fib.db")
RET_DIR = os.path.join(PROJECT_ROOT, "experiments", "retrieval")
DEFAULT_FILES = ["coverage_sweep.json", "multitarget_stagger.json", "marked_union.json"]

DDL = """
CREATE TABLE IF NOT EXISTS retrieval_results (
    source     TEXT NOT NULL,            -- sweep | multitarget | marked
    mech       TEXT NOT NULL,            -- sweep mechanism, or 'fib_blur' for multitarget/marked
    depth      INTEGER NOT NULL,
    seed       INTEGER NOT NULL,
    staggered  INTEGER NOT NULL,         -- 0/1; -1 = N/A (sweep)
    train_d    INTEGER NOT NULL,         -- the cell's TRAINED distance (model identity); -1 = trained on a set
    distance   INTEGER NOT NULL,         -- eval offset d
    kind       TEXT NOT NULL,            -- trained | control | target
    acc        REAL NOT NULL,
    alphas     TEXT,                     -- json: per-layer alphas / report
    peak_gb    REAL,
    src_file   TEXT,
    extra      TEXT,                     -- json: full original record
    PRIMARY KEY (source, mech, depth, seed, staggered, train_d, distance, kind)
);
"""


def _rows_from_sweep(rec):
    extra = json.dumps(rec)
    mech, depth, seed, td = rec["mech"], rec["depth"], rec["seed"], rec["d"]
    al = json.dumps(rec.get("report"))
    pk = rec.get("peakGB")
    rows = [("sweep", mech, depth, seed, -1, td, td, "trained", rec["acc"], al, pk, extra)]
    for cd, a in (rec.get("acc_control") or {}).items():
        rows.append(("sweep", mech, depth, seed, -1, td, int(cd), "control", a, al, pk, extra))
    return rows


def _rows_from_multi(rec, source):
    extra = json.dumps(rec)
    depth, seed = rec["depth"], rec["seed"]
    stag = 1 if rec.get("staggered") else 0
    al = json.dumps(rec.get("alphas"))
    pk = rec.get("peakGB")
    rows = []
    for d, a in (rec.get("accs") or {}).items():
        rows.append((source, "fib_blur", depth, seed, stag, -1, int(d), "target", a, al, pk, extra))
    for cd, a in (rec.get("controls") or {}).items():
        rows.append((source, "fib_blur", depth, seed, stag, -1, int(cd), "control", a, al, pk, extra))
    return rows


def _detect(records, fname):
    if any("acc_control" in r or ("mech" in r and "accs" not in r) for r in records):
        return "sweep"
    if "marked" in fname:
        return "marked"
    return "multitarget"


def load_file(path):
    records = json.load(open(path, encoding="utf-8"))
    fname = os.path.basename(path)
    source = _detect(records, fname)
    rows = []
    for rec in records:
        rows += _rows_from_sweep(rec) if source == "sweep" else _rows_from_multi(rec, source)
    # each row ends (..., peak_gb, extra); splice src_file in before extra
    rows = [r[:11] + (fname, r[11]) for r in rows]
    return source, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", default=DEFAULT_FILES,
                    help="json filenames under experiments/retrieval/ (or absolute paths)")
    ap.add_argument("--db", default=DB_PATH)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.db), exist_ok=True)
    conn = _db.connect(args.db)
    conn.executescript(DDL)
    conn.commit()

    total = 0
    for f in args.files:
        path = f if os.path.isabs(f) else os.path.join(RET_DIR, f)
        if not os.path.exists(path):
            print(f"  skip (missing): {path}")
            continue
        source, rows = load_file(path)
        conn.executemany(
            "INSERT OR REPLACE INTO retrieval_results "
            "(source,mech,depth,seed,staggered,train_d,distance,kind,acc,alphas,peak_gb,src_file,extra) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        conn.commit()
        total += len(rows)
        print(f"  {source:12s} {os.path.basename(path):24s} -> {len(rows):4d} rows")

    n = conn.execute("SELECT COUNT(*) FROM retrieval_results").fetchone()[0]
    by_src = conn.execute("SELECT source, COUNT(*) FROM retrieval_results GROUP BY source").fetchall()
    conn.close()
    print(f"synced {total} rows this run; table now holds {n} ({dict((r[0], r[1]) for r in by_src)}) -> {args.db}")


if __name__ == "__main__":
    main()
