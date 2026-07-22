"""GADS bridge: generate the deterministic aah warehouse, export the 6 RAW tables
(mess intact) to CSV for GADS's datasets root, and compute the 25 gold answers over
the clean star. Run inside the aah clone's env (has faker/duckdb/pandas)."""
from __future__ import annotations
import json, shutil
from pathlib import Path
import duckdb

from data.generate import generate, OUT_PATH
from harness.warehouse import RAW_TABLES, STAR_SQL, _statements
from evaluation.gold import load_questions, compute_gold

DEST = Path("/home/joergf/datasets/aah")
DEST.mkdir(parents=True, exist_ok=True)

# 1. deterministic generation
counts = generate()
print("generated warehouse:", counts)

con = duckdb.connect(str(OUT_PATH))

# 2. export the 6 raw (messy) tables to CSV — the star is rebuilt downstream in-sandbox
for t in RAW_TABLES:
    out = DEST / f"{t}.csv"
    con.execute(f"COPY (SELECT * FROM {t}) TO '{out}' (HEADER, DELIMITER ',')")
    n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
    print(f"  exported {t:6s} -> {out.name} ({n} rows)")

# 3. build the star views (needed only to compute golds), then compute golds
for stmt in _statements(STAR_SQL.read_text()):
    con.execute(stmt)
golds = compute_gold(con)

# 4. attach question metadata (tier, grader, tolerance) for the benchmark expected.json
qmeta = {}
for q in load_questions():
    qmeta[q["id"]] = {
        "tier": q["tier"], "question": q["question"],
        "grader": q.get("grader", "numeric"),
        "tolerance": q.get("tolerance"),
        "gold": golds.get(q["id"]),
        "gold_keywords": q.get("gold_keywords"),
        "gold_diagnostic": q.get("gold_diagnostic"),
    }
(Path(__file__).parent / "_golds.json").write_text(json.dumps(qmeta, indent=2))
print("\nGOLDS:")
for qid, m in qmeta.items():
    print(f"  {qid:28s} {m['tier']:11s} {m['grader']:10s} gold={m['gold']}")
con.close()
print("\nwrote _golds.json and CSVs to", DEST)
