"""aah rung-2 grounding: the clean star schema, as DuckDB views over the raw CSVs.

Self-contained (only needs duckdb). The 6 raw messy CSVs (u, hab, evt, subs, spend,
ref) must sit in `data_dir` alongside star.sql. `connect()` registers them as the raw
tables and layers the dim_/fct_ views on top — the exact modelling the analytics
engineer does. The agent then writes SQL over the clean views, never the raw tables.

Provenance: ported from d-n-ust/ai-analytics-harness (MIT). See research/harness/aah/.
"""
from __future__ import annotations
import re
from pathlib import Path
import duckdb

RAW_TABLES = ("u", "hab", "evt", "subs", "spend", "ref")
STAR_TABLES = (
    "dim_users", "dim_habits", "fct_value_moments", "fct_reminders",
    "fct_subscriptions", "fct_marketing_spend", "fct_referrals",
)


def _statements(sql: str):
    """Yield statements from a ;-separated script, stripping -- line comments first
    so semicolons inside comments don't split a statement."""
    no_comments = "\n".join(re.sub(r"--.*$", "", ln) for ln in sql.splitlines())
    for chunk in no_comments.split(";"):
        body = chunk.strip()
        if body:
            yield body


def connect(data_dir: str = ".") -> duckdb.DuckDBPyConnection:
    """Return an in-memory DuckDB connection with the raw tables loaded and the
    star views (dim_*, fct_*, and the internal agg_* marts) created."""
    d = Path(data_dir)
    con = duckdb.connect()
    for t in RAW_TABLES:
        csv = d / f"{t}.csv"
        con.execute(
            f"CREATE TABLE {t} AS SELECT * FROM read_csv_auto('{csv}', sample_size=-1)"
        )
    for stmt in _statements((d / "star.sql").read_text()):
        con.execute(stmt)
    return con


if __name__ == "__main__":
    con = connect()
    print("raw + star ready. star tables:")
    for t in STAR_TABLES:
        n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        print(f"  {t:22s} {n:>8,d} rows")
