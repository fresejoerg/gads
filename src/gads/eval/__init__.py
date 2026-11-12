"""Code-only run evaluation: one scorecard per GADS execution (no LLM-as-judge).

    from gads.eval.scorecard import evaluate_run, write_scorecard, score_and_write

Kept import-light on purpose: gads.eval.benchmark is stdlib-only and is imported by
scripts/score_benchmark.py without a database; only gads.eval.record touches Postgres,
and only inside load_run().
"""
