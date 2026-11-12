# Run scorecards

One JSON file per GADS execution, `<project_id>.json`, produced by the code-only run
scorecard in `src/gads/eval/` (no LLM-as-judge). Written automatically at the end of every
run (completed, halted, cancelled or failed) and by the CLI; a re-score overwrites the file.

```bash
PYTHONPATH=src uv run python scripts/eval_run.py <project_id> [--failures] [--show]
PYTHONPATH=src uv run python scripts/eval_run.py --all --parity    # backfill + ledger parity
```

## Reading a scorecard

Every field is defined in the data dictionary, [SCHEMA.md](SCHEMA.md). The essentials:

| Field | Meaning |
|---|---|
| `verdict` | `pass` iff every critical check passes; `fail` if any critical check fails; `inconclusive` if none fails but some lack evidence; `not_evaluable` for runs in progress, never started, or whose workflow rows are gone |
| `critical` | pass / fail / not_evaluable counts over the critical checks: run reached final reporting, and on the final attempt every node executed and met its contract's `required_metrics` and `required_variables` |
| `dimensions` | the same counts for every check, by dimension (execution, contract, integrity, deliverables, benchmark) |
| `benchmark` | present when the run's spec is a benchmark spec: the `score_benchmark.py` checks, a separate verdict that does not affect `verdict` |
| `run.void_suspected` | the run stopped mid-way with no halt/cancel record (the backend died); a harness failure, not evidence about the model |
| `attribution` | pass@model, computed by the same function as `research/dial_ledger.jsonl` |
| `usage` | tokens and cost; `cost_complete: false` means some calls had no price (local model, unpriced providers), so `cost_usd` is a lower bound |
| `nodes` | per final-attempt node: who did the work (`invocation`: model / native_fallback / cloud_fallback / native_primary), whether its contract was recipe-curated or Planner-drafted, retries, check counts |
| `prior_attempts` | superseded attempts and which node failed in each; not scored |
| `llm_judgments` | Critique / CompletenessVerifier activity and the dial ledger's `outcome`. LLM judgments, recorded for context only, never part of the verdict |
| `checks` | every individual check with its status and the evidence behind it |

Scorecards with different `rubric_version` values were produced by different instruments
and should not be compared directly.
