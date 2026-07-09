# GADS Benchmark Repository

Specs with **expected results** — including intermediate steps and artifacts — used to quantify
run quality across routing modes and engine tiers. This is the measurement instrument for the
two research metrics (see `research/JOURNAL.md`): **reproducibility** and **methodological
appropriateness**.

## Layout

```
research/benchmarks/<benchmark_id>/
  spec.md          # the exact spec (self-contained copy; also runnable from specs/)
  expected.json    # machine-checkable expectations (schema below)
  notes.md         # provenance: which runs established the expectations, tolerance rationale
```

## Running & scoring

```bash
# 1. launch the benchmark spec (any mode)
curl -X POST localhost:8001/projects/from-spec -H "Content-Type: application/json" \
     -d '{"filename": "<spec>"}'
# 2. score the finished workspace against expectations
uv run python scripts/score_benchmark.py \
     --benchmark research/benchmarks/<benchmark_id> \
     --workspace /home/joergf/projects/MyLocalStack/data/workspaces/<project_id>
```

The scorer emits a scorecard over three dimensions and a machine-readable JSON line for
aggregation across runs/modes:

| Dimension | What it checks | Metric target |
|---|---|---|
| **reproducibility** | scalar metrics vs. canonical values (exact and tolerance bands) | identical results across consecutive runs |
| **completeness** | required artifacts exist; forbidden outputs absent | the run delivered the full latent plan |
| **methodology** | code-level checks on `workflow_execution.py` (required & forbidden patterns) | the realization matches the latent, correct plan |

## `expected.json` schema

```jsonc
{
  "benchmark_id": "...",
  "spec_file": "...",                    // filename under specs/
  "recipe_id": "...",                    // the latent plan's codified form (null if uncodified)
  "reference_runs": {"<mode>": "<project uuid>"},   // provenance
  "metrics": {                           // from metrics.json in the workspace
    "<name>": {
      "value": 0.0,                      // canonical value
      "exact": true,                     // must match bitwise (pure functions of the data)
      "tol": 0.0,                        // else: |observed - value| <= tol
      "note": "why this tolerance"
    }
  },
  "artifacts": {
    "required": ["file", ...],           // must exist in the workspace
    "forbidden_stdout": ["pattern", ...] // hallucination markers etc. (checked if logs present)
  },
  "tasks": {                             // intermediate-step expectations
    "expected_count": 3,
    "produces": {"<task match substr>": ["kernel var", ...]}
  },
  "methodology": {
    "required_patterns": ["regex", ...], // must appear in workflow_execution.py
    "forbidden_patterns": ["regex", ...] // must NOT appear
  }
}
```

## Conventions

- **A benchmark is frozen once referenced in a write-up** — fix mistakes by versioning
  (`fraud_autogluon_v2`), never by editing in place.
- Canonical metric values come from **verified reference runs**, cited in `notes.md` with
  project UUIDs and the commit the harness was at.
- If a metric can't be exact, the tolerance *must* be justified in `notes.md` — an unexplained
  tolerance is a reproducibility bug being hidden (see journal 2026-07-09 on `time_limit`).
- Intermediate expectations (`tasks.produces`) mirror the recipe DAG's `produces` lists — for
  uncodified benchmarks, write down the latent plan's steps explicitly.
