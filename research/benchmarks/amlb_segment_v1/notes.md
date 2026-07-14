# amlb_segment_v1 — provenance & tolerance rationale

**Established:** 2026-07-10, cloud mode, deterministic recipe (`tabular_automl.autogluon.deterministic`).
**Dataset:** OpenML data_id 40984 ("segment", the AMLB task) — 2,310 instances, 19 numeric
features, 7 perfectly balanced region classes (target `class`). Fetched via
`sklearn.datasets.fetch_openml` → `~/datasets/amlb/segment.csv`.

This is the first **multiclass** benchmark — it exercises the recipe's `f1_macro` branch and
the non-calibrated prediction path (threshold calibration is binary-only), neither of which the
fraud or adult benchmarks touch.

## Reference runs

| Mode | Project | Notes |
|---|---|---|
| cloud | `11f012e1` | 3/3 tasks ✓ (escalation churn from gemini 503s + one watchdog re-queue of the Synthesizer; all absorbed). macro-F1 **0.9281**, naive_baseline 1/7. |
| local | `f4e64460` | 3/3 tasks ✓ (gemma-4-12b, 0 escalations). test_score **bitwise-identical** to the cloud run: `0.9281148405938683`. |

## Tolerance rationale

- `naive_baseline` exact — 1/7 by class balance.
- `test_score` exact (`tol 0.0`) since 2026-07-10: cloud and local runs produced the identical
  16-digit float under the deterministic portfolio (same policy as amlb_adult_v1).
