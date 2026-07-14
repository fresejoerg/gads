# amlb_adult_v1 — provenance & tolerance rationale

**Established:** 2026-07-10, cloud mode, deterministic recipe (`tabular_automl.autogluon.deterministic`).
**Dataset:** OpenML data_id 1590 ("adult", the AMLB task) — 48,842 census rows, binary target
`class` ('<=50K'/'>50K', ~24% positive). Fetched via `sklearn.datasets.fetch_openml` →
`~/datasets/amlb/adult.csv`. No `sample_rows` cap (under the 50K threshold — full data).

## Reference runs

| Mode | Project | Notes |
|---|---|---|
| cloud | `75fba428` | 3/3 tasks ✓ (with intra-tier escalations through gemini 503s / a timeout — includes the first live `gpt-5.6-terra` completion). ROC-AUC **0.9311**, naive_baseline 0.7607. |
| local | `32952d0c` | 3/3 tasks ✓ (gemma-4-12b, 0 escalations). test_score **bitwise-identical** to the cloud run: `0.931129925396338`. |

An earlier attempt (`52406af9`, cancelled) exposed the label-dtype defect in the recipe
calibration pattern and the `gads_calibrate_threshold` helper — see journal 2026-07-10. This
benchmark was the first with string class labels.

## Tolerance rationale

- `naive_baseline` exact — pure function of the full dataset.
- `test_score` exact (`tol 0.0`) since 2026-07-10: cloud and local runs produced the identical
  15-digit float under the deterministic portfolio, so any deviation is a real regression.
  (The original ±0.005 was a placeholder pending exactly this repeat-run verification.)
- External anchor: AMLB's published AutoGluon results on adult are ~0.93 AUC under 10-fold CV;
  ours is a seeded 80/20 stratified holdout — same ballpark, different protocol; not directly
  comparable to the third decimal.
