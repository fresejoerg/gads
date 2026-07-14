---
name: "QRData online classroom (causal effect of course format)"
datasets:
  - qrdata/online_classroom.csv
target_column: falsexam
domain: education
recipe_id: causal_effect.observational.dowhy
---
Did moving students to an online-only classroom format change their final exam
performance?

Dataset: 323 students assigned to face-to-face, blended, or online-only course formats.
- TREATMENT column: `format_ol` (1 = online-only format, 0 = otherwise)
- OUTCOME column: `falsexam` (final exam score)
- `format_blended` is the indicator of the ALTERNATIVE treatment arm (blended format),
  not a confounder — EXCLUDE it from the adjustment set.
- The demographic columns (`gender`, `asian`, `black`, `hawaiian`, `hispanic`,
  `unknown`, `white`) are pre-treatment student characteristics — select the
  confounders among them.

Estimate the average treatment effect (ATE) of `format_ol` on `falsexam` and report it
alongside both refutation checks.
