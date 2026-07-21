---
name: "QRData IHDP-0 (causal effect, semi-synthetic) [dial D5]"
datasets:
  - qrdata/ihdp_0.csv
target_column: y
domain: infant health and development program
recipe_id: causal_effect.observational.dowhy.mechanized
---
What is the average causal effect of the IHDP home-visit intervention on the child's
cognitive test score?

Dataset: 747 infants (IHDP semi-synthetic benchmark, replicate 0).
- TREATMENT column: `treatment` (1 = received specialist home visits, 0 = control)
- OUTCOME column: `y` (continuous cognitive test score)
- `x1`–`x25` are pre-treatment child and mother characteristics — select the
  confounders among them.

Estimate the average treatment effect (ATE) of `treatment` on `y`, adjusting for
pre-treatment confounders, and report it alongside both refutation checks.
