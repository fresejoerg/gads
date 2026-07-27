---
name: "QRData IHDP-1 (causal effect, semi-synthetic) [dial D1]"
datasets:
  - qrdata/ihdp_1.csv
target_column: y
domain: infant health and development program
disable_recipes: true
taxonomy:
  intent: causal
  task: [causal.effect_estimation]
  modality: [tabular]
  domain: healthcare
  domain_detail: "infant health & development program, replicate 1 (IHDP)"
  deliverable: [estimate]
  validation: [causal_identification]

---
What is the average causal effect of the IHDP home-visit intervention on the child's
cognitive test score?

Dataset: 747 infants (IHDP semi-synthetic benchmark, replicate 1).
- TREATMENT column: `treatment` (1 = received specialist home visits, 0 = control)
- OUTCOME column: `y` (continuous cognitive test score)
- `x1`–`x25` are pre-treatment child and mother characteristics — select the
  confounders among them.

Estimate the average treatment effect (ATE) of `treatment` on `y`, adjusting for
pre-treatment confounders, and report it alongside both refutation checks.
