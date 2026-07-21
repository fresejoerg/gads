---
name: "QRData IHDP-0 (causal effect, semi-synthetic) [dial D0]"
datasets:
  - qrdata/ihdp_0.csv
disable_recipes: true
---
What is the average causal effect of the IHDP home-visit intervention on the child's
cognitive test score?

Dataset: 747 infants (IHDP semi-synthetic benchmark, replicate 0).
- `treatment`: 1 = received specialist home visits, 0 = control
- `y`: continuous cognitive test score
- `x1`–`x25`: child and mother characteristics measured before the intervention
