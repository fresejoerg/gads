---
name: "Bayesian Causal Effect of Transaction Amount on Fraud"
datasets:
  - creditcard.csv
target_column: Class
domain: financial fraud
recipe_id: causal_effect.bayesian.pymc
taxonomy:
  intent: causal
  task: [causal.effect_estimation]
  modality: [tabular]
  domain: finance
  domain_detail: "financial fraud, Bayesian estimation (PyMC)"
  deliverable: [estimate]
  validation: [causal_identification, statistical_inference]

---
Using Bayesian inference, estimate whether transaction amount causally affects
fraud probability, with full uncertainty quantification.

Dataset: 284,807 credit card transactions.
- `Amount`: transaction value in currency units (continuous — the treatment of interest)
- `Class`: 0 = legitimate, 1 = fraud (binary outcome, severely imbalanced ~0.17%)
- `V1`–`V28`: PCA-anonymised behavioural features that may confound the relationship
- `Time`: temporal column — not a confounder

Provide the posterior distribution of the causal effect, a 94% HDI,
and the probability that the effect is positive. Compare with the naive
unadjusted association.
