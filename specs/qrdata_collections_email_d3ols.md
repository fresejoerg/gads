---
name: "QRData collections email (causal effect with post-treatment traps) [dial D3-ols]"
datasets:
  - qrdata/collections_email.csv
target_column: payments
domain: debt collections marketing
recipe_id: causal_effect.observational.ols.directed
taxonomy:
  intent: causal
  task: [causal.effect_estimation]
  modality: [tabular]
  domain: marketing
  domain_detail: "debt-collections email treatment effect (QRData)"
  deliverable: [estimate]
  validation: [causal_identification]

---
Did sending the collections email cause customers to pay more of their debt?

Dataset: 5,000 customers of a collections operation.
- TREATMENT column: `email` (1 = received the collections email, 0 = not)
- OUTCOME column: `payments` (amount paid)
- CONFOUNDERS: `credit_limit` and `risk_score` ONLY — these are pre-treatment
  customer characteristics.
- `opened` (customer opened the email) and `agreement` (customer made a payment
  agreement) are POST-TREATMENT consequences of the email — they must be EXCLUDED
  from the adjustment set. Adjusting for them would bias the effect estimate.

Estimate the average treatment effect (ATE) of `email` on `payments`, adjusting for
`credit_limit` and `risk_score` only, and report it alongside both refutation checks.
