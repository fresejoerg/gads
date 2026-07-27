---
name: "QRData hospital treatment (causal effect, confounded by severity) [dial D5]"
datasets:
  - qrdata/hospital_treatment.csv
target_column: days
domain: healthcare
recipe_id: causal_effect.observational.dowhy.mechanized
taxonomy:
  intent: causal
  task: [causal.effect_estimation]
  modality: [tabular]
  domain: healthcare
  domain_detail: "hospital treatment effect (QRData)"
  deliverable: [estimate]
  validation: [causal_identification]

---
Does the new hospital treatment reduce the number of days until discharge?

Dataset: 80 patients across two hospitals.
- TREATMENT column: `treatment` (1 = received the new treatment, 0 = standard care)
- OUTCOME column: `days` (days hospitalized until discharge)
- CONFOUNDER: `severity` (pre-treatment illness severity) — sicker patients are more
  likely to be treated AND stay longer, so adjusting for severity is essential.
- `hospital` is a site indicator that affects treatment assignment only through the
  severity mix of its patients — EXCLUDE it from the adjustment set; adjust for
  `severity` only.

Estimate the average treatment effect (ATE) of `treatment` on `days`, adjusting for
`severity`, and report it alongside both refutation checks.
