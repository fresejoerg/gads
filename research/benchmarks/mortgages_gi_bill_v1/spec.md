---
name: "GI Bill mortgage-subsidy eligibility effect on home ownership"
datasets:
  - causaldata/mortgages_gi_bill.csv
disable_recipes: true
taxonomy:
  intent: causal
  task: [causal.effect_estimation]
  modality: [tabular]
  domain: finance
  domain_detail: "mid-century US GI Bill mortgage-subsidy eligibility, by birth-cohort cutoff (Fetter 2013)"
  deliverable: [estimate]
  validation: [causal_identification]
---
US veterans of WWII or the Korean War became eligible for mortgage subsidies
under mid-century GI Bill legislation. Eligibility for these veteran benefits was
determined by birth-cohort cutoffs tied to military service windows.

Dataset: 214,144 individual records (US Census).
- `bpl`: state of birth
- `qob`: quarter of birth
- `qob_minus_kw`: quarter of birth, re-centered so that 0 and above means "eligible for
  the mortgage subsidy" based on service-window timing
- `nonwhite`: 1 if the person is recorded as nonwhite, 0 otherwise
- `vet_wwko`: 1 if the person is a veteran of WWII or the Korean War, 0 otherwise
- `home_ownership`: 1 if the person owns their home, 0 otherwise

Did eligibility for the GI Bill mortgage subsidy increase home ownership?
