---
name: "California Proposition 99 effect on cigarette sales"
datasets:
  - qrdata/prop99_cigarette_sales.csv
disable_recipes: true
taxonomy:
  intent: causal
  task: [causal.effect_estimation]
  modality: [tabular]
  domain: healthcare
  domain_detail: "state tobacco-tax policy effect on cigarette consumption"
  deliverable: [estimate]
  validation: [causal_identification]
---
In 1988 California passed Proposition 99, a tobacco-control law that raised
cigarette taxes and restricted sales and advertising. Other US states did not pass a
comparable law at the same time.

Dataset: annual state-level records, 39 states, 1970-2000.
- `state`, `year`: state identifier and year
- `cigsale`: cigarette sales (packs per capita)
- `california`: 1 if the record is California, 0 otherwise
- `after_treatment`: 1 if the year is 1989 or later, 0 otherwise
- `lnincome`, `beer`, `age15to24`, `retprice`: state-level economic and demographic covariates

Did Proposition 99 reduce cigarette sales in California?
