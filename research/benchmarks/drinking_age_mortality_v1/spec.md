---
name: "Legal drinking age effect on mortality"
datasets:
  - qrdata/drinking_age_mortality.csv
disable_recipes: true
taxonomy:
  intent: causal
  task: [causal.effect_estimation]
  modality: [tabular]
  domain: healthcare
  domain_detail: "legal minimum drinking age (21) effect on mortality around the age cutoff"
  deliverable: [estimate]
  validation: [causal_identification]
---
In the US the legal minimum drinking age is 21. Mortality records were
aggregated into narrow age bins spanning ages just below and just above 21.

Dataset: 51 age-bin records.
- `agecell`: average age of the bin
- `all`: deaths per 10,000 population, all causes
- `mva`: deaths per 10,000 population, motor vehicle accidents
- `suicide`: deaths per 10,000 population, suicide
- `alcohol`: an alcohol-consumption measure for the bin

Does turning 21 change the death rate from all causes?
