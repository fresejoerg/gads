---
name: "UK Parliament membership effect on personal wealth"
datasets:
  - qrdata/mps_wealth.csv
disable_recipes: true
taxonomy:
  intent: causal
  task: [causal.effect_estimation]
  modality: [tabular]
  domain: public_sector
  domain_detail: "UK MPs 1950-1970, personal wealth at death vs. election outcome"
  deliverable: [estimate]
  validation: [causal_identification]
---
Does holding political office increase a politician's personal wealth?
Researchers collected records on UK parliamentary candidates who ran in general
elections between 1950 and 1970, including their wealth at the time of death.

Dataset: 427 candidate records.
- `surname`, `firstname`: candidate name
- `party`: `labour` or `tory`
- `ln.gross`, `ln.net`: log gross and log net personal wealth at death
- `yob`, `yod`: year of birth and year of death
- `margin`: the candidate's margin of victory (or defeat) in the election, as a vote share
- `margin.pre`: the candidate's party's margin in the previous election
- `region`: electoral region

For Tory (Conservative) candidates, does winning a seat in Parliament change personal
wealth?
