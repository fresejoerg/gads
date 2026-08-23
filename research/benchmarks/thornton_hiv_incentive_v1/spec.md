---
name: "Cash incentive effect on learning HIV test results"
datasets:
  - causaldata/thornton_hiv_incentive.csv
disable_recipes: true
taxonomy:
  intent: causal
  task: [causal.effect_estimation]
  modality: [tabular]
  domain: healthcare
  domain_detail: "randomized cash-incentive experiment for HIV test result pickup, rural Malawi (Thornton 2008)"
  deliverable: [estimate]
  validation: [causal_identification]
---
In rural Malawi, people were offered an HIV test and then randomly assigned
a cash voucher (of varying amounts, including zero) redeemable when they came back to
a testing center to learn their result.

Dataset: 4,820 individual records.
- `villnum`: village identifier
- `got`: 1 if the person came back to learn their HIV test result, 0 otherwise
- `any`: 1 if the person was assigned any cash voucher (nonzero), 0 otherwise
- `tinc`: the total incentive amount assigned
- `distvct`: distance in kilometers from the person's home to the nearest testing center
- `age`: the person's age
- `hiv2004`: the person's actual HIV test result

Did being offered a cash incentive increase the likelihood that someone came back to
learn their HIV status?
