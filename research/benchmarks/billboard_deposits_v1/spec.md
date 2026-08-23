---
name: "Billboard advertising effect on bank deposits (Porto Alegre)"
datasets:
  - qrdata/billboard_deposits.csv
disable_recipes: true
taxonomy:
  intent: causal
  task: [causal.effect_estimation]
  modality: [tabular]
  domain: marketing
  domain_detail: "outdoor advertising campaign effect on customer bank deposits"
  deliverable: [estimate]
  validation: [causal_identification]
---
A bank ran a billboard advertising campaign in Porto Alegre, Brazil, but not
in the nearby city of Florianopolis. Deposit records were collected before and after
the campaign in both cities.

Dataset: 4,600 customer-level records.
- `deposits`: average bank deposits per customer, in Brazilian Reais
- `poa`: 1 if the record is from Porto Alegre, 0 if from Florianopolis
- `jul`: 1 if the record is from July (after the campaign), 0 if from May (before)

Did the advertising campaign increase customer deposits in Porto Alegre?
