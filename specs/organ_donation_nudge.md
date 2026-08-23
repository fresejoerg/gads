---
name: "Active-choice organ-donor registration effect"
datasets:
  - causaldata/organ_donation_nudge.csv
disable_recipes: true
taxonomy:
  intent: causal
  task: [causal.effect_estimation]
  modality: [tabular]
  domain: healthcare
  domain_detail: "California active-choice organ-donor registration policy change (Kessler & Roth 2014)"
  deliverable: [estimate]
  validation: [causal_identification]
---
In Q3 2011, California changed the wording of its organ-donor registration
question (on driver's-license applications) to an "active choice" format, requiring
applicants to explicitly answer yes or no rather than simply opt in. Other US states
did not make this change over the same period.

Dataset: 162 state-quarter records, Q4 2010 - Q1 2012.
- `State`: US state (California is the state that changed its wording)
- `Quarter`: calendar quarter, e.g. "Q42010"
- `Rate`: organ-donor registration rate for that state-quarter

Did switching to the active-choice wording change California's organ-donor
registration rate?
