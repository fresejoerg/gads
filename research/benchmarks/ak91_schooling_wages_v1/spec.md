---
name: "Years of schooling effect on wages (compulsory-schooling natural experiment)"
datasets:
  - qrdata/ak91_schooling_wages.csv
disable_recipes: true
taxonomy:
  intent: causal
  task: [causal.effect_estimation]
  modality: [tabular]
  domain: education
  domain_detail: "Angrist-Krueger 1991 quarter-of-birth natural experiment on returns to schooling"
  deliverable: [estimate]
  validation: [causal_identification]
---
Does an additional year of schooling raise wages? US compulsory-schooling
law requires a child to have turned 6 by January 1 of the year they enter school, and
allows students to drop out once they turn 16 — so a child's birth quarter mechanically
affects how many years of schooling they are required to complete before they are
legally allowed to leave.

Dataset: 329,509 individual records (US Census).
- `log_wage`: the person's log wage
- `years_of_schooling`: completed years of schooling
- `year_of_birth`, `quarter_of_birth`: when the person was born
- `state_of_birth`: the person's state of birth

Estimate the effect of an additional year of schooling on wages.
