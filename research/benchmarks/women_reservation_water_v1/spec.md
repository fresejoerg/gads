---
name: "Political reservation for women effect on water infrastructure"
datasets:
  - qrdata/women_reservation_water.csv
disable_recipes: true
taxonomy:
  intent: causal
  task: [causal.effect_estimation]
  modality: [tabular]
  domain: public_sector
  domain_detail: "randomized political-reservation policy effect on local infrastructure investment (West Bengal)"
  deliverable: [estimate]
  validation: [causal_identification]
---
Since the mid-1990s, India has randomly reserved one-third of village
council leadership positions (Gram Panchayat, or GP) for women. This dataset is from a
study of GPs in West Bengal; two villages were sampled from each GP.

Dataset: 322 village-level records.
- `GP`: identifier for the Gram Panchayat
- `village`: identifier for the village
- `reserved`: 1 if the GP's leadership position was reserved for a woman, 0 otherwise
- `female`: 1 if the GP actually had a female leader, 0 otherwise
- `irrigation`: count of new or repaired irrigation facilities in the village since the
  policy started
- `water`: count of new or repaired drinking-water facilities in the village since the
  policy started

Did the reservation policy change the number of new or repaired drinking-water
facilities?
