---
name: "New Jersey minimum wage increase effect on fast-food employment"
datasets:
  - qrdata/minwage_employment.csv
disable_recipes: true
taxonomy:
  intent: causal
  task: [causal.effect_estimation]
  modality: [tabular]
  domain: public_sector
  domain_detail: "state minimum-wage policy effect on fast-food employment (Card-Krueger design)"
  deliverable: [estimate]
  validation: [causal_identification]
---
In 1992 New Jersey (NJ) raised its minimum wage from $4.25 to $5.05/hour;
the neighboring state of Pennsylvania (PA) did not change its minimum wage.
Researchers surveyed fast-food restaurants in both states before and after the change.

Dataset: 358 restaurant-level records.
- `chain`: restaurant chain name
- `location`: NJ region (centralNJ/northNJ/shoreNJ/southNJ) or PA
- `wageBefore`, `wageAfter`: starting wage before and after the change
- `fullBefore`, `fullAfter`: number of full-time employees before and after
- `partBefore`, `partAfter`: number of part-time employees before and after

Did raising the minimum wage in NJ reduce full-time employment at these restaurants,
relative to what happened in PA over the same period?
