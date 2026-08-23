---
name: "Contaminated water supply effect on cholera deaths (John Snow, 1854)"
datasets:
  - causaldata/snow_cholera_water.csv
disable_recipes: true
taxonomy:
  intent: causal
  task: [causal.effect_estimation]
  modality: [tabular]
  domain: healthcare
  domain_detail: "John Snow's 1854 London cholera investigation, by water-pump supplier"
  deliverable: [estimate]
  validation: [causal_identification]
---
In 1854, London's water was supplied by several private companies, drawing
from different points on the Thames — some of which were contaminated with sewage,
others not. A cholera outbreak occurred that year.

Dataset: 4 aggregated records.
- `year`: year of the record
- `supplier`: which water supplier(s) served the area ("Non-Lambeth Only" = drew
  contaminated water; "Lambeth + Others" is a mixed group; entries also cover a later
  year after one supplier moved its water intake to a cleaner source)
- `treatment`: a label describing the water-source condition for that record
  ("Dirty", "Mix Dirty and Clean", etc.)
- `deathrate`: deaths per 10,000 (1851 population) in that area

Did drawing water from a contaminated source increase the death rate?
