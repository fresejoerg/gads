---
name: "Castle Doctrine self-defense law effect on homicide"
datasets:
  - causaldata/castle_doctrine_homicide.csv
disable_recipes: true
taxonomy:
  intent: causal
  task: [causal.effect_estimation]
  modality: [tabular]
  domain: public_sector
  domain_detail: "state self-defense ('stand your ground') law effect on homicide rate (Cheng & Hoekstra 2013)"
  deliverable: [estimate]
  validation: [causal_identification]
---
Between 2000 and 2010 more than 20 US states passed "Castle Doctrine" (also
called "stand your ground") laws, which expand the legal justification for using
lethal force in self-defense. States adopted these laws in different years; some
states never adopted one.

Dataset: state-year panel records, FBI Uniform Crime Reports.
- `year`, `sid`: year and state identifier
- `post`: 1 if the state had a Castle Doctrine law in effect that year, 0 otherwise
- `homicide`, `murder`, `robbery`, `assault`, `burglary`, `larceny`, `motor`: crime
  counts per 100,000 state population
- `unemployrt`, `poverty`: state economic conditions
- `blackm_15_24`, `whitem_15_24`: population shares by race/age group
- `l_income`, `l_police`, `l_prisoner`: log state income, police presence, prison
  population

Did adopting a Castle Doctrine law change the homicide rate?
