---
name: "Social-network information effect on insurance take-up"
datasets:
  - causaldata/social_insurance_takeup.csv
disable_recipes: true
taxonomy:
  intent: causal
  task: [causal.effect_estimation]
  modality: [tabular]
  domain: finance
  domain_detail: "rural Chinese farmers, weather-insurance take-up experiment (Cai, De Janvry & Sadoulet 2015)"
  deliverable: [estimate]
  validation: [causal_identification]
---
Rice farmers in rural China were offered a new weather insurance product.
Researchers ran a two-round experiment varying how information about the product was
delivered, including which households were randomly given more intensive information
sessions.

Dataset: 1,409 household records.
- `address`, `village`: household's natural and administrative village
- `takeup_survey`: 1 if the household purchased insurance, 0 otherwise
- `age`, `agpop`, `male`, `literacy`: household head characteristics
- `ricearea_2010`: area of rice production
- `disaster_prob`: household's perceived probability of a disaster next year
- `risk_averse`: a risk-aversion measurement
- `default`: 1 if the household's assigned default was "buy," 0 if "don't buy"
- `intensive`: 1 if the household was assigned to a more intensive information session
- `pre_takeup_rate`: the village's insurance take-up rate before this round

Did the intensive information session increase insurance take-up?
