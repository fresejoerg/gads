---
name: "National Supported Work job-training program (LaLonde)"
datasets:
  - qrdata/jobs_lalonde_training.csv
disable_recipes: true
taxonomy:
  intent: causal
  task: [causal.effect_estimation]
  modality: [tabular]
  domain: hr_people
  domain_detail: "job training program effect on earnings/employment (LaLonde NSW study)"
  deliverable: [estimate]
  validation: [causal_identification]
---
Did the National Supported Work (NSW) job-training program improve economic
outcomes for the people who went through it?

Dataset: 2,570 people who were candidates for the NSW program.
- `t`: whether the person went through the training program (1) or not (0)
- `y`: an economic outcome measured after the program
- `x0`-`x16`: pre-program characteristics of each person (demographics, education, prior earnings)

Estimate the effect of the training program on the outcome for the people who
participated.
