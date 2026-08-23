---
name: "Job-trainee program effect on earnings"
datasets:
  - qrdata/trainee_program_earnings.csv
disable_recipes: true
taxonomy:
  intent: causal
  task: [causal.effect_estimation]
  modality: [tabular]
  domain: hr_people
  domain_detail: "small trainee-program dataset, age is a plausible confounder"
  deliverable: [estimate]
  validation: [causal_identification]
---
Does a job-trainee program raise the earnings of people who go through it?

Dataset: 58 individual records.
- `trainees`: whether the person went through the trainee program (1) or not (0)
- `age`: the person's age
- `earnings`: the person's earnings

Note: trainees are noticeably younger, on average, than non-trainees in this data.

Estimate the effect of the trainee program on earnings for the people who
participated.
