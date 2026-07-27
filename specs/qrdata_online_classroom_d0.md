---
name: "QRData online classroom (causal effect of course format) [dial D0]"
datasets:
  - qrdata/online_classroom.csv
disable_recipes: true
taxonomy:
  intent: causal
  task: [causal.effect_estimation]
  modality: [tabular]
  domain: education
  domain_detail: "online classroom treatment effect (QRData)"
  deliverable: [estimate]
  validation: [causal_identification]

---
Did moving students to an online-only classroom format change their final exam
performance?

Dataset: 323 students assigned to face-to-face, blended, or online-only course formats.
- `format_ol`: 1 = online-only format
- `format_blended`: 1 = blended format
- `falsexam`: final exam score
- `gender`, `asian`, `black`, `hawaiian`, `hispanic`, `unknown`, `white`: student
  demographic indicators
