---
name: "Growth-mindset intervention effect on academic achievement"
datasets:
  - qrdata/learning_mindset_achievement.csv
disable_recipes: true
taxonomy:
  intent: causal
  task: [causal.effect_estimation]
  modality: [tabular]
  domain: education
  domain_detail: "growth-mindset seminar effect on standardized academic achievement (National Study of Learning Mindsets)"
  deliverable: [estimate]
  validation: [causal_identification]
---
The National Study of Learning Mindsets ran a seminar in US public high
schools intended to instill a "growth mindset" in students, then followed up on their
academic performance.

Dataset: 10,391 student records (simulated to match the real study's structure).
- `intervention`: 1 if the student received the seminar, 0 otherwise
- `achievement_score`: standardized academic achievement score, measured afterward
- `schoolid`: the student's school
- `success_expect`: self-reported expectation of future success, measured before the seminar
- `ethnicity`, `gender`, `frst_in_family`: student demographics
- `school_urbanicity`, `school_mindset`, `school_achievement`, `school_ethnic_minority`,
  `school_poverty`, `school_size`: school-level characteristics

Did the growth-mindset seminar improve students' academic achievement?
