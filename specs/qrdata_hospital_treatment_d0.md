---
name: "QRData hospital treatment (causal effect, confounded by severity) [dial D0]"
datasets:
  - qrdata/hospital_treatment.csv
disable_recipes: true
---
Does the new hospital treatment reduce the number of days until discharge?

Dataset: 80 patients across two hospitals.
- `treatment`: 1 = received the new treatment, 0 = standard care
- `days`: days hospitalized until discharge
- `severity`: illness severity measured at admission
- `hospital`: which hospital the patient was admitted to
