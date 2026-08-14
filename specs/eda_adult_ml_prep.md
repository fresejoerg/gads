---
name: "Adult Census — EDA & modelling preparation"
datasets:
  - amlb/adult.csv
target_column: class
recipe_id: tabular_eda.descriptive.standard
taxonomy:
  intent: descriptive
  task: [eda]
  modality: [tabular]
  domain: public_sector
  domain_detail: "US census income extract (Adult / UCI)"
  deliverable: [report, dataset]
  validation: [data_quality]

---
Perform a comprehensive exploratory data analysis of the Adult census extract, and
recommend how the data should be prepared before a model is trained on it.

The dataset describes 1994 US census respondents. `class` is the outcome of interest
(whether the person earns more than $50K) — treat it as the modelling target, and leave it
untransformed.

Characterise every column, assess data quality (missing values, duplicates, outliers,
unusable columns), show the distributions and the relationships between fields, and produce
a transformation manifest recording how each field should be imputed, scaled and encoded.
The transformed data is intended for a supervised machine-learning model, so also recommend
how the dataset should be split into training, validation and test partitions.
