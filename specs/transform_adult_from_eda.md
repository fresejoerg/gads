---
name: "Adult census — apply EDA transformations"
artifacts_from: 5b30faa6-c657-48ed-99fd-2046bd50821e
recipe_id: tabular_transform.apply_manifest
taxonomy:
  intent: descriptive
  task: [data_preparation]
  modality: [tabular]
  domain: public_sector
  domain_detail: "US census income extract (Adult / UCI)"
  deliverable: [dataset]
  validation: [data_quality]

---
Apply the transformation manifest produced by the earlier exploratory analysis and write the
transformed dataset.

The upstream run's artifacts are linked under `upstream/` — including
`eda_transformations.meta.json`, which records the per-column imputation, scaling and encoding
decisions along with the recommended train/validation/test split, and `adult.csv`, the file it
describes.

Apply the manifest exactly as recorded and verify what was written: row and column counts per
partition, no residual nulls in imputed columns, and no overlap between partitions.

Note: this spec declares no `datasets:` of its own — the data it transforms comes from the
upstream run, which is the point of `artifacts_from`.
