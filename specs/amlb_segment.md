---
name: "AMLB Image Segmentation (multiclass classification)"
datasets:
  - amlb/segment.csv
target_column: class
domain: image region classification
recipe_id: tabular_automl.autogluon.deterministic
---
Which image region type does each instance belong to?

Dataset: 2,310 instances (OpenML data_id 40984, the AMLB 'segment' task).
- `class`: 7 balanced region types (brickface, sky, foliage, cement, window, path, grass)
- 19 numeric features describing 3x3 pixel regions (centroid, color moments, edge densities)

Train the best possible multiclass classifier using AutoML with the deterministic
model portfolio. Report the macro-F1 on the held-out test set and the top
predictive features.
