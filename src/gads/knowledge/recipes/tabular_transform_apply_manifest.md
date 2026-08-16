---
id: tabular_transform.apply_manifest
version: 1.0.0
schema_version: 1
author: gads-core

# ——— ROUTING METADATA ———
applies_when:
  task_type: [data_transformation, feature_preparation, data_preparation]
  data_modality: [tabular]
  signals:
    - objective_contains: [apply the transformations, apply the manifest, transform the dataset,
                           prepare the data for modelling, prepare the data for modeling,
                           produce the transformed dataset, split into train]
  anti_signals:
    - objective_contains: [explore, exploratory, train a model, forecast]

# ——— SCHEMA CONTRACT ———
requires:
  variables:
    - name: df
      kind: pandas.DataFrame
  capabilities: [pandas, pyarrow]

# ——— DAG TEMPLATE ———
dag:
  - id: apply_manifest
    intent: >
      Apply a transformation manifest produced by an earlier EDA run.
      The manifest is `upstream/eda_transformations.meta.json` when this run was launched
      with `artifacts_from`, otherwise `eda_transformations.meta.json` in the workspace —
      check for the former first. Read it to find which source file it describes
      (the single key under `files`), load that file from the workspace into `df`, then call:
        result = gads_apply_transformations(df, <manifest path>, source_file=<that key>)
        n_rows_out = result["n_rows_out"]; n_cols_out = result["n_cols_out"]
        n_columns_dropped = len(result["columns_dropped"])
      Do NOT re-implement the imputation, scaling, encoding or splitting yourself and do NOT
      write the parquet files yourself — the native does all of it, in the one order that is
      safe (split first, fit on train only, apply those fitted parameters to every
      partition). Print the returned summary.
    worker_tier: T3
    produces: [df, result, n_rows_out, n_cols_out, n_columns_dropped]
    required_metrics: [n_rows_out, n_cols_out, n_columns_dropped]
    attached_skills: []
    fallback_native: gads_apply_transformations
    fallback_call: "import glob as _g, json as _j, pandas as _pd, os as _os; _mp = 'upstream/eda_transformations.meta.json' if _os.path.exists('upstream/eda_transformations.meta.json') else 'eda_transformations.meta.json'; _m = _j.load(open(_mp)); _src = list(_m['files'])[0]; _cand = [p for p in (_os.path.join('upstream', _src), _src) if _os.path.exists(p)]; df = _pd.read_parquet(_cand[0]) if _cand[0].endswith('.parquet') else _pd.read_csv(_cand[0]); result = gads_apply_transformations(df, _m, source_file=_src); n_rows_out = result['n_rows_out']; n_cols_out = result['n_cols_out']; n_columns_dropped = len(result['columns_dropped'])"
    postconditions:
      - "n_rows_out > 0"
      - "n_cols_out > 0"

  - id: verify_outputs
    intent: >
      Verify what was written and report it. Load each parquet the previous step produced
      (the paths are in `result["outputs"]`) and confirm for each partition: the row count,
      the column count, and that no nulls remain in columns the manifest gave an imputation
      strategy. If the manifest carried a `split`, also confirm the partitions do not
      overlap — for a grouped split no group id may appear in two partitions, and for a
      time-ordered split the training partition must end before the test partition begins.
      Print a short table of the results and store it in `verification`.
    depends_on: [apply_manifest]
    worker_tier: T2
    produces: [verification]
    attached_skills: []
    postconditions:
      - "verification is not None"

# ——— INVARIANTS ———
invariants:
  - "The transformation is applied by gads_apply_transformations, never re-implemented: the split-then-fit-on-train ordering is a leakage guard, and hand-written code reproduces it incorrectly more often than not."
  - "Files under `upstream/` are symlinks into a finished run's workspace — read them, never write through them. All outputs go to the downstream workspace root."
  - "The manifest names its own source file; do not guess which dataset to transform."
---

## Rationale

The companion to `tabular_eda.descriptive.standard` (approach_docs/021 §6). An EDA run decides
how each field should be imputed, scaled and encoded, and records that in
`eda_transformations.meta.json`; this recipe turns that decision into the transformed dataset,
in a separate run that can be launched from a spec with `artifacts_from: <eda project uuid>`.

Keeping it a separate run rather than a final node of the EDA recipe is deliberate. The
manifest is meant to be **reviewed** — an analyst can edit a recommendation before it is
applied — and a recommendation that is applied in the same breath as it is made cannot be. It
also means the transform can be re-run against an edited manifest without repeating the
profiling.

The recipe is thin on purpose. Almost all of its work is a single native call, because the
ordering inside that call (split first, fit on the training partition only, apply those fitted
parameters everywhere) is a correctness property, not a matter of style. Fitting before
splitting leaks held-out statistics into training and produces a model that scores well and
generalises badly. That is exactly the kind of invariant approach_docs/019 says to nativize —
and, empirically, calling the native is within a local 12B's reach even where authoring the
equivalent code is not (follow-up `b1f055ba`).

The second node exists because "the file was written" is a weak claim. It re-opens the outputs
and checks the properties that would actually be violated by a botched transform: residual
nulls, and partition overlap for grouped and time-ordered splits.
