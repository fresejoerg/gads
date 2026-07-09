# fraud_autogluon_v1 — provenance & tolerance rationale

**Established:** 2026-07-09, harness at commit `0bb7e85`.
**Dataset:** `creditcard.csv` (Kaggle ULB fraud, 284,807 rows, `Class` target ~0.17% positive),
capped to 50,000 rows by the spec's `sample_rows`.
**Latent plan:** codified in recipe `tabular_automl.autogluon.standard` (3 nodes:
Profile → Train → Extract/visualise). Spec pins the recipe, so all reference runs used the
deterministic front-end (compiled plan, zero front-end LLM calls).

## Reference runs

| Mode | Project | Coder engine | test_score | Outcome |
|---|---|---|---|---|
| cloud | `faa3870e` | claude-sonnet-4.6 (via 503-escalations) | 0.9692712906057946 | 3/3 tasks ✓ |
| local | `e9ecf496` | gemma-4-12b-qat | 0.9570593538427091 | 3/3 first-shot, 0 escalations |
| hybrid | `3fcb048a` | gemma-4-12b-qat | 0.9570593538427091 | 3/3 ✓, cloud synth/critique |
| cloud_pinned | `151a323d` | claude-haiku-4.5 | 0.9570593538427091 | attempt 1 all ✓; verifier replan re-ran (Extract failed attempts 2–3) |

## Tolerance rationale

- `naive_baseline` **exact**: value-counts fraction of the seeded sample; any deviation means
  the sampling constraint (`sample(50000, random_state=42)`) was violated → methodology failure,
  not noise.
- `test_score` **tol 0.02**: identical generated code produced 0.9571 (3 runs, 2 engines) vs
  0.9693 (1 run) purely from `fit(time_limit=120)` wall-clock dependence — see journal entry
  2026-07-09. The canonical value is the thrice-reproduced 0.9571. **TODO:** version to
  `fraud_autogluon_v2` with a fixed AutoGluon model portfolio in the recipe, then set
  `exact: true`.

## Known scoring caveats

- `tasks.produces` mirrors the recipe DAG's `produces` lists; the scorer treats them as
  documentation unless given DB access (workspace-only scoring checks metrics/artifacts/
  methodology).
- The verifier-replan artifact-overwrite hazard (journal 2026-07-08) means a scored workspace
  reflects the *last* attempt's Train outputs; identical scores across attempts masked this in
  `151a323d`.
