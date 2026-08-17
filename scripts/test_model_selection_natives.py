"""Live-sandbox unit check for the model-selection natives (approach_docs/022).

Runs the ACTUAL injected preamble source against a real dataset in the real sandbox, so
what is tested is what the kernel will get. Not a pytest — matches the repo's ad-hoc
script convention.

    PYTHONPATH=src uv run python scripts/test_model_selection_natives.py
"""
import asyncio
import json
import sys

import httpx

from gads.knowledge.native import MODEL_SELECTION_PREAMBLE, NATIVE_SOURCE

SANDBOX = "http://localhost:8000"
SESSION = "ms_native_test"

# The fallback-only natives are not in the preamble (by design), so inject their source
# the same way the executor's fallback path does.
FALLBACK_SRC = "\n\n".join(
    NATIVE_SOURCE[n] for n in ("gads_load_prepared_split", "gads_dataset_facts",
                               "gads_default_shortlist", "gads_evaluate_holdout",
                               "gads_model_card"))

PREAMBLE = (MODEL_SELECTION_PREAMBLE + "\n\n" + FALLBACK_SRC + "\n\n"
            "if '_gads_insights' not in globals(): _gads_insights = []\n"
            "def gads_emit_insight(artifact, insight, evidence=''):\n"
            "    _gads_insights.append({'artifact': artifact, 'insight': insight})\n")


async def run(code, label, timeout=600.0):
    async with httpx.AsyncClient(timeout=timeout + 60) as c:
        r = await c.post(f"{SANDBOX}/execute",
                         json={"code": code, "session_id": SESSION, "timeout": timeout})
        r.raise_for_status()
        d = r.json()
    err = d.get("error")
    print(f"\n{'=' * 78}\n### {label}   ({d.get('execution_time_ms', 0) / 1000:.1f}s)\n{'=' * 78}")
    out = d.get("stdout", "")
    print(out[-4000:] if len(out) > 4000 else out)
    if err:
        print(f"!!! ERROR {err.get('ename')}: {err.get('evalue')}")
        tb = err.get("traceback") or []
        print("\n".join(tb[-6:]))
    return d, err


TESTS = {}

TESTS["1. setup + load raw adult.csv"] = PREAMBLE + """
import pandas as pd, numpy as np, time, os
os.chdir('/tmp')
df = pd.read_csv('/home/joergf/datasets/amlb/adult.csv')
print('shape', df.shape)
print('dtypes:'); print(df.dtypes.value_counts())
print('object cols:', [c for c in df.columns if df[c].dtype == object][:12])
df.to_csv('/tmp/adult_probe.csv', index=False)
"""

TESTS["2. load_prepared_split on a RAW csv (categoricals present)"] = """
r = gads_load_prepared_split(source='/tmp/adult_probe.csv', target_column='class')
X_train, y_train = r['X_train'], r['y_train']
X_test, y_test = r['X_test'], r['y_test']
print('non-numeric feature columns remaining:',
      [c for c in X_train.columns if X_train[c].dtype == object])
"""

TESTS["3. dataset_facts + default_shortlist"] = """
facts = gads_dataset_facts(X_train, y_train)
sl = gads_default_shortlist(facts)
candidates = sl['candidates']
"""

TESTS["4. bakeoff incl. xgboost + lightgbm (the real package test)"] = """
import time; t0=time.time()
b = gads_candidate_bakeoff(X_train, y_train,
        ['logistic_regression','random_forest','xgboost','lightgbm','hist_gradient_boosting'],
        cv=3, seed=42)
print('elapsed %.1fs' % (time.time()-t0))
print('failures:', b['failures'])
bakeoff_table, best_candidate = b['bakeoff_table'], b['best_candidate']
best_cv_score = b['best_cv_score']
"""

TESTS["5. tune_model — BUDGET ENFORCEMENT (timeout_s=25)"] = """
import time; t0=time.time()
t = gads_tune_model(X_train, y_train, best_candidate, n_trials=500, timeout_s=25, cv=3)
el = time.time()-t0
print('WALL CLOCK: %.1fs (budget 25s)' % el)
print('n_trials_completed:', t['n_trials_completed'], ' timed_out:', t['timed_out'])
print('BUDGET HONOURED:', el < 70)
print('TRUNCATION REPORTED HONESTLY:', t['timed_out'] is True and t['n_trials_completed'] < 500)
tuning_result = t; tuned_model = t['tuned_model']
"""

TESTS["6. default search space (fallback path, no space supplied)"] = """
t2 = gads_tune_model(X_train, y_train, 'random_forest', n_trials=4, timeout_s=60, cv=3)
print('space used:', list(t2['search_space_used'].keys()))
print('NON-EMPTY DEFAULT SPACE:', len(t2['search_space_used']) > 0)
print('trials:', t2['n_trials_completed'])
"""

TESTS["7. audit_model_choice gate"] = """
a = gads_audit_model_choice(best_candidate, facts, bakeoff_table, tuning_result)
print('n_selection_issues:', a['n_selection_issues'], 'blocking:', a['n_blocking'])
print('checks file written:', __import__('os').path.exists('model_choice_checks.json'))
selection_audit = a
"""

TESTS["8. holdout evaluation (metric binding)"] = """
e = gads_evaluate_holdout(tuned_model, X_train, y_train, X_test, y_test)
macro_f1, roc_auc, log_loss = e['macro_f1'], e['roc_auc'], e['log_loss']
print('BOUND:', {'macro_f1': round(macro_f1,4), 'roc_auc': round(roc_auc,4),
                 'log_loss': round(log_loss,4)})
print('beats baseline:', e['beats_baseline'])
evaluation = e
"""

TESTS["9. feature importance (held-out permutation)"] = """
imp = gads_feature_importance(tuned_model, X_test, y_test, n_repeats=3)
importance = imp
nz = int((imp['importance_table']['importance_mean'] != 0).sum())
print('NON-ZERO IMPORTANCES: %d / %d' % (nz, len(imp['importance_table'])))
"""

TESTS["10. model card"] = """
c = gads_model_card(chosen=best_candidate, dataset_facts=facts, bakeoff_table=bakeoff_table,
                    tuning_result=tuning_result, evaluation=evaluation, importance=importance,
                    selection_rationale=sl['selection_rationale'], selection_audit=selection_audit)
print('CARD SECTIONS:', c['sections'])
print('CARD CHARS:', len(c['model_card_text']))
"""


async def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    failures = []
    for label, code in TESTS.items():
        if only and not label.startswith(only):
            continue
        _, err = await run(code, label)
        if err:
            failures.append(label)
            if label.startswith("1.") or label.startswith("2."):
                print("\n*** aborting: setup step failed ***")
                break
    print(f"\n\n{'#' * 78}")
    print("FAILED STEPS:", failures if failures else "none")
    print(f"{'#' * 78}")


if __name__ == "__main__":
    asyncio.run(main())
