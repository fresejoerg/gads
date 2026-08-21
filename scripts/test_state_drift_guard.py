"""Offline check of the upstream-output immutability guard (executor.check_state_drift).

Regression test for the silent correctness failure in local run 22a8524e (2026-08-20): a
failed attempt at `holdout_evaluation` rebound `X_test` from the 4,000-row test split to the
full 46,842-row dataset — target column included — and the native fallback then computed the
run's headline metrics on it. macro_f1 came out 0.8211, indistinguishable from the legitimate
cloud runs, and the CompletenessVerifier passed the run because completeness is not
correctness.

    PYTHONPATH=src uv run python scripts/test_state_drift_guard.py
"""
import sys
sys.path.insert(0, "src")

from gads.core.executor import ExecutionManager


def fresh():
    m = ExecutionManager.__new__(ExecutionManager)
    m.authoritative_state = {}
    m.protected_state = {}
    return m


def split_state(m):
    m.authoritative_state.update({
        "X_train": {"type": "DataFrame", "shape": [16000, 14]},
        "y_train": {"type": "Series", "shape": [16000]},
        "X_test": {"type": "DataFrame", "shape": [4000, 14]},
        "y_test": {"type": "Series", "shape": [4000]},
    })
    m.record_produced_state(["X_train", "y_train", "X_test", "y_test"])
    return m


# 1. the real failure — full dataset swapped in for the test split
m = split_state(fresh())
m.authoritative_state["X_test"] = {"type": "DataFrame", "shape": [46842, 15]}
m.authoritative_state["y_test"] = {"type": "Series", "shape": [46842]}
drift = m.check_state_drift(["y_pred", "y_prob", "macro_f1", "roc_auc", "log_loss"])
assert len(drift) == 2, drift
assert any("X_test" in d for d in drift) and any("y_test" in d for d in drift), drift
print("caught the 22a8524e contamination:", len(drift), "violations")

# 2. an untouched downstream node is silent
m = split_state(fresh())
assert m.check_state_drift(["y_pred", "y_prob"]) == []
print("clean node: no false positive")

# 3. a node may rebind what it DECLARES (holdout_evaluation genuinely refits tuned_model)
m = split_state(fresh())
m.authoritative_state["tuned_model"] = {"type": "Pipeline"}
m.record_produced_state(["tuned_model"])
m.authoritative_state["tuned_model"] = {"type": "Pipeline", "shape": None}
assert m.check_state_drift(["tuned_model"]) == [], m.check_state_drift(["tuned_model"])
print("declared rebind allowed")

# 4. rebinding an upstream output IS caught even when shapes stay plausible
m = split_state(fresh())
m.authoritative_state["X_test"] = {"type": "DataFrame", "shape": [4000, 15]}   # target added
d = m.check_state_drift(["y_pred"])
assert len(d) == 1 and "X_test" in d[0], d
print("column-count-only drift caught (target leaking into X):", d[0])

# 5. a variable that vanished from the kernel is not reported as drift (nothing to compare)
m = split_state(fresh())
del m.authoritative_state["X_test"]
assert m.check_state_drift(["y_pred"]) == []
print("missing variable: no spurious drift")

print("\nALL CHECKS PASSED")
