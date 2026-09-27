"""Offline check of the two predict_proba rewrites in executor._sanitize_code.

Both used to fire on the mere presence of a token and rewrote CORRECT code into broken code,
and the resulting error was charged to the model:

  * the log_loss rule stripped every `.predict_proba(...)[:, 1]` once `log_loss` appeared, so
    the binary idiom `y_prob = m.predict_proba(X)[:, 1]` became a 2-D matrix: roc_auc_score
    raised "y should be a 1d array", thresholding raised "can't handle a mix of binary and
    multiclass-multioutput targets" (26x on model-selection holdout_evaluation);
  * the AutoGluon rule rewrote any `predict_proba(...)[:, n]` / `y_prob[:, n]` to `.iloc`,
    which is an AttributeError on sklearn's ndarray (edf7f79).

    PYTHONPATH=src uv run python scripts/test_sanitizer_proba.py
"""
import contextlib
import io
import sys

sys.path.insert(0, "src")

from gads.core.executor import _sanitize_code

FAILS = []


def sanitize(code, kernel_state=None):
    with contextlib.redirect_stdout(io.StringIO()):
        return _sanitize_code(code, kernel_state=kernel_state)


def check(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"  — {detail}" if not cond and detail else ""))
    if not cond:
        FAILS.append(name)


SKLEARN_BINARY = """\
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)
roc = roc_auc_score(y_test, y_prob)
ll = log_loss(y_test, y_prob)
f1 = f1_score(y_test, y_pred, average="macro")
"""

print("sklearn (ndarray) code must pass through untouched")
out = sanitize(SKLEARN_BINARY)
check("binary idiom with log_loss unchanged", out == SKLEARN_BINARY, repr(out))
out = sanitize(SKLEARN_BINARY, {"model": {"type": "LogisticRegression"},
                                "y_prob": {"type": "ndarray", "shape": [40, 2]}})
check("unchanged with an ndarray y_prob already in the kernel", out == SKLEARN_BINARY, repr(out))
code = "p1 = clf.predict_proba(X_test)[:, 1]\n"
check("inline sklearn slice without log_loss unchanged", sanitize(code) == code)

print("log_loss gets the full matrix, inside the call only")
code = "ll = log_loss(y_test, model.predict_proba(X_test)[:, 1])\n"
check("slice inside log_loss(...) removed",
      sanitize(code) == "ll = log_loss(y_test, model.predict_proba(X_test))\n", sanitize(code))
code = "ll = log_loss(y_test, model.predict_proba(X_test.drop(columns=['id']))[:, 1])\n"
check("nested parentheses inside log_loss(...)",
      sanitize(code) == "ll = log_loss(y_test, model.predict_proba(X_test.drop(columns=['id'])))\n",
      sanitize(code))
code = "p = model.predict_proba(X_test)[:, 1]\nll = log_loss(y_test, model.predict_proba(X_test)[:, 1])\n"
out = sanitize(code)
check("slice outside log_loss kept, inside removed",
      out == "p = model.predict_proba(X_test)[:, 1]\nll = log_loss(y_test, model.predict_proba(X_test))\n", out)

print("AutoGluon objects get .iloc")
code = ("predictor = TabularPredictor(label='y').fit(train)\n"
        "y_prob = predictor.predict_proba(X_test)\n"
        "roc = roc_auc_score(y_test, y_prob[:, 1])\n")
out = sanitize(code)
check("predictor built in this code → y_prob.iloc", "y_prob.iloc[:, 1]" in out, out)
code = "p = predictor.predict_proba(X_test)[:, 1]\n"
out = sanitize(code, {"predictor": {"type": "TabularPredictor"}})
check("predictor from kernel state → inline .iloc",
      out == "p = predictor.predict_proba(X_test).iloc[:, 1]\n", out)
code = "roc = roc_auc_score(y_test, y_prob[:, 1])\n"
check("y_prob is a DataFrame in the kernel → .iloc",
      "y_prob.iloc[:, 1]" in sanitize(code, {"y_prob": {"type": "DataFrame"}}))
check("y_prob is an ndarray in the kernel → unchanged",
      sanitize(code, {"y_prob": {"type": "ndarray"}}) == code)
check("y_prob unknown → unchanged (ambiguous is left alone)", sanitize(code) == code)
check("polars frame (sandbox labels it DataFrame) → unchanged",
      sanitize(code, {"y_prob": {"type": "DataFrame", "engine": "polars"}}) == code)

print("mixed evidence stays conservative")
code = "y_prob = model.predict_proba(X_test)\nroc = roc_auc_score(y_test, y_prob[:, 1])\n"
out = sanitize(code, {"predictor": {"type": "TabularPredictor"}, "model": {"type": "RandomForestClassifier"}})
check("y_prob from sklearn while a predictor exists → unchanged", out == code, out)

print("the sanitized sklearn code actually runs")
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, roc_auc_score

X, y = make_classification(200, random_state=0)
X_test, y_test = X[150:], y[150:]
model = LogisticRegression().fit(X[:150], y[:150])
env = dict(model=model, X_test=X_test, y_test=y_test, np=np,
           roc_auc_score=roc_auc_score, log_loss=log_loss, f1_score=f1_score)
try:
    exec(sanitize(SKLEARN_BINARY), env)
    check("binary idiom executes (roc_auc, log_loss, macro-F1)", 0 < env["roc"] <= 1)
except Exception as e:
    check("binary idiom executes (roc_auc, log_loss, macro-F1)", False, f"{type(e).__name__}: {e}")

print()
if FAILS:
    sys.exit(f"{len(FAILS)} check(s) failed: {FAILS}")
print("all checks passed")
