"""
GADS Native Model-Audit Node

A reusable, general-purpose methodological-soundness gate for any sklearn-compatible
supervised estimator. Wraps `skore`'s EstimatorReport to run its automated diagnostic
checks (data leakage, over/under-fitting, class imbalance, worse-than-baseline, MDI
feature-importance bias, useless features, untuned hyperparameters, time-series
train/test overlap, ...) and its metric summary, then emits the flagged issues as a
structured `model_checks.json` artifact plus a human-readable digest.

Why this exists: GADS already gates on completeness and output contracts, but had no
signal for *methodological appropriateness* — the axis on which agentic DS most often
fails silently (an LLM can write code that runs cleanly and still leaks the test set or
never beats a baseline). skore turns those judgment calls into measurable checks; this
node makes that signal available deterministically, independent of what the Coder wrote.

Design: one general primitive parameterized by (estimator, data). Nothing dataset- or
error-case-specific. Annotation-free and self-contained (imports inside) so the source
can be injected verbatim into the sandbox kernel via the preamble.
"""


def gads_audit_model(estimator, X_train=None, y_train=None, X_test=None, y_test=None,
                     X=None, y=None, test_size=0.25, random_state=42,
                     write_path="model_checks.json", emit_insights=True):
    """Audit a supervised estimator for methodological soundness with skore.

    Provide EITHER an explicit split (X_train/y_train/X_test/y_test) — preferred, and
    required to honestly assess leakage/overfitting — OR a full (X, y) which is split
    internally (stratified for classification). The estimator may be fitted or unfitted;
    skore refits it on the training portion so the audit reflects the reported pipeline.

    Runs skore's EstimatorReport automated checks + metric summary, writes
    `model_checks.json` (every check with code/title/severity/explanation/doc URL, plus a
    flat metrics dict), prints a digest, and — if a `gads_emit_insight` emitter is present
    in the kernel — emits one insight per `issue`-severity finding.

    Returns a dict: {issues, tips, passed, not_applicable, n_issues, n_tips, metrics,
    ml_task, checks_path}. Fail-open: on any skore error it returns a dict with an
    `error` key and does not raise, so a diagnostic failure never fails the task.
    """
    import json
    import numpy as np

    result = {"issues": [], "tips": [], "passed": [], "not_applicable": [],
              "n_issues": 0, "n_tips": 0, "metrics": {}, "ml_task": None,
              "checks_path": write_path}
    try:
        from skore import EstimatorReport
    except Exception as e:  # skore not installed — degrade gracefully
        print(f"[gads_audit_model] skore unavailable ({e}); skipping methodological audit")
        result["error"] = f"skore unavailable: {e}"
        return result

    try:
        # Resolve the train/test split.
        if X_train is None or X_test is None:
            if X is None or y is None:
                raise ValueError("provide either (X_train,y_train,X_test,y_test) or (X,y)")
            from sklearn.model_selection import train_test_split
            ys = np.asarray(y)
            is_cls = ys.dtype.kind in "OUSb" or (ys.dtype.kind in "iu" and len(np.unique(ys)) <= 20)
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state,
                    stratify=y if is_cls else None)
            except Exception:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state)

        report = EstimatorReport(estimator, X_train=X_train, y_train=y_train,
                                 X_test=X_test, y_test=y_test)
        result["ml_task"] = str(getattr(report, "ml_task", None))

        # --- Automated methodological checks ---
        cf = report.checks.summarize().frame()
        records = cf.to_dict(orient="records")
        by_section = {}
        for rec in records:
            row = {"code": rec.get("code"), "title": rec.get("title"),
                   "severity": rec.get("section"),
                   "explanation": (None if (rec.get("explanation") is None
                                   or (isinstance(rec.get("explanation"), float)
                                       and np.isnan(rec.get("explanation"))))
                                   else str(rec.get("explanation"))),
                   "documentation_url": rec.get("documentation_url")}
            by_section.setdefault(rec.get("section"), []).append(row)
        result["issues"] = by_section.get("issue", [])
        result["tips"] = by_section.get("tip", [])
        result["passed"] = by_section.get("passed", [])
        result["not_applicable"] = by_section.get("not_applicable", [])
        result["n_issues"] = len(result["issues"])
        result["n_tips"] = len(result["tips"])

        # --- Metric summary (flatten to a scalar dict) ---
        try:
            mf = report.metrics.summarize().frame()
            col = mf.columns[0]
            flat = {}
            for idx, val in mf[col].items():
                if isinstance(idx, tuple):
                    key = "_".join(str(p) for p in idx if str(p) != "").strip("_")
                else:
                    key = str(idx)
                try:
                    flat[key] = float(val)
                except (TypeError, ValueError):
                    flat[key] = val
            result["metrics"] = flat
        except Exception as me:
            print(f"[gads_audit_model] metric summary unavailable: {me}")

        # --- Persist the machine-readable audit artifact ---
        with open(write_path, "w") as f:
            json.dump({"ml_task": result["ml_task"], "metrics": result["metrics"],
                       "issues": result["issues"], "tips": result["tips"],
                       "passed": [c["code"] for c in result["passed"]],
                       "not_applicable": [c["code"] for c in result["not_applicable"]]},
                      f, indent=2)

        # --- Human-readable digest to stdout ---
        print(f"[gads_audit_model] {result['ml_task']} — "
              f"{result['n_issues']} issue(s), {result['n_tips']} tip(s), "
              f"{len(result['passed'])} passed, {len(result['not_applicable'])} n/a")
        for c in result["issues"]:
            print(f"  [ISSUE] {c['code']} {c['title']}: {c['explanation']}")
        for c in result["tips"]:
            print(f"  [tip]   {c['code']} {c['title']}: {c['explanation']}")

        # --- Emit issues as insights when an emitter is available in the kernel ---
        if emit_insights:
            emit = globals().get("gads_emit_insight")
            if callable(emit):
                for c in result["issues"]:
                    try:
                        emit(f"Methodological check {c['code']} ({c['title']}): "
                             f"{c['explanation']} See {c['documentation_url']}")
                    except Exception:
                        pass
        return result
    except Exception as e:
        print(f"[gads_audit_model] audit failed ({type(e).__name__}: {e}); continuing without it")
        result["error"] = f"{type(e).__name__}: {e}"
        return result
