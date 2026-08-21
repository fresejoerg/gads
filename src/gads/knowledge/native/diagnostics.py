"""Classification diagnostic curves — ROC and precision-recall.

Why this is a native rather than left to codegen: the curves themselves are an invariant
operation with one right answer (approach_docs/019's rule for what earns nativisation).
Everything that makes them *wrong* is mechanical and recurring — slicing the wrong
`predict_proba` column, plotting a thresholded 0/1 prediction instead of a score, treating a
multiclass problem as binary, comparing against no baseline at all, or emitting Plotly JSON
with numpy `bdata` the dashboard cannot render. None of that is a judgment call, so none of
it is worth spending model capability on.

What is deliberately NOT nativised: whether curves are appropriate at all, and what the
report concludes from them. The recipes call this where probabilities exist; the model still
writes the interpretation.

Both curves carry their baseline, because a curve without one invites the reader to over-read
it: the ROC diagonal (a coin flip) and the PR prevalence line (the positive rate, which is
where a no-skill classifier sits — and unlike ROC's diagonal it moves with class balance,
which is exactly why PR is the honest curve on imbalanced data).

IMPORTANT — this function is declared as a `fallback_native`, and the executor's fallback
path injects exactly ONE function's source. It must stay fully self-contained: imports
inside, no annotations, no calls to siblings.
"""


def gads_plot_classification_curves(y_true, y_prob, class_labels=None,
                                    roc_path="figure_roc_curve.json",
                                    pr_path="figure_precision_recall.json",
                                    max_classes=12, emit_insights=True):
    """Plot ROC and precision-recall curves from predicted probabilities.

    y_true      : true labels, any dtype (strings fine — they are not cast).
    y_prob      : probabilities. 1D = positive-class score for a binary problem; 2D /
                  DataFrame = one column per class, in `class_labels` order (a DataFrame's
                  own columns are used when class_labels is None).
    class_labels: class order for a 2D y_prob. Inferred from sorted(unique(y_true)) if None.

    BINARY    : one ROC curve and one PR curve, each against its baseline.
    MULTICLASS: one-vs-rest per class plus a macro-average, capped at `max_classes` curves
                so a high-cardinality target cannot render an unreadable plot.

    Writes two Plotly JSON figures the dashboard renders natively, and returns
    {roc_auc, average_precision, mode, n_classes, roc_path, pr_path, per_class}.
    Fail-open: on any error it returns the dict with an "error" key rather than raising, so
    a diagnostic plot can never fail an evaluation node that already produced its metrics.
    """
    import json
    import numpy as np
    import pandas as pd

    result = {"roc_auc": None, "average_precision": None, "mode": None,
              "n_classes": None, "roc_path": roc_path, "pr_path": pr_path,
              "per_class": {}}
    try:
        from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                                     average_precision_score, roc_auc_score)

        y_true_arr = np.asarray(getattr(y_true, "values", y_true)).ravel()

        # ---- normalise the probability matrix -------------------------------------
        cols = None
        if isinstance(y_prob, pd.DataFrame):
            cols = [str(c) for c in y_prob.columns]
            prob = y_prob.values
        else:
            prob = np.asarray(y_prob)
            if prob.dtype == object:
                prob = np.asarray(prob.tolist())
        if prob.ndim == 1:
            prob = prob.reshape(-1, 1)

        observed = sorted(pd.Series(y_true_arr).unique().tolist(), key=lambda v: str(v))
        if class_labels is not None:
            labels = [str(c) for c in class_labels]
        elif cols is not None and prob.shape[1] > 1:
            labels = cols
        else:
            labels = [str(c) for c in observed]

        n_true = len(observed)
        is_binary = (prob.shape[1] <= 2) and n_true <= 2
        result["n_classes"] = n_true
        result["mode"] = "binary" if is_binary else "multiclass"

        roc_traces, pr_traces = [], []

        def _curves_for(y_bin, score, name):
            """One ROC + one PR trace for a binary indicator. Lists, not numpy arrays:
            Plotly encodes arrays as base64 `bdata`, which the dashboard cannot render."""
            fpr, tpr, _ = roc_curve(y_bin, score)
            a = float(auc(fpr, tpr))
            prec, rec, _ = precision_recall_curve(y_bin, score)
            ap = float(average_precision_score(y_bin, score))
            roc_traces.append({"type": "scatter", "mode": "lines",
                               "x": [float(v) for v in fpr], "y": [float(v) for v in tpr],
                               "name": f"{name} (AUC {a:.3f})"})
            pr_traces.append({"type": "scatter", "mode": "lines",
                              "x": [float(v) for v in rec], "y": [float(v) for v in prec],
                              "name": f"{name} (AP {ap:.3f})"})
            return a, ap

        if is_binary:
            positive = labels[-1] if len(labels) > 1 else (str(observed[-1]) if observed else "1")
            y_bin = (pd.Series(y_true_arr).astype(str) == str(positive)).astype(int).values
            score = prob[:, 1] if prob.shape[1] == 2 else prob[:, 0]
            a, ap = _curves_for(y_bin, score, f"class {positive}")
            result["roc_auc"], result["average_precision"] = a, ap
            prevalence = float(np.mean(y_bin))
        else:
            shown = labels[:max_classes]
            aucs, aps = [], []
            for i, lab in enumerate(shown):
                if i >= prob.shape[1]:
                    break
                y_bin = (pd.Series(y_true_arr).astype(str) == str(lab)).astype(int).values
                if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                    continue          # a class absent from the split has no curve
                a, ap = _curves_for(y_bin, prob[:, i], str(lab))
                result["per_class"][str(lab)] = {"roc_auc": a, "average_precision": ap}
                aucs.append(a)
                aps.append(ap)
            result["roc_auc"] = float(np.mean(aucs)) if aucs else None
            result["average_precision"] = float(np.mean(aps)) if aps else None
            try:                       # prefer sklearn's own macro OvR where it applies
                result["roc_auc"] = float(roc_auc_score(y_true_arr, prob,
                                                        multi_class="ovr", average="macro"))
            except Exception:
                pass
            prevalence = 1.0 / max(len(labels), 1)
            if len(labels) > len(shown):
                print(f"[gads_plot_classification_curves] {len(labels)} classes; showing the "
                      f"first {len(shown)} one-vs-rest curves.")

        # ---- baselines -------------------------------------------------------------
        roc_traces.append({"type": "scatter", "mode": "lines", "x": [0.0, 1.0], "y": [0.0, 1.0],
                           "name": "chance", "line": {"dash": "dash", "color": "#94a3b8"}})
        pr_traces.append({"type": "scatter", "mode": "lines", "x": [0.0, 1.0],
                          "y": [prevalence, prevalence],
                          "name": f"no skill ({prevalence:.3f})",
                          "line": {"dash": "dash", "color": "#94a3b8"}})

        suffix = "" if is_binary else " (one-vs-rest)"
        roc_fig = {"data": roc_traces,
                   "layout": {"title": {"text": "ROC curve" + suffix},
                              "xaxis": {"title": {"text": "False positive rate"},
                                        "range": [0, 1]},
                              "yaxis": {"title": {"text": "True positive rate"},
                                        "range": [0, 1.02]},
                              "template": "plotly_white"}}
        pr_fig = {"data": pr_traces,
                  "layout": {"title": {"text": "Precision-recall curve" + suffix},
                             "xaxis": {"title": {"text": "Recall"}, "range": [0, 1]},
                             "yaxis": {"title": {"text": "Precision"}, "range": [0, 1.02]},
                             "template": "plotly_white"}}

        with open(roc_path, "w") as f:
            json.dump(roc_fig, f)
        with open(pr_path, "w") as f:
            json.dump(pr_fig, f)

        auc_s = "n/a" if result["roc_auc"] is None else f"{result['roc_auc']:.4f}"
        ap_s = "n/a" if result["average_precision"] is None else f"{result['average_precision']:.4f}"
        print(f"[gads_plot_classification_curves] {result['mode']}: ROC-AUC={auc_s} "
              f"AP={ap_s} (no-skill AP={prevalence:.4f})")
        print(f"[gads_plot_classification_curves] wrote {roc_path} and {pr_path}")

        if emit_insights:
            try:
                gads_emit_insight(
                    roc_path,
                    f"ROC-AUC {auc_s} against a chance baseline of 0.500.",
                    f"mode={result['mode']}, n_classes={result['n_classes']}")
                gads_emit_insight(
                    pr_path,
                    f"Average precision {ap_s} against a no-skill baseline of "
                    f"{prevalence:.4f} (the positive rate).",
                    "Precision-recall is the more honest curve under class imbalance: its "
                    "baseline moves with prevalence while ROC's does not.")
            except Exception:
                pass

        return result
    except Exception as e:
        print(f"[gads_plot_classification_curves] skipped: {e}")
        result["error"] = str(e)
        return result
