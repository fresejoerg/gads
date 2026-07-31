"""
GADS Native Survival-Analysis Nodes

Deterministic front-end primitives for right-censored time-to-event analysis. These
replace the LLM-generated code for the operations where survival workflows most reliably
break on a small model:

  1. `gads_make_surv_target` — build scikit-survival's structured target array. The
     `(event: bool, time: float)` structured-array format is THE classic failure point:
     models pass a 1-D label, forget the boolean cast, swap the field order, or silently
     drop censored rows. One correct, validated primitive removes the whole error class.

  2. `gads_evaluate_survival` — censoring-aware model evaluation in one call: Harrell's
     C-index, IPCW C-index, time-dependent AUC, and the Integrated Brier Score, evaluated
     at follow-up times chosen safely inside the overlap of train/test support (the other
     recurring failure: sksurv raises when evaluation times fall outside that range).

  3. `gads_cox_ph_report` — a lifelines CoxPH fit plus the PROPORTIONAL-HAZARDS assumption
     test, turned into a structured pass/fail with hazard ratios. This is to survival what
     the skore audit is to supervised learning: the methodological-soundness gate. A Cox
     model whose PH assumption is violated reports misleading constant hazard ratios, and
     that failure is silent unless it is explicitly tested.

Design mirrors the other native nodes: general primitives parameterized by (data, cols),
nothing dataset-specific, annotation-free and self-contained (imports inside) so the source
injects verbatim into the sandbox kernel, and FAIL-OPEN so a diagnostic never fails a task.
"""


def gads_make_surv_target(df, time_col, event_col):
    """Build a scikit-survival structured target array from a DataFrame.

    Returns the structured array `y` with fields ('event': bool, 'time': float) in that
    order — the exact format sksurv estimators and metrics require. The event column is
    coerced to boolean from the common encodings (0/1, True/False, 'dead'/'alive',
    'yes'/'no', '1'/'0'); censored rows are KEPT (dropping them is the cardinal error of
    survival analysis — censoring carries information). Prints a censoring summary.

    Raises ValueError only on genuinely un-coercible input, so callers see a clear message
    instead of a downstream dtype error.
    """
    import numpy as np
    import pandas as pd
    from sksurv.util import Surv

    if time_col not in df.columns:
        raise ValueError(f"time_col '{time_col}' not in dataframe columns {list(df.columns)}")
    if event_col not in df.columns:
        raise ValueError(f"event_col '{event_col}' not in dataframe columns {list(df.columns)}")

    time = pd.to_numeric(df[time_col], errors="coerce").astype(float)

    raw = df[event_col]
    if raw.dtype == bool:
        event = raw.to_numpy()
    elif np.issubdtype(raw.dtype, np.number):
        uniq = set(pd.unique(raw.dropna()).tolist())
        if not uniq.issubset({0, 1, 0.0, 1.0}):
            raise ValueError(
                f"numeric event_col '{event_col}' has values {sorted(uniq)}; expected 0/1 "
                "(0=censored, 1=event)")
        event = raw.astype(float).to_numpy() == 1.0
    else:
        s = raw.astype(str).str.strip().str.lower()
        true_tokens = {"1", "true", "t", "yes", "y", "dead", "death", "died", "event",
                       "relapse", "recurred", "recurrence", "failed", "failure"}
        false_tokens = {"0", "false", "f", "no", "n", "alive", "censored", "censor",
                        "no event", "no_event", "survived"}
        mapped = s.map(lambda v: True if v in true_tokens else (False if v in false_tokens else None))
        if mapped.isna().any():
            bad = sorted(set(s[mapped.isna()].tolist()))[:6]
            raise ValueError(
                f"could not map event_col '{event_col}' values {bad} to boolean; "
                "recode as 1=event, 0=censored before calling")
        event = mapped.to_numpy(dtype=bool)

    if np.isnan(time).any():
        n_bad = int(np.isnan(time).sum())
        raise ValueError(f"time_col '{time_col}' has {n_bad} non-numeric/NaN value(s); clean before use")
    if (time < 0).any():
        raise ValueError(f"time_col '{time_col}' has negative durations; survival time must be >= 0")

    y = Surv.from_arrays(event=event, time=time.to_numpy())

    n = len(y)
    n_event = int(event.sum())
    cens_rate = 1.0 - (n_event / n if n else 0.0)
    med = float(np.median(time.to_numpy()))
    print(f"[gads_make_surv_target] n={n}  events={n_event} ({n_event / n:.1%})  "
          f"censored={n - n_event} ({cens_rate:.1%})  median_time={med:.2f}")
    return y


def gads_kaplan_meier(df, time_col, event_col, group_col=None,
                      fig_path="km_curve.png", write_path="km_summary.json", emit_insights=True):
    """Kaplan-Meier survival description + log-rank test, done deterministically.

    Fits an overall KM estimator (reporting the median survival time), and — if `group_col`
    is given, or a low-cardinality categorical is auto-detected — fits per-group KM curves,
    plots them on one axis, and runs the multivariate log-rank test across groups. Saves the
    figure, writes `km_summary.json`, prints a digest, and emits the median-survival and
    log-rank insights. Returns a dict {overall_median, group_col, group_medians, logrank_p,
    n, n_events, fig_path}. Fail-open.

    Exists because assembling KaplanMeierFitter + grouping + the log-rank test + a plot is a
    reliable codegen failure on small models (wrong `median_survival_time_` attribute,
    mismatched brackets); one correct primitive removes that whole class of error.
    """
    import json
    import numpy as np

    result = {"overall_median": None, "group_col": None, "group_medians": {},
              "logrank_p": None, "n": int(len(df)), "n_events": None,
              "fig_path": fig_path, "summary_path": write_path}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import multivariate_logrank_test
    except Exception as e:
        print(f"[gads_kaplan_meier] lifelines/matplotlib unavailable ({e}); skipping")
        result["error"] = f"unavailable: {e}"
        return result

    try:
        T = df[time_col]
        E = df[event_col].astype(bool)
        result["n_events"] = int(E.sum())

        # Auto-pick a grouping column when none is given: a non-time/event column with 2–5
        # distinct values (categorical or low-cardinality numeric).
        if group_col is None:
            for c in df.columns:
                if c in (time_col, event_col):
                    continue
                nun = df[c].nunique(dropna=True)
                if 2 <= nun <= 5:
                    group_col = c
                    break

        fig, ax = plt.subplots(figsize=(8, 5))
        kmf = KaplanMeierFitter()
        kmf.fit(T, event_observed=E, label="overall")
        result["overall_median"] = (None if kmf.median_survival_time_ is None
                                    or (isinstance(kmf.median_survival_time_, float)
                                        and np.isinf(kmf.median_survival_time_))
                                    else float(kmf.median_survival_time_))

        if group_col is not None and group_col in df.columns:
            result["group_col"] = str(group_col)
            for gval, sub in df.groupby(group_col):
                if len(sub) < 2:
                    continue
                k = KaplanMeierFitter()
                k.fit(sub[time_col], event_observed=sub[event_col].astype(bool), label=f"{group_col}={gval}")
                k.plot_survival_function(ax=ax)
                med = k.median_survival_time_
                result["group_medians"][str(gval)] = (None if med is None
                    or (isinstance(med, float) and np.isinf(med)) else float(med))
            try:
                lr = multivariate_logrank_test(df[time_col], df[group_col], df[event_col].astype(bool))
                result["logrank_p"] = float(lr.p_value)
            except Exception as le:
                print(f"[gads_kaplan_meier] log-rank test unavailable: {le}")
        else:
            kmf.plot_survival_function(ax=ax)

        ax.set_xlabel(f"time ({time_col})")
        ax.set_ylabel("survival probability")
        ax.set_title("Kaplan-Meier survival" + (f" by {group_col}" if result["group_col"] else ""))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=110, bbox_inches="tight")
        plt.close(fig)

        with open(write_path, "w") as f:
            json.dump({k: v for k, v in result.items() if k not in ("fig_path", "summary_path")}, f, indent=2)

        med_str = "not reached" if result["overall_median"] is None else f"{result['overall_median']:.1f}"
        print(f"[gads_kaplan_meier] n={result['n']} events={result['n_events']} "
              f"median_survival={med_str}"
              + (f" | group='{result['group_col']}' log-rank p={result['logrank_p']}"
                 if result["group_col"] else ""))

        if emit_insights:
            emit = globals().get("gads_emit_insight")
            if callable(emit):
                try:
                    emit(f"Overall median survival time: {med_str} "
                         f"({result['n_events']} events across {result['n']} subjects).")
                except Exception:
                    pass
                if result["logrank_p"] is not None:
                    verdict = ("differ significantly" if result["logrank_p"] < 0.05
                               else "do not differ significantly")
                    try:
                        emit(f"Survival curves across {result['group_col']} {verdict} "
                             f"(log-rank p={result['logrank_p']:.4g}).")
                    except Exception:
                        pass
        return result
    except Exception as e:
        print(f"[gads_kaplan_meier] KM failed ({type(e).__name__}: {e}); continuing")
        result["error"] = f"{type(e).__name__}: {e}"
        return result


def gads_evaluate_survival(model, X_train, y_train, X_test, y_test, times=None,
                           write_path="survival_metrics.json", emit_insights=True):
    """Censoring-aware evaluation of a fitted scikit-survival model.

    Computes, each guarded independently so one failure never sinks the rest:
      - Harrell's C-index  (concordance_index_censored on the model risk score)
      - IPCW C-index       (concordance_index_ipcw — unbiased under heavy censoring)
      - time-dependent AUC (cumulative_dynamic_auc, mean over the evaluation times)
      - Integrated Brier Score (integrated_brier_score of the predicted survival curves;
                                lower is better, 0.25 = uninformative)

    `times` defaults to the 10th..90th percentiles of the TEST event times, clamped to sit
    strictly inside the overlap of train and test follow-up — outside that window sksurv's
    IPCW estimator raises. Writes `survival_metrics.json`, prints a digest, and emits one
    insight per headline metric. Returns a dict of the scalar metrics. Fail-open.
    """
    import json
    import numpy as np

    result = {"harrell_cindex": None, "ipcw_cindex": None, "mean_auc": None,
              "integrated_brier_score": None, "eval_times": None, "metrics_path": write_path}
    try:
        from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw,
                                     cumulative_dynamic_auc, integrated_brier_score)
    except Exception as e:
        print(f"[gads_evaluate_survival] scikit-survival unavailable ({e}); skipping")
        result["error"] = f"scikit-survival unavailable: {e}"
        return result

    try:
        ev_tr, t_tr = y_train.dtype.names[0], y_train.dtype.names[1]
        ev_te, t_te = y_test.dtype.names[0], y_test.dtype.names[1]
        risk = np.asarray(model.predict(X_test))

        # Harrell's C-index — always available from the risk score.
        try:
            result["harrell_cindex"] = float(
                concordance_index_censored(y_test[ev_te], y_test[t_te], risk)[0])
        except Exception as e:
            print(f"[gads_evaluate_survival] Harrell C-index unavailable: {e}")

        # Choose evaluation times inside the train/test follow-up overlap.
        if times is None:
            lo = max(float(y_train[t_tr].min()), float(y_test[t_te].min()))
            hi = min(float(y_train[t_tr].max()), float(y_test[t_te].max()))
            ev_times_test = y_test[t_te][y_test[ev_te]]
            pool = ev_times_test[(ev_times_test > lo) & (ev_times_test < hi)]
            if pool.size >= 5:
                times = np.percentile(pool, np.linspace(10, 90, 9))
            elif hi > lo:
                times = np.linspace(lo, hi, 9)[1:-1]
            else:
                times = None
        if times is not None:
            times = np.unique(np.asarray(times, dtype=float))
            times = times[(times > float(y_test[t_te].min())) & (times < float(y_test[t_te].max()))]
            if times.size == 0:
                times = None
        result["eval_times"] = None if times is None else [float(t) for t in times]

        # IPCW C-index — unbiased under censoring.
        try:
            result["ipcw_cindex"] = float(concordance_index_ipcw(y_train, y_test, risk)[0])
        except Exception as e:
            print(f"[gads_evaluate_survival] IPCW C-index unavailable: {e}")

        if times is not None and times.size > 0:
            # Time-dependent AUC.
            try:
                _auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk, times)
                result["mean_auc"] = float(mean_auc)
            except Exception as e:
                print(f"[gads_evaluate_survival] time-dependent AUC unavailable: {e}")
            # Integrated Brier Score — needs predicted survival probabilities at `times`.
            try:
                surv_fns = model.predict_survival_function(X_test)
                surv_prob = np.vstack([[float(fn(t)) for t in times] for fn in surv_fns])
                result["integrated_brier_score"] = float(
                    integrated_brier_score(y_train, y_test, surv_prob, times))
            except Exception as e:
                print(f"[gads_evaluate_survival] Integrated Brier Score unavailable: {e}")

        with open(write_path, "w") as f:
            json.dump({k: v for k, v in result.items() if k != "metrics_path"}, f, indent=2)

        print(f"[gads_evaluate_survival] Harrell C={result['harrell_cindex']}  "
              f"IPCW C={result['ipcw_cindex']}  mean AUC={result['mean_auc']}  "
              f"IBS={result['integrated_brier_score']} (lower better)")

        if emit_insights:
            emit = globals().get("gads_emit_insight")
            if callable(emit):
                c = result["ipcw_cindex"] or result["harrell_cindex"]
                if c is not None:
                    quality = ("no better than random" if c < 0.55 else
                               "modest" if c < 0.65 else "good" if c < 0.75 else "strong")
                    try:
                        emit(f"Survival model discrimination: C-index={c:.3f} ({quality}; "
                             "0.5=random, >0.7=good). IPCW-corrected for censoring where available.")
                    except Exception:
                        pass
                if result["integrated_brier_score"] is not None:
                    try:
                        emit(f"Integrated Brier Score={result['integrated_brier_score']:.3f} "
                             "(calibration+discrimination over time; lower is better, 0.25=uninformative).")
                    except Exception:
                        pass
        return result
    except Exception as e:
        print(f"[gads_evaluate_survival] evaluation failed ({type(e).__name__}: {e}); continuing")
        result["error"] = f"{type(e).__name__}: {e}"
        return result


def gads_cox_ph_report(df, duration_col, event_col, covariates=None, penalizer=0.0,
                       ph_p_threshold=0.05, write_path="cox_report.json", emit_insights=True):
    """Fit a lifelines Cox proportional-hazards model and TEST its core assumption.

    Fits `CoxPHFitter` on `df[covariates + [duration_col, event_col]]`, then runs the
    proportional-hazards test (Schoenfeld residuals). Returns a dict with the concordance
    index, per-covariate hazard ratios (exp(coef)) + 95% CI + p-values, the global PH
    p-value, and the list of covariates that VIOLATE proportional hazards at
    `ph_p_threshold`. Writes `cox_report.json`, prints a digest, and — via any live
    `gads_emit_insight` — emits the concordance, the significant hazard ratios, and any PH
    violation (which invalidates constant-hazard-ratio interpretation and calls for
    stratification or a time-varying term). Fail-open.
    """
    import json
    import numpy as np

    result = {"concordance": None, "hazard_ratios": {}, "ph_global_p": None,
              "ph_violations": [], "n": int(len(df)), "report_path": write_path}
    try:
        from lifelines import CoxPHFitter
        from lifelines.statistics import proportional_hazard_test
    except Exception as e:
        print(f"[gads_cox_ph_report] lifelines unavailable ({e}); skipping")
        result["error"] = f"lifelines unavailable: {e}"
        return result

    try:
        if covariates:
            cols = list(covariates) + [duration_col, event_col]
            fit_df = df[cols].copy()
        else:
            fit_df = df.copy()

        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(fit_df, duration_col=duration_col, event_col=event_col)
        result["concordance"] = float(cph.concordance_index_)

        summ = cph.summary  # DataFrame indexed by covariate
        for cov, row in summ.iterrows():
            result["hazard_ratios"][str(cov)] = {
                "hazard_ratio": float(np.exp(row["coef"])),
                "ci_lower": float(np.exp(row["coef lower 95%"])),
                "ci_upper": float(np.exp(row["coef upper 95%"])),
                "p": float(row["p"]),
            }

        # Proportional-hazards assumption test (Schoenfeld residuals).
        try:
            zph = proportional_hazard_test(cph, fit_df, time_transform="rank")
            ph = zph.summary  # per-covariate 'p'
            result["ph_violations"] = [str(c) for c, r in ph.iterrows() if float(r["p"]) < ph_p_threshold]
            result["ph_global_p"] = float(ph["p"].min())
        except Exception as e:
            print(f"[gads_cox_ph_report] PH assumption test unavailable: {e}")

        with open(write_path, "w") as f:
            json.dump({k: v for k, v in result.items() if k != "report_path"}, f, indent=2)

        print(f"[gads_cox_ph_report] n={result['n']}  concordance={result['concordance']:.4f}  "
              f"PH violations: {result['ph_violations'] or 'none'}")
        for cov, hr in result["hazard_ratios"].items():
            sig = "*" if hr["p"] < 0.05 else " "
            print(f"  {sig} {cov}: HR={hr['hazard_ratio']:.3f} "
                  f"[{hr['ci_lower']:.3f}, {hr['ci_upper']:.3f}]  p={hr['p']:.4g}")

        if emit_insights:
            emit = globals().get("gads_emit_insight")
            if callable(emit):
                try:
                    emit(f"Cox model concordance (C-index) = {result['concordance']:.3f} "
                         "(0.5=random, >0.7=good discrimination).")
                except Exception:
                    pass
                for cov, hr in result["hazard_ratios"].items():
                    if hr["p"] < 0.05:
                        direction = "increases" if hr["hazard_ratio"] > 1 else "decreases"
                        try:
                            emit(f"{cov}: hazard ratio {hr['hazard_ratio']:.2f} "
                                 f"(95% CI {hr['ci_lower']:.2f}-{hr['ci_upper']:.2f}, p={hr['p']:.3g}) "
                                 f"— each unit {direction} the instantaneous event risk.")
                        except Exception:
                            pass
                if result["ph_violations"]:
                    try:
                        emit(f"PROPORTIONAL-HAZARDS VIOLATION for {result['ph_violations']}: their "
                             "hazard ratios are NOT constant over time, so the single reported HR is "
                             "misleading. Stratify on, or add a time-interaction for, these covariates.")
                    except Exception:
                        pass
        return result
    except Exception as e:
        print(f"[gads_cox_ph_report] Cox report failed ({type(e).__name__}: {e}); continuing")
        result["error"] = f"{type(e).__name__}: {e}"
        return result
