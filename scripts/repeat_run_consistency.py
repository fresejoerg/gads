"""Repeat-run consistency: does the same spec, on the same engine, give the same answer?

Reproducibility is one of this project's two stated research metrics, but it has only ever
been checked WITHIN a run (a postcondition holds, a metric is captured). Nothing measures
whether re-running an identical specification produces an identical conclusion. That is the
question a decision-maker actually cares about, and the one "analytical drift" names.

The design isolates the model's own variability:

  * ONE spec, launched N times, nothing else changed.
  * A recipe-compiled (D3+) spec by preference, so the DAG is FIXED and plan-level variation
    cannot contaminate the measurement — what varies is only what the model chose inside
    each node.
  * Sequential, never parallel: one GPU and one kernel session, so concurrent runs would
    contend and the variance measured would be the scheduler's, not the model's.
  * `engine_id` is recorded per run and the report REFUSES to pool across engines. Swapping
    models in LM Studio mid-experiment is easy and leaves no other trace (observed
    2026-09-04); pooling across that silently measures the wrong thing.

Metrics are reported separately as:

  PROCESS  — what the model chose (rows sampled, candidates tried, trials run)
  OUTCOME  — what it concluded (scores, effects, errors)

The distinction matters: existing data on `adult_model_selection` shows process metrics
varying by >100% while outcome metrics move under 1%. A system can be wildly inconsistent
in method and stable in conclusion, or the reverse, and only the second is alarming.

    # analyse runs that already exist, no compute
    PYTHONPATH=src uv run python scripts/repeat_run_consistency.py --spec adult_model_selection.md --analyse-only

    # launch 5 fresh local runs, then analyse
    PYTHONPATH=src uv run python scripts/repeat_run_consistency.py --spec adult_model_selection.md --runs 5
"""
import argparse
import collections
import json
import os
import statistics as st
import sys
import time

sys.path.insert(0, "src")

import httpx
from sqlalchemy import create_engine, text

BACKEND = os.getenv("GADS_BACKEND", "http://localhost:8001")
LEDGER = "research/dial_ledger.jsonl"

# Metrics describing HOW the analysis was done rather than what it concluded. Matched by
# substring; anything unmatched is treated as an outcome, because mislabelling an outcome
# as process would hide exactly the instability this script exists to find.
PROCESS_HINTS = ("n_train", "n_rows", "n_sample", "n_candidates", "n_trials", "n_features",
                 "n_series", "prediction_length", "n_classes", "n_folds", "rows_used")


def db():
    url = os.environ.get("GADS_DATABASE_URL")
    if not url:
        sys.exit("GADS_DATABASE_URL is required.")
    return create_engine(url, pool_pre_ping=True)


def ledger_index():
    idx = {}
    if os.path.exists(LEDGER):
        for line in open(LEDGER):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("project_id"):
                idx[d["project_id"]] = d
    return idx


def collect(engine, spec):
    """Every run of `spec`, with its metrics and provenance."""
    idx = ledger_index()
    runs = {}
    with engine.connect() as c:
        for pid, created, eng, narr in c.execute(text(
                "select id::text, created_at, last_state_json->>'engine_id',"
                "       coalesce(narrative,'') from project"
                " where last_state_json->>'spec_filename' = :s order by created_at"), {"s": spec}):
            runs[pid] = {"project_id": pid, "created": str(created)[:19], "metrics": {},
                         # Engine from the PROJECT row: stamped at launch, so it survives a
                         # run that halted before the ledger was written. The ledger is only
                         # a fallback for runs predating that.
                         "engine_row": eng,
                         "halted": narr.startswith("[HALTED]")}
        if not runs:
            return []
        for pid, m in c.execute(text(
                "select project_id::text, result_json->'metrics_captured' from task"
                " where project_id::text = any(:ids)"
                " and result_json->'metrics_captured' is not null"),
                {"ids": list(runs)}):
            for k, v in (m or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    runs[pid]["metrics"][k] = float(v)
    for pid, r in runs.items():
        d = idx.get(pid, {})
        r.update(mode=d.get("routing_mode"), rung=d.get("rung"),
                 engine=r.get("engine_row") or d.get("engine_id"),
                 outcome=d.get("outcome") or ("halted" if r.get("halted") else None))
    return list(runs.values())


def launch(spec):
    r = httpx.post(f"{BACKEND}/projects/from-spec", json={"filename": spec}, timeout=180.0)
    r.raise_for_status()
    return r.json()["project"]["id"]


def wait(engine, pid, poll=60, timeout_s=4 * 3600, ledger_grace_s=600):
    """Block until the workflow settles. A run that never settles is reported, not silently
    counted as a result.

    Settled = the run wrote its dial-ledger record (every normal ending does, pass or fail),
    or it was halted/cancelled, or it has a final narrative and nothing running for longer
    than `ledger_grace_s` (a ledger write that failed). Leftover `pending` tasks are NOT a
    liveness signal: replans strand never-run tasks, and waiting on them made every failed
    run of 2026-09-09 look like a 2-hour timeout."""
    t0 = time.time()
    quiet_since = None
    while time.time() - t0 < timeout_s:
        with engine.connect() as c:
            narr = c.execute(text("select narrative from project where id=:p"), {"p": pid}).scalar()
            st_counts = {r[0]: r[1] for r in c.execute(text(
                "select status, count(*) from task where project_id=:p group by 1"), {"p": pid})}
        if pid in ledger_index():
            return True, st_counts
        if narr is not None and not st_counts.get("running"):
            if narr.startswith(("[HALTED]", "[CANCELLED]")):
                return True, st_counts
            quiet_since = quiet_since or time.time()
            if time.time() - quiet_since > ledger_grace_s:
                return True, st_counts
        else:
            quiet_since = None
        time.sleep(poll)
    return False, {}


def report(runs, spec, min_runs):
    """Variance per metric, split process vs outcome, never pooled across engines."""
    by_engine = collections.defaultdict(list)
    for r in runs:
        by_engine[(r.get("mode"), r.get("engine"))].append(r)

    print(f"\n{'='*74}\nREPEAT-RUN CONSISTENCY — {spec}\n{'='*74}")
    print(f"{len(runs)} run(s) total across {len(by_engine)} (mode, engine) group(s)\n")

    for (mode, eng), group in sorted(by_engine.items(), key=lambda x: str(x[0])):
        tag = f"mode={mode or '?'} engine={eng or 'UNSTAMPED'}"
        print(f"--- {tag} — {len(group)} run(s) ---")
        if eng is None:
            print("    ! engine unstamped (pre-2026-09-04): these runs may mix engines and")
            print("      cannot support a consistency claim. Reported, not pooled.")
        if len(group) < min_runs:
            print(f"    (fewer than {min_runs} runs — shown for completeness, not analysed)\n")
            continue

        vals = collections.defaultdict(list)
        for r in group:
            for k, v in r["metrics"].items():
                vals[k].append(v)

        for kind, want_process in (("PROCESS (how it was done)", True),
                                   ("OUTCOME (what it concluded)", False)):
            keys = [k for k in sorted(vals)
                    if (any(h in k for h in PROCESS_HINTS)) == want_process
                    and len(vals[k]) >= min_runs]
            if not keys:
                continue
            print(f"    {kind}")
            print(f"      {'metric':26s} {'n':>2s} {'identical':>9s} {'min':>11s} {'max':>11s} {'CV':>7s}")
            for k in keys:
                v = vals[k]
                mean = sum(v) / len(v)
                cv = (st.pstdev(v) / abs(mean)) if mean else 0.0
                ident = "yes" if len(set(v)) == 1 else f"{len(set(v))} vals"
                print(f"      {k[:26]:26s} {len(v):2d} {ident:>9s} "
                      f"{min(v):11.5g} {max(v):11.5g} {cv:7.2%}")
        outs = collections.Counter(r.get("outcome") for r in group)
        print(f"    workflow outcome: {dict(outs)}\n")

    print("Reading this: identical process + identical outcome is full reproducibility.")
    print("Varying process + stable outcome means the method drifts but the conclusion")
    print("holds — tolerable, and worth knowing. Varying OUTCOME is the alarming case,")
    print("and a sign flip in a causal effect is the worst version of it.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="spec filename, e.g. adult_model_selection.md")
    ap.add_argument("--runs", type=int, default=0, help="fresh runs to launch (0 = analyse only)")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--min-runs", type=int, default=3,
                    help="minimum runs in a group before variance is reported (default 3)")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--timeout-h", type=float, default=4.0,
                    help="per-run wall-clock cap in hours (default 4)")
    ap.add_argument("--out", default="research/consistency")
    args = ap.parse_args()

    engine = db()
    n = 0 if args.analyse_only else args.runs

    if n:
        try:
            cfg = httpx.get(f"{BACKEND}/config", timeout=20.0).json()
        except Exception as e:
            sys.exit(f"backend unreachable at {BACKEND}: {type(e).__name__}")
        print(f"routing_mode={cfg.get('routing_mode')} | launching {n} sequential run(s) of "
              f"{args.spec}\n(sequential by design: one GPU, one kernel session)\n")
        for i in range(1, n + 1):
            pid = launch(args.spec)
            print(f"  [{i}/{n}] {pid} launched ...", flush=True)
            ok, counts = wait(engine, pid, poll=args.poll, timeout_s=args.timeout_h * 3600)
            print(f"  [{i}/{n}] {'settled' if ok else 'TIMED OUT'} {counts}", flush=True)
            if not ok:
                # The run may still be executing. Launching the next one would put two
                # workflows on one GPU and one kernel — the contention this harness exists
                # to exclude — so stop here and analyse what settled.
                print(f"  stopping: run {pid} did not settle; not launching the rest", flush=True)
                break

    runs = collect(engine, args.spec)
    if not runs:
        sys.exit(f"no runs found for {args.spec}")
    report(runs, args.spec, args.min_runs)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.spec.replace('.md','')}.json")
    with open(path, "w") as f:
        json.dump({"spec": args.spec, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "runs": runs}, f, indent=2)
    print(f"\nraw run records -> {path}")


if __name__ == "__main__":
    main()
