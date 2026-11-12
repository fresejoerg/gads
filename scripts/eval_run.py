#!/usr/bin/env python3
"""Score GADS runs with the code-only run scorecard (src/gads/eval/).

One eval object per execution, written to research/evals/<project_id>.json (overwritten on
re-score). No LLM calls. Needs GADS_DATABASE_URL (reads Postgres) and the workspaces.

Usage:
    PYTHONPATH=src uv run python scripts/eval_run.py <project_id> [<project_id> ...]
    PYTHONPATH=src uv run python scripts/eval_run.py --all            # backfill every project
    PYTHONPATH=src uv run python scripts/eval_run.py <pid> --show     # print the full card
    PYTHONPATH=src uv run python scripts/eval_run.py --all --no-write --parity

    --parity   check the scorecard's pass@model attribution against every dial-ledger row
               (the two share dial.node_attribution, so any mismatch is a loader bug, e.g.
               a follow-up task leaking into the execution). Rows whose project has since
               had follow-ups or was re-run are reported, not failed.

Project ids may be given as unique prefixes (8 hex chars is enough).
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _resolve(prefixes):
    import sqlalchemy as sa
    from sqlmodel import Session, select
    from gads.core.database import engine
    from gads.core.models import Project
    with Session(engine) as s:
        if not prefixes:
            return [str(p) for p in s.exec(select(Project.id).order_by(Project.created_at)).all()]
        out = []
        for pre in prefixes:
            hits = s.exec(select(Project.id).where(Project.id.cast(sa.String).like(f"{pre}%"))).all()
            if len(hits) != 1:
                sys.exit(f"{pre}: {len(hits)} matching projects")
            out.append(str(hits[0]))
        return out


def _line(ev):
    c, r = ev["critical"], ev["run"]
    b = ev.get("benchmark")
    a = ev.get("attribution") or {}
    bench = f" bench={b['verdict']} {b['passed']}/{b['total']}" if b else ""
    pam = a.get("pass_at_model")
    return (f"{ev['project_id'][:8]}  {ev['verdict']:<13} crit {c['pass']}/{sum(c.values())}"
            f" (fail {c['fail']}, n/a {c['not_evaluable']})  {r['terminal_state']:<11}"
            f" att={r['workflow_attempts']} nodes={r['exec_nodes']} pass@model={pam}{bench}"
            f"  {(ev['name'] or '')[:40]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("projects", nargs="*", help="project ids or unique prefixes")
    ap.add_argument("--all", action="store_true", help="score every project in the DB")
    ap.add_argument("--no-write", action="store_true", help="do not write research/evals/")
    ap.add_argument("--show", action="store_true", help="print each full scorecard as JSON")
    ap.add_argument("--failures", action="store_true", help="list failed checks per run")
    ap.add_argument("--parity", action="store_true", help="compare attribution with the dial ledger")
    args = ap.parse_args()
    if not args.projects and not args.all:
        ap.error("give project ids or --all")

    from gads.eval.scorecard import _dial_row, evaluate_run, write_scorecard

    pids = _resolve([] if args.all else args.projects)
    tally, parity_bad, parity_n, errors = {}, [], 0, 0
    for pid in pids:
        try:
            ev = evaluate_run(pid)
        except Exception as e:
            errors += 1
            print(f"{pid[:8]}  ERROR {type(e).__name__}: {e}")
            continue
        tally[ev["verdict"]] = tally.get(ev["verdict"], 0) + 1
        if not args.no_write:
            write_scorecard(ev)
        print(_line(ev))
        if args.failures:
            for c in ev["checks"]:
                if c["status"] == "fail":
                    print(f"      ✗ {c.get('node') or 'run'} · {c['id']}: {c['detail']}")
        if args.show:
            print(json.dumps(ev, indent=1))
        if args.parity:
            row = _dial_row(pid)
            if row:
                parity_n += 1
                keys = ("exec_nodes", "attempted_nodes", "model_pass", "fallback_pass",
                        "native_primary", "pass_at_model")
                # failed_nodes only exists in rows since 2026-09-15; older rows used a
                # completed-only base, so their attempted_nodes/pass_at_model differ by design.
                if "failed_nodes" not in row:
                    keys = ("exec_nodes", "model_pass", "fallback_pass", "native_primary")
                diff = {k: (row.get(k), ev["attribution"].get(k)) for k in keys
                        if row.get(k) is not None and row.get(k) != ev["attribution"].get(k)}
                if diff:
                    parity_bad.append((pid, diff, ev["run"]["excluded_followup_tasks"]))

    print(f"\n{len(pids)} run(s): " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items()))
          + (f", errors={errors}" if errors else ""))
    if args.parity:
        print(f"parity vs dial ledger: {parity_n - len(parity_bad)}/{parity_n} rows match")
        for pid, diff, fu in parity_bad:
            print(f"  ≠ {pid[:8]} {diff}" + (f"  ({fu} follow-up task(s) excluded)" if fu else ""))
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
