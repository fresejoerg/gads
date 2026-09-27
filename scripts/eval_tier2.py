"""Stage-level Coder evaluation — tier 2: EXECUTE the code and judge it the way production does.

Tiers 0-1 (eval_coder.py) never run anything, so they cannot support a capability claim
(approach_docs/031 P5, 035 §6). Tier 2 answers the question that matters: would this
generated code have been ACCEPTED if the executor had received it at that point in the run?

For each held-out example (one Coder task from a real run):

  1. copy the project's workspace to a scratch directory, so nothing the evaluated code
     writes can touch the original project's files;
  2. rebuild the kernel as it was just BEFORE that task, by replaying the accepted code of
     the task's completed predecessors in a fresh sandbox session (core/kernel_state.py,
     the same replay the follow-up lane uses);
  3. run the candidate code through production's own acceptance path, in production's order:
       sanitizer (with kernel state) -> parse gate -> kernel-poisoning gate -> runtime-oracle
       bypass -> wrap_for_execution -> sandbox -> state-drift guard -> validate_contract ->
       hallucination guard -> required_metrics probe.
     Everything is imported from the executor/server/hub, not reimplemented.

The instrument validates itself first: the task's own ACCEPTED code (the reference) runs
through the same path. An example whose reference fails is UNRECONSTRUCTABLE (replay drifted,
data moved, a native changed) and is excluded from every model's denominator, and listed, so
the instrument can never charge its own failure to the model. Reference verdicts are cached
per (task, reference code, harness commit).

Beyond the production verdict, a STRICT verdict also requires every `required_variables`
name to be bound in the kernel afterwards. Production does not enforce that at acceptance
(it only freezes them for the drift guard), so strict is reported separately, never merged.

One attempt per example: this scores the first generation, not the retry loop. Sequential by
design: the sandbox is capped at 3 GB and each example owns a kernel. Eval sessions are named
`t2eval-*`, which the backend's stale-session sweep never touches, so a live GADS run cannot
kill them. They do share the sandbox's memory with it, though, so avoid heavy concurrent runs.

The column check inside validate_contract is an LLM call. By default it uses the task's own
assigned model, as production does for that task. When scoring a LOCAL engine, pass
`--validation-model local_model`, so the judge is the one production would use for that
engine. Only a minority of tasks declare required_columns.

    # validate the instrument (no model involved)
    PYTHONPATH=src uv run python scripts/eval_tier2.py --reference-only

    # score saved generations (e.g. from eval_coder.py --tier 1)
    PYTHONPATH=src uv run python scripts/eval_tier2.py \\
        --generations "research/finetune/generations_gemma-4-12b@base.jsonl" --tag gemma-4-12b@base

    # or generate and score in one go
    PYTHONPATH=src uv run python scripts/eval_coder.py --tier 1,2 --tag base
"""
import argparse
import asyncio
import ast
import collections
import contextlib
import hashlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import uuid

sys.path.insert(0, "src")

WORKSPACE_ROOT = "/home/joergf/projects/MyLocalStack/data/workspaces"
REF_CACHE = "research/finetune/tier2_reference_cache.json"
VERDICTS = ("pass", "no_program", "kernel_poisoning", "bypassed", "exec_error", "state_drift",
            "contract", "hallucination", "missing_metrics", "unreconstructable")


def git_rev():
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "src"]) != 0
        return rev + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def sha(text):
    return hashlib.sha256((text or "").encode()).hexdigest()[:16]


@contextlib.contextmanager
def quiet(enabled=True):
    """Production code narrates every step; keep the eval's own output readable."""
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def load_tasks(task_ids):
    """Target tasks plus, per project, every task (for prefix reconstruction). Detached."""
    from sqlmodel import Session, select
    from gads.core.database import engine
    from gads.core.models import Task
    with Session(engine) as s:
        targets = {str(t.id): t for t in s.exec(
            select(Task).where(Task.id.in_([uuid.UUID(t) for t in task_ids]))).all()}
        projects = {t.project_id for t in targets.values()}
        by_project = collections.defaultdict(list)
        for t in s.exec(select(Task).where(Task.project_id.in_(projects))
                        .order_by(Task.created_at)).all():
            by_project[t.project_id].append(t)
        s.expunge_all()
    return targets, by_project


def prefix_tasks(target, project_tasks):
    """Completed predecessors whose code built the kernel this task ran against.

    Same filter as kernel_state.replayable_tasks, cut at the target's creation time. Replans
    re-create steps, so the same step can appear once per attempt; replaying stale versions
    is wasted time (and for a drafted plan, state the target never saw), so keep only the
    LATEST completed version of each step (recipe node id, else description), in run order.
    """
    from gads.core.kernel_state import _SYSTEM_AGENTS
    # A drafted plan resets the kernel on every replan (CLAUDE.md, resume-from-failed-node),
    # so its state starts at the latest Planner row before the target. A recipe-compiled
    # plan (the target carries a recipe_node_id) preserves the kernel across replans, so
    # earlier attempts' completed nodes are genuinely part of its state.
    start = None
    if not (target.postcondition_json or {}).get("recipe_node_id"):
        planners = [t.created_at for t in project_tasks
                    if t.assigned_to == "Planner" and t.created_at < target.created_at]
        start = max(planners) if planners else None
    latest = {}
    for t in project_tasks:
        if start is not None and t.created_at < start:
            continue
        rj = t.result_json or {}
        if (t.created_at >= target.created_at or t.status != "completed"
                or t.assigned_to in _SYSTEM_AGENTS or rj.get("resumed_from_prior_attempt")
                or not (rj.get("code") or "").strip()):
            continue
        key = (t.postcondition_json or {}).get("recipe_node_id") or \
            " ".join((t.description or "").split())[:160].lower()
        latest[key] = t
    return sorted(latest.values(), key=lambda t: t.created_at)


async def judge(sandbox, mgr, target, code, sid, args):
    """Production's acceptance path for one attempt. Returns (verdict, detail, extras)."""
    from gads.core.executor import _sanitize_code, _detect_kernel_poisoning, wrap_for_execution
    from gads.core.execution_hub import ExecutionHub
    from gads.core.runtime_oracle import RuntimeOracle
    from gads.core.server import HALLUCINATION_TOKENS, _probe_kernel_for_metrics
    from gads.core.stdout_clean import clean_stdout

    contract = target.postcondition_json or {}
    with quiet(not args.verbose):
        code = _sanitize_code(code or "", kernel_state=mgr.authoritative_state)
    if not code.strip():
        return "no_program", "empty generation", {}
    try:
        ast.parse(code)
    except SyntaxError as e:
        return "no_program", f"{e.msg} (line {e.lineno})", {}
    poison = _detect_kernel_poisoning(code)
    if poison:
        return "kernel_poisoning", poison[:160], {}

    n_rows = m_cols = 0
    for info in mgr.authoritative_state.values():
        if isinstance(info, dict) and info.get("type") == "DataFrame":
            shape = info.get("shape") or [0, 0]
            n_rows, m_cols = max(n_rows, shape[0]), max(m_cols, shape[1])
    est = RuntimeOracle.estimate_runtime(code, n_rows, m_cols)
    if est > 280.0:
        return "bypassed", f"oracle estimate {est:.0f}s > 280s", {}

    with quiet(not args.verbose):
        wrapped = wrap_for_execution(code)
    res = await asyncio.wait_for(
        sandbox.execute(wrapped, project_id=target.project_id, session_id=sid, workspace_id=sid),
        timeout=args.exec_timeout)
    insights = []
    if "GADS_INSIGHTS_JSON:" in (res.stdout or ""):
        head, tail = res.stdout.split("GADS_INSIGHTS_JSON:", 1)
        try:
            insights = json.loads(tail.strip().split("\n")[0])
        except Exception:
            pass
        res.stdout = head + "\n".join(tail.strip().split("\n")[1:])
    stdout = clean_stdout(res.stdout or "")
    if res.error:
        return "exec_error", f"{res.error.get('ename')}: {str(res.error.get('evalue'))[:160]}", {}

    mgr.authoritative_state.update(res.kernel_state or {})
    own = list(contract.get("required_variables") or [])
    drift = mgr.check_state_drift(own)
    if drift:
        return "state_drift", "; ".join(drift)[:160], {}

    hub = ExecutionHub.__new__(ExecutionHub)   # validate_contract needs no DB session
    with quiet(not args.verbose):
        err = await hub.validate_contract(
            target, stdout, mgr.authoritative_state, semantic_insights=insights,
            validation_model=args.validation_model or target.assigned_to)
    if err:
        return "contract", err[:160], {}
    if any(tok in stdout.lower() for tok in HALLUCINATION_TOKENS):
        return "hallucination", "stdout contains a simulate/skip token", {}

    req_metrics = list(contract.get("required_metrics") or [])
    if req_metrics:
        with quiet(not args.verbose):
            found = await _probe_kernel_for_metrics(sandbox, target.project_id, sid, req_metrics)
        missing = [m for m in req_metrics if m not in found]
        if missing:
            return "missing_metrics", ", ".join(missing), {}

    unbound = [v for v in own if v not in mgr.authoritative_state]
    return "pass", "", {"strict": not unbound, "unbound": unbound}


async def evaluate_one(sandbox, target, project_tasks, code, label, args):
    """Scratch workspace + fresh session + prefix replay, then judge. Always cleans up."""
    from gads.core.executor import ExecutionManager
    from gads.core.kernel_state import replay_script_for_tasks

    sid = f"t2eval-{label}-{str(target.id)[:8]}"
    ws = os.path.join(WORKSPACE_ROOT, sid)
    t0 = time.time()
    out = {"task_id": str(target.id), "seconds": 0.0}
    try:
        shutil.rmtree(ws, ignore_errors=True)
        src = os.path.join(WORKSPACE_ROOT, str(target.project_id))
        if os.path.isdir(src):
            shutil.copytree(src, ws, symlinks=True)   # datasets are symlinks; keep them so
        else:
            os.makedirs(ws)
        await sandbox.reset_session(sid)

        mgr = ExecutionManager.__new__(ExecutionManager)
        mgr.authoritative_state, mgr.protected_state = {}, {}
        prefix = prefix_tasks(target, project_tasks)
        out["prefix_tasks"] = len(prefix)
        if prefix:
            with quiet(not args.verbose):
                script = replay_script_for_tasks(prefix)
            res = await asyncio.wait_for(
                sandbox.execute(script, project_id=target.project_id, session_id=sid,
                                workspace_id=sid),
                timeout=args.replay_timeout)
            if res.error:
                out.update(verdict="unreconstructable",
                           detail=f"replay failed: {res.error.get('ename')}: "
                                  f"{str(res.error.get('evalue'))[:140]}")
                return out
            mgr.authoritative_state.update(res.kernel_state or {})
            for t in prefix:
                mgr.record_produced_state((t.postcondition_json or {}).get("required_variables") or [])

        verdict, detail, extras = await judge(sandbox, mgr, target, code, sid, args)
        out.update(verdict=verdict, detail=detail, **extras)
    except asyncio.TimeoutError:
        out.update(verdict="exec_error", detail="TimeoutError: eval wall-clock limit")
    except Exception as e:
        out.update(verdict="unreconstructable", detail=f"harness: {type(e).__name__}: {str(e)[:140]}")
    finally:
        with contextlib.suppress(Exception):
            await sandbox.reset_session(sid)
        shutil.rmtree(ws, ignore_errors=True)
        out["seconds"] = round(time.time() - t0, 1)
    return out


async def run_examples(rows, codes, label, args, targets, by_project):
    from gads.tools.sandbox import SandboxClient
    sandbox = SandboxClient()
    results = []
    try:
        for i, (r, code) in enumerate(zip(rows, codes), 1):
            target = targets.get(r["task_id"])
            if target is None:
                res = {"task_id": r["task_id"], "verdict": "unreconstructable",
                       "detail": "task row no longer in the DB", "seconds": 0.0}
            else:
                res = await evaluate_one(sandbox, target, by_project[target.project_id],
                                         code, label, args)
            res["spec"] = r.get("spec")
            results.append(res)
            print(f"  [{label} {i}/{len(rows)}] {res['verdict']:17s} {res['seconds']:6.1f}s  "
                  f"{(r.get('spec') or '')[:28]:28s} {res.get('detail', '')[:70]}", flush=True)
    finally:
        await sandbox.close()
    return results


def harness_key():
    """Everything a reference verdict depends on: GADS source + this script's own logic."""
    return f"{git_rev()}|{sha(open(__file__).read())[:8]}"


def reference_verdicts(rows, args, targets, by_project):
    """Reference (accepted-code) verdicts, from cache when the harness has not changed."""
    rev = harness_key()
    cache = {}
    if os.path.exists(REF_CACHE) and not args.refresh_reference:
        with contextlib.suppress(Exception):
            cache = json.load(open(REF_CACHE))
    from eval_coder import unwrap
    todo, out = [], {}
    for r in rows:
        ref = unwrap(r["messages"][-1]["content"])
        key = f"{r['task_id']}|{sha(ref)}|{rev}"
        if key in cache and "+dirty" not in rev:
            out[r["task_id"]] = cache[key]
        else:
            todo.append((r, ref, key))
    if todo:
        print(f"\nREFERENCE pass (validates the instrument): {len(todo)} example(s) "
              f"at harness {rev}", flush=True)
        res = asyncio.run(run_examples([t[0] for t in todo], [t[1] for t in todo], "ref",
                                       args, targets, by_project))
        for (r, _, key), v in zip(todo, res):
            out[r["task_id"]] = v
            cache[key] = v
        pathlib.Path(REF_CACHE).parent.mkdir(parents=True, exist_ok=True)
        json.dump(cache, open(REF_CACHE, "w"), indent=1)
    return out


def aggregate(results, ref):
    """Rates over RECONSTRUCTABLE examples only: those whose reference passes.

    A candidate can still come back `unreconstructable` (its prefix replay failed this time:
    sandbox OOM, a timeout). The reference proved the replay works, so that is a harness
    event, never a model failure. It is excluded from the denominator and counted apart.
    """
    ok_ids = {tid for tid, v in ref.items() if v.get("verdict") == "pass"}
    harness_events = [x for x in results
                      if x["task_id"] in ok_ids and x["verdict"] == "unreconstructable"]
    scored = [x for x in results
              if x["task_id"] in ok_ids and x["verdict"] != "unreconstructable"]
    counts = collections.Counter(x["verdict"] for x in scored)
    n = len(scored)
    strict = sum(1 for x in scored if x["verdict"] == "pass" and x.get("strict"))
    return {
        "n_examples": len(results), "n_reconstructable": n,
        "pass": counts.get("pass", 0),
        "rate": round(counts.get("pass", 0) / n, 3) if n else None,
        "strict_pass": strict, "strict_rate": round(strict / n, 3) if n else None,
        "verdicts": {v: counts[v] for v in VERDICTS if counts.get(v)},
        "excluded": sorted(set(x["task_id"] for x in results) - ok_ids),
        "harness_events": [x["task_id"] for x in harness_events],
    }


def run_tier2(rows, codes, tag, args):
    """Entry point shared with eval_coder.py. `codes` None = score the references."""
    targets, by_project = load_tasks({r["task_id"] for r in rows})
    ref = reference_verdicts(rows, args, targets, by_project)
    bad = [(tid, v) for tid, v in ref.items() if v.get("verdict") != "pass"]
    print(f"\n  instrument: {len(rows) - len(bad)}/{len(rows)} examples reconstructable "
          f"(reference code accepted)")
    for tid, v in bad[:10]:
        print(f"    excluded {tid[:8]}  {v.get('verdict')}: {v.get('detail', '')[:90]}")

    if codes is None:
        results = [dict(ref[r["task_id"]], spec=r.get("spec")) for r in rows]
    else:
        print(f"\nCANDIDATE pass: {tag}", flush=True)
        results = asyncio.run(run_examples(rows, codes, "cand", args, targets, by_project))
    agg = aggregate(results, ref)
    agg["harness"] = harness_key()
    if agg["harness_events"]:
        print(f"  ! {len(agg['harness_events'])} candidate replay(s) failed this run: harness "
              f"events, excluded from the rate: {[t[:8] for t in agg['harness_events']]}")

    print(f"\n  TIER 2 — {tag}: {agg['pass']}/{agg['n_reconstructable']} accepted"
          + (f" ({agg['rate']:.1%})" if agg["rate"] is not None else "")
          + f" | strict {agg['strict_pass']}/{agg['n_reconstructable']}")
    for v, c in agg["verdicts"].items():
        print(f"    {v:18s} {c}")
    by_spec = collections.defaultdict(lambda: [0, 0])
    ok_ids = {tid for tid, v in ref.items() if v.get("verdict") == "pass"}
    for x in results:
        if x["task_id"] in ok_ids and x["verdict"] != "unreconstructable":
            by_spec[x["spec"]][1] += 1
            by_spec[x["spec"]][0] += x["verdict"] == "pass"
    for spec, (p, n) in sorted(by_spec.items()):
        print(f"    {spec[:40]:40s} {p}/{n}")

    detail_path = pathlib.Path(args.out).parent / f"tier2_{tag.replace('/', '_')}.jsonl"
    with open(detail_path, "w") as f:
        for x in results:
            f.write(json.dumps(x) + "\n")
    print(f"  per-example verdicts -> {detail_path}")
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", default="research/finetune/sft_holdout.jsonl")
    ap.add_argument("--generations", help="jsonl of {task_id, generation} to score")
    ap.add_argument("--reference-only", action="store_true",
                    help="score the accepted code itself: validates the instrument")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--refresh-reference", action="store_true",
                    help="ignore the cached reference verdicts")
    ap.add_argument("--validation-model", default=None,
                    help="model for validate_contract's semantic column check; default is "
                         "the task's own assigned model, as in production")
    ap.add_argument("--replay-timeout", type=float, default=900.0)
    ap.add_argument("--exec-timeout", type=float, default=720.0)
    ap.add_argument("--verbose", action="store_true", help="show production code's own logging")
    ap.add_argument("--out", default="research/finetune/eval_report.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.holdout)]
    if args.limit:
        rows = rows[:args.limit]
    if args.reference_only:
        codes, tag = None, args.tag or "reference"
    elif args.generations:
        from eval_coder import unwrap
        gens = {g["task_id"]: g.get("generation") or ""
                for g in map(json.loads, open(args.generations))}
        codes = [unwrap(gens.get(r["task_id"], "")) for r in rows]
        tag = args.tag or pathlib.Path(args.generations).stem.replace("generations_", "")
    else:
        sys.exit("pass --reference-only or --generations FILE "
                 "(or use eval_coder.py --tier 2 to generate and score)")

    print(f"holdout: {len(rows)} examples over {len({r.get('spec') for r in rows})} spec(s) "
          f"| tag={tag}")
    agg = run_tier2(rows, codes, tag, args)
    report = {"tag": tag, "n": len(rows), "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "tier2": agg}
    with open(args.out, "a") as f:
        f.write(json.dumps(report) + "\n")
    print(f"\nappended -> {args.out}")


if __name__ == "__main__":
    main()
