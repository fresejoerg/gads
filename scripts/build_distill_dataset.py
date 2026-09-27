"""Build a distillation training snapshot from everything captured so far (approach_docs/035).

Sources, joined and de-duplicated by task (capture wins over backfill):

  capture   research/finetune/capture/{attempts,accepts}-*.jsonl, written by every run since
            core/distill_capture.py landed. An accepted example is the attempt whose executed
            code matches the task's accept row. A task completed by resume-from-failed-node,
            or finished after a backend restart, has an accept but no attempt in this
            process's capture; it lands in `accept_no_matching_attempt` by design.
  backfill  the Langfuse harvest already on disk (harvest_coder_traces.py outputs: sft_*.jsonl,
            dpo_train.jsonl, manifest.jsonl), so the snapshot is not empty on day one.

Every example is RE-RENDERED into the format the local student is actually served: the
Coder's raw-code mode (agents/workers/coder.render_messages), whatever engine produced it.
Cloud runs are recorded under the JSON-envelope instruction, and training a local model on
that would teach a format it never sees (035 §3b). Targets are fenced ```python blocks,
which is what raw-code mode asks for.

Also written per snapshot:
  router_{train,holdout}.jsonl      Router calls from the capture stream (calls-*.jsonl) whose
                                    output matches the spec's gold taxonomy + recipe (the
                                    ground truth scripts/eval_routing.py scores against). The
                                    Router reaches cloud and local models through the same
                                    structured-completion path, so its messages ARE the
                                    student's format.
  sft_*_noskills.jsonl              the capture examples with the skills block replaced by
                                    the Coder's no-skills placeholder: the input for the
                                    skills-into-weights experiment (035 §7).

Classes (035 §2):
  teacher_demo     a cloud model's accepted code                               -> SFT
  teacher_onpol    cloud fallback accepted after the local model failed it     -> SFT + DPO pair
  student_success  local_model's accepted code                                 -> SFT
  student_pair     local failed attempt -> later accepted local attempt        -> DPO pair
  excluded_native  completed by a native function, not a model                 -> dropped

Guards:
  * Split by SPEC, never by task. Held-out specs are stable as data grows: the specs of the
    original held-out file, every benchmark spec (never train on the instrument, 031 §4),
    and a fixed hash bucket of everything else.
  * Sanitizer: the executed code is the target only when it differs from what the model
    wrote by whitespace/imports at most. Otherwise the example is dropped: the student
    should learn what a model can write, not what the harness patched (035 §3d).
  * Per (recipe, node) cap, so the most-run workflow does not dominate.
  * Cloud examples are private-research data (031 §4a): `--no-cloud` builds without them.

    PYTHONPATH=src uv run python scripts/build_distill_dataset.py            # build + stats
    PYTHONPATH=src uv run python scripts/build_distill_dataset.py --stats    # stats only
    scripts/distill.sh --model unsloth/gemma-3-12b-it-unsloth-bnb-4bit       # build + train
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import pathlib
import re
import sys
import time

sys.path.insert(0, "src")

FT = "research/finetune"
CAPTURE = os.environ.get("GADS_DISTILL_CAPTURE_DIR", f"{FT}/capture")
OUT_ROOT = f"{FT}/datasets"
# The held-out specs of the first harvest; pinned so every snapshot shares the eval set that
# eval_coder / eval_tier2 baselines were measured on.
PINNED_HOLDOUT = {"gbsg2_survival_cox.md", "jobs_lalonde_training.md",
                  "thornton_hiv_incentive.md", "trainee_program_earnings.md"}
NATIVE_PREFIXES = ("native_fallback:", "native_primary:")
FENCE = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL)


def read_jsonl(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def _spec_datasets(path):
    """The `datasets:` list from a spec's YAML frontmatter."""
    try:
        import yaml
        text = open(path, encoding="utf-8").read()
        if text.startswith("---"):
            return set((yaml.safe_load(text.split("---")[1]) or {}).get("datasets") or [])
    except Exception:
        pass
    return set()


def benchmark_specs(strict_dataset=False):
    """Specs that must never be trained on, because they are (or restate) a benchmark.

    Always: each benchmark's spec file and its delegation-dial variants (`<stem>_d0.md` …
    `<stem>_d5.md`, `_d3ols`, `_d3skl`), which ask the same question of the same data.
    With `strict_dataset`: also every spec that uses a benchmark's DATASET, even for a
    different question (e.g. the Adult model-selection specs vs. the amlb_adult benchmark).
    """
    stems, datasets = set(), set()
    for p in glob.glob("research/benchmarks/*/expected.json"):
        try:
            spec = json.load(open(p)).get("spec_file")
        except Exception:
            spec = None
        if spec:
            stems.add(os.path.splitext(os.path.basename(spec))[0])
        datasets |= _spec_datasets(os.path.join(os.path.dirname(p), "spec.md"))
    out = set()
    for path in glob.glob("specs/*.md"):
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]
        if stem in stems or any(re.fullmatch(re.escape(b) + r"_d\d\w*", stem) for b in stems):
            out.add(name)
        elif strict_dataset and _spec_datasets(path) & datasets:
            out.add(name)
    return out | {s + ".md" for s in stems}


def is_local(model):
    return (model or "").startswith("local_model")


def unfence(text):
    m = FENCE.search(text or "")
    return (m.group(1) if m else (text or "")).strip()


def fenced(code):
    return f"```python\n{code.strip()}\n```"


def _norm_for_sanitizer(code):
    lines = []
    for ln in (code or "").splitlines():
        s = ln.strip()
        if not s or s.startswith("```") or s.startswith("import ") or s.startswith("from "):
            continue
        lines.append(s)
    return lines


def sanitizer_status(raw, executed):
    """identical | trivial (whitespace/imports only) | rewritten."""
    if raw is None:
        return "n/a"
    r = unfence(raw)
    if r.strip() == (executed or "").strip():
        return "identical"
    return "trivial" if _norm_for_sanitizer(r) == _norm_for_sanitizer(executed) else "rewritten"


def core_from_rendered(messages):
    """Recover (system core, user core) from a recorded Coder request in either format."""
    from gads.agents.workers.coder import RAW_SYSTEM_SUFFIX, RAW_USER_SUFFIX, JSON_USER_SUFFIX
    sys_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
    usr_msg = next((m["content"] for m in messages if m["role"] == "user"), None)
    if sys_msg is None or usr_msg is None:
        return None
    if usr_msg.endswith(JSON_USER_SUFFIX):
        return sys_msg, usr_msg[:-len(JSON_USER_SUFFIX)]
    if usr_msg.endswith(RAW_USER_SUFFIX) and sys_msg.endswith(RAW_SYSTEM_SUFFIX):
        return sys_msg[:-len(RAW_SYSTEM_SUFFIX)], usr_msg[:-len(RAW_USER_SUFFIX)]
    return None


def classify(model_used):
    m = model_used or ""
    if m.startswith(NATIVE_PREFIXES):
        return "excluded_native"
    if m.startswith("cloud_fallback:"):
        return "teacher_onpol"
    if m.startswith("teacher_assist:"):
        return "teacher_onpol"
    return "student_success" if is_local(m) else "teacher_demo"


def from_capture(stats):
    """Examples and pairs from the always-on capture streams."""
    attempts = read_jsonl(f"{CAPTURE}/attempts-*.jsonl")
    accepts = read_jsonl(f"{CAPTURE}/accepts-*.jsonl")
    by_task = collections.defaultdict(list)
    for a in attempts:
        if a.get("task_id"):
            by_task[a["task_id"]].append(a)
    for v in by_task.values():
        v.sort(key=lambda a: a.get("ts", 0))
    stats["capture_attempts"] = len(attempts)
    stats["capture_accepts"] = len(accepts)

    examples, pairs = [], []
    latest_accept = {}
    for acc in accepts:                       # a task can be accepted once per run; keep last
        latest_accept[acc["task_id"]] = acc
    for tid, acc in latest_accept.items():
        cls = classify(acc.get("model_used"))
        if cls == "excluded_native":
            stats["excluded_native"] += 1
            continue
        tries = by_task.get(tid, [])
        hit = [a for a in tries if a.get("executed_sha") == acc.get("executed_sha")
               and a.get("outcome") == "executed"]
        if not hit or not hit[-1].get("system"):
            stats["accept_no_matching_attempt"] += 1
            continue
        chosen = hit[-1]
        san = sanitizer_status(chosen.get("raw_output"), chosen.get("executed_code"))
        base = {
            "task_id": tid, "source": "capture", "cls": cls,
            "spec": acc.get("spec"), "project_id": acc.get("project_id"),
            "model_used": acc.get("model_used"), "engine_id": acc.get("engine_id"),
            "prompt_version": acc.get("prompt_version"), "recipe_id": acc.get("recipe_id"),
            "recipe_node_id": acc.get("recipe_node_id"), "mode": acc.get("mode"),
            "required_variables": acc.get("required_variables") or [],
            "attempt_index": tries.index(chosen) + 1, "n_attempts": len(tries),
            "sanitizer": san,
            "system": chosen["system"], "user_core": chosen["user_core"],
            "skills_context": chosen.get("skills_context"),
            "code": chosen["executed_code"],
        }
        examples.append(base)
        # Preference pairs: the same task's earlier failures by a LOCAL engine are the
        # rejected side (on-policy for the student); the accepted code is chosen.
        for bad in tries[:tries.index(chosen)]:
            if not is_local(bad.get("model")) or not bad.get("raw_output") or not bad.get("system"):
                continue
            # On-policy means the rejected side came from the engine the student IS: the
            # engine serving when the task ran (the accept's engine_id), for student pairs
            # and teacher corrections alike.
            if bad.get("engine_id") and base["engine_id"] and bad["engine_id"] != base["engine_id"]:
                continue
            pairs.append({**base, "cls": "teacher_onpol" if cls == "teacher_onpol" else "student_pair",
                          "system": bad["system"], "user_core": bad["user_core"],
                          "rejected": unfence(bad["raw_output"]),
                          "rejected_outcome": bad.get("outcome")})
    return examples, pairs


def from_backfill(stats):
    """The Langfuse harvest already on disk, re-rendered to the student's format."""
    man = {r["task_id"]: r for r in read_jsonl(f"{FT}/manifest.jsonl")}
    examples, pairs = [], []
    for fname in ("sft_train.jsonl", "sft_holdout.jsonl"):
        for r in read_jsonl(f"{FT}/{fname}"):
            core = core_from_rendered(r["messages"])
            if core is None:
                stats["backfill_unrenderable"] += 1
                continue
            m = man.get(r["task_id"], {})
            cls = classify(m.get("model_used"))
            if cls == "excluded_native":
                stats["excluded_native"] += 1
                continue
            examples.append({
                "task_id": r["task_id"], "source": "harvest", "cls": cls, "spec": r.get("spec"),
                "project_id": None, "model_used": m.get("model_used"),
                "engine_id": m.get("engine_id"), "prompt_version": m.get("prompt_version"),
                "recipe_id": None, "recipe_node_id": None, "mode": "workflow",
                "required_variables": _as_list(r.get("required_variables")),
                "attempt_index": m.get("attempt"), "n_attempts": m.get("n_attempts"),
                "sanitizer": "n/a", "system": core[0], "user_core": core[1],
                "code": unfence(r["messages"][-1]["content"]),
            })
    # The harvested pairs carry no task id or spec, so the split cannot see them. Recover the
    # spec from the TASK line, which every attempt at a task repeats verbatim; a pair that
    # cannot be attributed is dropped, because an unattributed pair may be benchmark data.
    task_line_spec = {}
    for e in examples:
        tl = _task_line(e["user_core"])
        if tl:
            task_line_spec.setdefault(tl, (e.get("spec"), e["task_id"]))
    for r in read_jsonl(f"{FT}/dpo_train.jsonl"):
        core = core_from_rendered(r["messages"])
        if core is None:
            stats["backfill_unrenderable"] += 1
            continue
        spec, tid = task_line_spec.get(_task_line(core[1]), (None, None))
        if tid is None:
            stats["dropped_unattributed_pair"] += 1
            continue
        pairs.append({"task_id": tid, "source": "harvest", "cls": "student_pair",
                      "spec": spec, "model_used": "local_model",
                      "system": core[0], "user_core": core[1],
                      "code": unfence(r["chosen"]), "rejected": unfence(r["rejected"]),
                      "required_variables": [], "recipe_id": None, "recipe_node_id": None})
    return examples, pairs


def spec_gold():
    """{spec filename: gold routing labels}, the same ground truth scripts/eval_routing.py
    scores against: the spec's declared taxonomy (many-valued) and its pinned recipe, unless
    that pin is a pin-only research arm (an experimental condition, not a routing target)."""
    import yaml
    from gads.core.knowledge import KnowledgeRegistry
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        registry = KnowledgeRegistry("src/gads/knowledge/recipes")
    fm_re = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    gold = {}
    for path in glob.glob("specs/*.md"):
        m = fm_re.search(open(path, encoding="utf-8").read())
        if not m:
            continue
        try:
            fm = yaml.safe_load(m.group(1)) or {}
        except Exception:
            continue
        tax = fm.get("taxonomy") or {}
        if not tax:
            continue
        as_list = lambda v: [str(x) for x in v] if isinstance(v, list) else ([str(v)] if v else [])
        recipe = fm.get("recipe_id")
        rec = registry.get_recipe(recipe) if recipe else None
        arm = bool(rec and registry.is_pin_only(rec.applies_when))
        gold[os.path.basename(path)] = {"tasks": as_list(tax.get("task")),
                                        "modalities": as_list(tax.get("modality")),
                                        "recipe": None if arm else recipe}
    return gold


def router_correct(output, gold):
    """True/False against the spec's gold; None if the spec carries no usable labels."""
    from gads.core import taxonomy as tx
    if not gold or not gold["tasks"] or not gold["modalities"]:
        return None
    task_ok = any(tx.tasks_overlap(tx.canonical_task(output.get("task_type")), tx.canonical_task(t))
                  for t in gold["tasks"])
    mods = {tx.canonical_modality(m) for m in gold["modalities"]}
    mod_ok = tx.canonical_modality(output.get("data_modality")) in mods
    recipe_ok = (output.get("matched_recipe_id") == gold["recipe"]) if gold["recipe"] else True
    return task_ok and mod_ok and recipe_ok


def router_examples(stats, cap_per_spec):
    """Correct Router calls from the capture stream, labelled against spec gold."""
    calls = [c for c in read_jsonl(f"{CAPTURE}/calls-*.jsonl")
             if c.get("stage_name") == "Router" and c.get("output")]
    stats["router_calls"] = len(calls)
    if not calls:
        return []
    gold = spec_gold()
    out, per_spec = [], collections.Counter()
    for c in sorted(calls, key=lambda c: c.get("ts", 0), reverse=True):
        verdict = router_correct(c["output"], gold.get(c.get("spec") or ""))
        if verdict is None:
            stats["router_unlabelled"] += 1
            continue
        if not verdict:
            stats["router_wrong"] += 1
            continue
        if per_spec[(c["spec"], c.get("model"))] >= cap_per_spec:
            stats["router_capped"] += 1
            continue
        per_spec[(c["spec"], c.get("model"))] += 1
        out.append({"task_id": c.get("task_id") or f"router:{c['spec']}:{c.get('ts')}",
                    "spec": c["spec"], "project_id": c.get("project_id"),
                    "cls": "student_success" if is_local(c.get("model")) else "teacher_demo",
                    "model_used": c.get("model"), "source": c.get("source"),
                    "messages": c["messages"],
                    "target": json.dumps(c["output"], ensure_ascii=False)})
    return out


def _task_line(user_core):
    first = (user_core or "").split("\n", 1)[0]
    return first if first.startswith("TASK:") else None


def _as_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.startswith("["):
        try:
            import ast
            return list(ast.literal_eval(v))
        except Exception:
            return []
    return []


def split_of(ex, holdout_specs, pct):
    group = ex.get("spec") or f"project:{ex.get('project_id') or ex.get('task_id')}"
    if group in holdout_specs:
        return "holdout"
    bucket = int(hashlib.sha256(group.encode()).hexdigest(), 16) % 100
    return "holdout" if bucket < pct else "train"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", action="store_true", help="report only, write nothing")
    ap.add_argument("--no-cloud", action="store_true",
                    help="exclude every cloud-model output (terms-of-service-clean build)")
    ap.add_argument("--no-backfill", action="store_true", help="capture streams only")
    ap.add_argument("--no-followup", action="store_true", help="exclude follow-up-lane tasks")
    ap.add_argument("--allow-sanitized", action="store_true",
                    help="keep examples whose executed code the sanitizer rewrote")
    ap.add_argument("--current-prompts-only", action="store_true",
                    help="drop examples captured under an older prompt regime")
    ap.add_argument("--holdout-pct", type=int, default=15,
                    help="hash bucket for non-pinned specs (stable across builds)")
    ap.add_argument("--strict-dataset-holdout", action="store_true",
                    help="also hold out every spec that shares a DATASET with a benchmark")
    ap.add_argument("--cap-per-node", type=int, default=40)
    ap.add_argument("--router-cap-per-spec", type=int, default=5,
                    help="max Router examples per (spec, model), so one spec cannot dominate")
    ap.add_argument("--min-train", type=int, default=50,
                    help="refuse to write a snapshot smaller than this")
    args = ap.parse_args()

    stats = collections.Counter()
    cap_ex, cap_pairs = from_capture(stats)
    bf_ex, bf_pairs = ([], []) if args.no_backfill else from_backfill(stats)
    seen = {e["task_id"] for e in cap_ex}
    examples = cap_ex + [e for e in bf_ex if e["task_id"] not in seen]
    seen_pairs = {(p["task_id"], p["rejected"][:200]) for p in cap_pairs}
    pairs = cap_pairs + [p for p in bf_pairs if (p["task_id"], p["rejected"][:200]) not in seen_pairs]

    current_pv = None
    if args.current_prompts_only:
        from gads.core.server import _compute_prompt_version
        current_pv = _compute_prompt_version()

    def keep(ex):
        if args.no_cloud and ex["cls"] in ("teacher_demo", "teacher_onpol"):
            stats["dropped_cloud"] += 1
            return False
        if args.no_followup and ex.get("mode") == "followup":
            stats["dropped_followup"] += 1
            return False
        if ex.get("sanitizer") == "rewritten" and not args.allow_sanitized:
            stats["dropped_sanitizer_rewrite"] += 1
            return False
        if current_pv and ex.get("prompt_version") != current_pv:
            stats["dropped_stale_prompt"] += 1
            return False
        if not (ex.get("code") or "").strip():
            stats["dropped_empty"] += 1
            return False
        return True

    examples = [e for e in examples if keep(e)]
    pairs = [p for p in pairs if keep(p) and p.get("rejected")]

    holdout_specs = PINNED_HOLDOUT | benchmark_specs(args.strict_dataset_holdout)
    for e in examples + pairs:
        e["split"] = split_of(e, holdout_specs, args.holdout_pct)

    # Per (recipe, node) cap on the TRAIN side only; the holdout is never thinned.
    counts, train = collections.Counter(), []
    for e in sorted((e for e in examples if e["split"] == "train"),
                    key=lambda e: (e.get("prompt_version") or "", e["task_id"]), reverse=True):
        key = (e.get("recipe_id"), e.get("recipe_node_id") or (e.get("user_core") or "")[:80])
        if counts[key] >= args.cap_per_node:
            stats["dropped_cap"] += 1
            continue
        counts[key] += 1
        train.append(e)
    holdout = [e for e in examples if e["split"] == "holdout"]
    dpo = [p for p in pairs if p["split"] == "train"]

    by_cls = collections.Counter(e["cls"] for e in train)
    report = {
        "train_examples": len(train), "holdout_examples": len(holdout), "dpo_pairs": len(dpo),
        "train_specs": len({e.get("spec") for e in train}),
        "holdout_specs": sorted({e.get("spec") for e in holdout if e.get("spec")}),
        "train_by_class": dict(by_cls),
        "train_by_source": dict(collections.Counter(e["source"] for e in train)),
        "train_by_model": dict(collections.Counter(e.get("model_used") for e in train).most_common(12)),
        "dpo_by_class": dict(collections.Counter(p["cls"] for p in dpo)),
        "sanitizer": dict(collections.Counter(e.get("sanitizer") for e in train)),
        "prompt_versions": len({e.get("prompt_version") for e in train}),
        **{k: v for k, v in stats.items()},
    }
    print(json.dumps(report, indent=2))
    gate = "MET" if len(train) >= 1000 and report["train_specs"] >= 20 else "NOT met"
    print(f"\n009/035 gate (>=1k train examples over >=20 specs): {gate} "
          f"({len(train)} examples, {report['train_specs']} specs)")
    if args.stats:
        return
    if len(train) < args.min_train:
        sys.exit(f"refusing to write: {len(train)} train examples < --min-train {args.min_train}")

    from gads.agents.workers.coder import render_messages

    def row(e):
        msgs = render_messages(e["system"], e["user_core"], raw=True)
        return {"task_id": e["task_id"], "spec": e.get("spec"),
                "required_variables": e.get("required_variables") or [],
                "messages": msgs + [{"role": "assistant", "content": fenced(e["code"])}]}

    snap = pathlib.Path(OUT_ROOT) / time.strftime("%Y%m%d-%H%M%S")
    snap.mkdir(parents=True)
    with open(snap / "sft_train.jsonl", "w") as f:
        for e in train:
            f.write(json.dumps(row(e)) + "\n")
    with open(snap / "sft_holdout.jsonl", "w") as f:
        for e in holdout:
            f.write(json.dumps(row(e)) + "\n")
    with open(snap / "dpo_train.jsonl", "w") as f:
        for p in dpo:
            f.write(json.dumps({"task_id": p["task_id"], "spec": p.get("spec"), "cls": p["cls"],
                                "messages": render_messages(p["system"], p["user_core"], raw=True),
                                "chosen": fenced(p["code"]), "rejected": fenced(p["rejected"])})
                    + "\n")
    # Skills-ablated renderings (035 §7): the same examples with the skills block replaced by
    # the placeholder the Coder uses when no skill applies. Capture examples only: the
    # harvest backfill never recorded which text was the skills block.
    from gads.agents.workers.coder import NO_SKILLS_TEXT

    def noskills(e):
        sk = (e.get("skills_context") or "").strip()
        if e.get("source") != "capture":
            return None
        if sk and sk not in e["system"]:
            return None
        return dict(e, system=e["system"].replace(sk, NO_SKILLS_TEXT, 1) if sk else e["system"])

    n_ns = collections.Counter()
    for name, rows in (("sft_train_noskills.jsonl", train), ("sft_holdout_noskills.jsonl", holdout)):
        with open(snap / name, "w") as f:
            for e in rows:
                ns = noskills(e)
                if ns is not None:
                    f.write(json.dumps(row(ns)) + "\n")
                    n_ns[name] += 1

    router = router_examples(stats, args.router_cap_per_spec)
    n_router = collections.Counter()
    for e in router:
        e["split"] = split_of(e, holdout_specs, args.holdout_pct)
    for split in ("train", "holdout"):
        with open(snap / f"router_{split}.jsonl", "w") as f:
            for e in router:
                if e["split"] == split:
                    f.write(json.dumps({"task_id": e["task_id"], "spec": e["spec"], "cls": e["cls"],
                                        "messages": e["messages"] + [
                                            {"role": "assistant", "content": e["target"]}]}) + "\n")
                    n_router[split] += 1
    report["noskills"] = dict(n_ns)
    report["router"] = {**dict(n_router), "calls": stats["router_calls"],
                        "wrong": stats["router_wrong"], "unlabelled": stats["router_unlabelled"]}
    print(f"skills-ablated rows: {dict(n_ns)} | router examples: {dict(n_router)} "
          f"(from {stats['router_calls']} captured Router calls)")

    with open(snap / "manifest.jsonl", "w") as f:
        for e in train + holdout:
            f.write(json.dumps({k: e.get(k) for k in (
                "task_id", "spec", "split", "cls", "source", "model_used", "engine_id",
                "prompt_version", "recipe_id", "recipe_node_id", "mode", "attempt_index",
                "n_attempts", "sanitizer")}) + "\n")
    json.dump({**report, "args": vars(args), "built": snap.name}, open(snap / "stats.json", "w"),
              indent=2)
    latest = pathlib.Path(OUT_ROOT) / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(snap.name)
    print(f"\nsnapshot -> {snap}  (latest -> {snap.name})")


if __name__ == "__main__":
    main()
