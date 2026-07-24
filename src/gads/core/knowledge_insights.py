"""Read-only insight endpoints for the Knowledge Studio (approach_docs/017 §2).

Everything here is derived, never mutating: the dependency graph, the task_type ×
dial-rung coverage matrix, per-file git history/diff, cross-references (impact), and
per-engine benchmark evidence from the dial ledger. Kept out of ``server.py`` so the
orchestrator file stays about orchestration and the SPA-facing surface is one module.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional

from gads.core.dial import NATIVE_STEP_FUNCTIONS, RUNG_ORDER, compiled_plan_dial, node_rung

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_LEDGER = os.getenv("GADS_DIAL_LEDGER", os.path.join(_REPO_ROOT, "research", "dial_ledger.jsonl"))
_SPECS_DIR = os.path.join(_REPO_ROOT, "specs")


# --------------------------------------------------------------------------- #
# Native function index — maps a native function name -> module filename, so a
# recipe node whose intent mandates a native step can be edged to its module.
# --------------------------------------------------------------------------- #
_DEF_RE = re.compile(r"^\s*def\s+([a-zA-Z_][\w]*)\s*\(", re.MULTILINE)


def _native_function_index(registry) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for fn in registry.list_native_files():
        try:
            src = registry.get_raw_native(fn)
        except Exception:
            continue
        for m in _DEF_RE.finditer(src):
            index[m.group(1)] = fn
    return index


def _node_dicts(recipe) -> List[Dict[str, Any]]:
    """DAG nodes as plain dicts (the shape dial.node_rung expects)."""
    return [
        {"id": t.id, "intent": t.intent, "attached_skills": t.attached_skills}
        for t in recipe.dag
    ]


# --------------------------------------------------------------------------- #
# Dependency graph
# --------------------------------------------------------------------------- #
def build_graph(registry) -> Dict[str, Any]:
    """recipe -> skill (attached) and recipe -> native (mechanized step) edges, plus
    every item as a node with its provenance. The spine the SPA renders as a map."""
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    native_idx = _native_function_index(registry)

    for rid, rec in registry.recipes.items():
        nodes.append({"id": rid, "type": "recipe",
                      "provenance": registry.provenance("recipes", rid)})
    for sid, sk in registry.skills.items():
        nodes.append({"id": sid, "type": "skill", "triggers": sk.triggers,
                      "provenance": registry.provenance("skills", sid)})
    native_files = registry.list_native_files()
    for fn in native_files:
        nodes.append({"id": fn, "type": "native"})

    known_skills = set(registry.skills.keys())
    for rid, rec in registry.recipes.items():
        attached_seen = set()
        native_seen = set()
        for t in rec.dag:
            for sk in t.attached_skills:
                if sk == "sandbox_environment" or (rid, sk) in attached_seen:
                    continue
                attached_seen.add((rid, sk))
                edges.append({"source": rid, "target": sk, "type": "recipe->skill",
                              "kind": "attaches", "node": t.id,
                              "dangling": sk not in known_skills})
            # mechanized step -> native module
            intent = t.intent or ""
            for func in NATIVE_STEP_FUNCTIONS:
                if func in intent and func in native_idx:
                    mod = native_idx[func]
                    if (rid, mod) not in native_seen:
                        native_seen.add((rid, mod))
                        edges.append({"source": rid, "target": mod, "type": "recipe->native",
                                      "kind": "mechanizes", "node": t.id, "function": func})

    return {"nodes": nodes, "edges": edges,
            "counts": {"recipes": len(registry.recipes), "skills": len(registry.skills),
                       "native": len(native_files), "edges": len(edges)}}


# --------------------------------------------------------------------------- #
# Coverage matrix
# --------------------------------------------------------------------------- #
def build_coverage(registry) -> Dict[str, Any]:
    """task_type × project-rung matrix over the recipe library, plus orphan analysis
    (skills reachable only by keyword, native modules no recipe mechanizes)."""
    matrix: Dict[str, Dict[str, List[str]]] = {}
    recipe_rungs: Dict[str, str] = {}
    untyped: List[str] = []

    for rid, rec in registry.recipes.items():
        rung = compiled_plan_dial(_node_dicts(rec), "routed")["rung"]
        recipe_rungs[rid] = rung
        task_types = (rec.applies_when or {}).get("task_type") or []
        if not task_types:
            untyped.append(rid)
        for tt in task_types:
            matrix.setdefault(tt, {}).setdefault(rung, []).append(rid)

    # skill reachability: attached to a recipe vs keyword-only
    attached_skills = set()
    for rec in registry.recipes.values():
        for t in rec.dag:
            attached_skills.update(s for s in t.attached_skills if s != "sandbox_environment")
    keyword_only = sorted(sid for sid in registry.skills if sid not in attached_skills)

    # native reachability: mechanized by some recipe intent vs unreferenced
    native_idx = _native_function_index(registry)
    referenced_modules = set()
    for rec in registry.recipes.values():
        for t in rec.dag:
            for func in NATIVE_STEP_FUNCTIONS:
                if func in (t.intent or "") and func in native_idx:
                    referenced_modules.add(native_idx[func])
    unreferenced_native = sorted(f for f in registry.list_native_files() if f not in referenced_modules)

    return {
        "matrix": matrix,
        "rungs": RUNG_ORDER,
        "recipe_rungs": recipe_rungs,
        "task_types": sorted(matrix.keys()),
        "orphans": {
            "recipes_without_task_type": sorted(untyped),
            "skills_keyword_only": keyword_only,
            "native_unreferenced": unreferenced_native,
        },
    }


# --------------------------------------------------------------------------- #
# Git history / diff (shipped files only — overlay is git-ignored)
# --------------------------------------------------------------------------- #
def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", _REPO_ROOT, *args],
                          capture_output=True, text=True, timeout=15)


def item_history(registry, item_type: str, item_id: str, limit: int = 50) -> Dict[str, Any]:
    filename = registry.resolve_filename(item_type, item_id)
    if not filename:
        raise ValueError(f"Unknown {item_type} '{item_id}'.")
    rel = os.path.relpath(registry.shipped_path(item_type, filename), _REPO_ROOT)
    fmt = "%H%x1f%h%x1f%an%x1f%aI%x1f%s"
    proc = _git("log", f"-n{limit}", f"--format={fmt}", "--", rel)
    commits: List[Dict[str, str]] = []
    if proc.returncode == 0:
        for line in proc.stdout.splitlines():
            parts = line.split("\x1f")
            if len(parts) == 5:
                commits.append({"sha": parts[0], "short": parts[1], "author": parts[2],
                                "date": parts[3], "subject": parts[4]})
    overlay = registry.provenance(registry.norm_type(item_type), item_id) in ("overlay", "overridden")
    return {"item_type": registry.norm_type(item_type), "id": item_id, "file": rel,
            "tracked": bool(commits), "has_overlay_edits": overlay, "commits": commits}


def item_diff(registry, item_type: str, item_id: str,
              from_ref: Optional[str] = None, to_ref: Optional[str] = None) -> Dict[str, Any]:
    filename = registry.resolve_filename(item_type, item_id)
    if not filename:
        raise ValueError(f"Unknown {item_type} '{item_id}'.")
    rel = os.path.relpath(registry.shipped_path(item_type, filename), _REPO_ROOT)
    if from_ref and to_ref:
        proc = _git("diff", f"{from_ref}..{to_ref}", "--", rel)
    elif from_ref:
        # single commit: what that commit changed in this file
        proc = _git("show", from_ref, "--", rel)
    else:
        # uncommitted working-tree changes (e.g. shipped file edited in place)
        proc = _git("diff", "HEAD", "--", rel)
    diff = proc.stdout if proc.returncode == 0 else (proc.stderr or "")
    return {"item_type": registry.norm_type(item_type), "id": item_id, "file": rel,
            "from": from_ref, "to": to_ref, "diff": diff}


# --------------------------------------------------------------------------- #
# Impact — who references this id (rename/deprecate guard)
# --------------------------------------------------------------------------- #
def item_impact(registry, item_type: str, item_id: str) -> Dict[str, Any]:
    t = registry.norm_type(item_type)
    referenced_by: Dict[str, List[Any]] = {"specs": [], "recipes": [], "ledger_runs": 0}

    if t == "recipes":
        # specs that pin this recipe (frontmatter recipe_id / recipes list)
        referenced_by["specs"] = _specs_referencing_recipe(item_id)
        referenced_by["ledger_runs"] = _ledger_run_count(item_id)
    elif t == "skills":
        # recipes whose DAG attaches this skill
        for rid, rec in registry.recipes.items():
            nodes = [t2.id for t2 in rec.dag if item_id in t2.attached_skills]
            if nodes:
                referenced_by["recipes"].append({"recipe": rid, "nodes": nodes})
    elif t == "native":
        # recipes whose intent mechanizes a function defined in this module
        native_idx = _native_function_index(registry)
        module_funcs = {f for f, m in native_idx.items() if m == item_id}
        for rid, rec in registry.recipes.items():
            hits = []
            for t2 in rec.dag:
                for func in module_funcs & set(NATIVE_STEP_FUNCTIONS):
                    if func in (t2.intent or ""):
                        hits.append({"node": t2.id, "function": func})
            if hits:
                referenced_by["recipes"].append({"recipe": rid, "uses": hits})

    referenced_by["total"] = (len(referenced_by["specs"]) + len(referenced_by["recipes"]))
    return {"item_type": t, "id": item_id, "referenced_by": referenced_by}


_SPEC_FM = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _specs_referencing_recipe(recipe_id: str) -> List[str]:
    """Specs that pin this recipe. Matches the frontmatter `recipe_id` / `recipes`
    fields by EXACT value — a substring test would count `foo.dowhy` as referenced by
    a spec pinning `foo.dowhy.mechanized`."""
    import yaml
    hits: List[str] = []
    if not os.path.isdir(_SPECS_DIR):
        return hits
    for fn in sorted(os.listdir(_SPECS_DIR)):
        if not fn.endswith(".md"):
            continue
        try:
            with open(os.path.join(_SPECS_DIR, fn)) as f:
                m = _SPEC_FM.match(f.read())
            if not m:
                continue
            data = yaml.safe_load(m.group(1)) or {}
        except Exception:
            continue
        pinned = data.get("recipe_id")
        listed = data.get("recipes") or []
        if not isinstance(listed, list):
            listed = [listed]
        if pinned == recipe_id or recipe_id in listed:
            hits.append(fn)
    return hits


# --------------------------------------------------------------------------- #
# Evidence — per-engine benchmark pass rate from the dial ledger
# --------------------------------------------------------------------------- #
def _iter_ledger() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(_LEDGER):
        return rows
    with open(_LEDGER) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _engine_label(rec: Dict[str, Any]) -> str:
    mode = rec.get("routing_mode") or "unknown"
    if mode == "cloud_pinned" and rec.get("pinned_model"):
        return f"cloud_pinned:{rec['pinned_model']}"
    return mode


def _ledger_run_count(recipe_id: str) -> int:
    return sum(1 for r in _iter_ledger() if r.get("recipe_id") == recipe_id)


def item_evidence(registry, item_id: str) -> Dict[str, Any]:
    """Per-engine pass/fail from the dial ledger for a recipe id. 'Engine' is the
    routing mode (the ledger's honest granularity — cloud runs span a tier ladder,
    so a single executing model isn't recorded), refined to the pinned model when
    a run was cloud_pinned."""
    runs = [r for r in _iter_ledger() if r.get("recipe_id") == item_id]
    by_engine: Dict[str, Dict[str, Any]] = {}
    rungs_seen = set()
    for r in runs:
        eng = _engine_label(r)
        b = by_engine.setdefault(eng, {"runs": 0, "pass": 0, "fail": 0})
        b["runs"] += 1
        passed = r.get("outcome") == "pass" or r.get("workflow_succeeded") is True
        b["pass" if passed else "fail"] += 1
        if r.get("rung"):
            rungs_seen.add(r["rung"])
    for b in by_engine.values():
        b["pass_rate"] = round(b["pass"] / b["runs"], 3) if b["runs"] else None

    total = len(runs)
    passes = sum(1 for r in runs
                 if r.get("outcome") == "pass" or r.get("workflow_succeeded") is True)
    recent = [
        {"ts": r.get("ts"), "engine": _engine_label(r), "rung": r.get("rung"),
         "outcome": r.get("outcome"), "spec": r.get("spec"),
         "project_id": (r.get("project_id") or "")[:8]}
        for r in runs[-10:][::-1]
    ]
    return {
        "id": item_id,
        "total_runs": total,
        "overall_pass_rate": round(passes / total, 3) if total else None,
        "rungs_observed": sorted(rungs_seen, key=lambda x: RUNG_ORDER.index(x) if x in RUNG_ORDER else 99),
        "by_engine": by_engine,
        "recent": recent,
    }
