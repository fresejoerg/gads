"""Recipe-driven composition of the final dashboard.

The dashboard used to be assembled from whatever artifacts happened to land in the
workspace: a flat card list, ordered by artifact creation. Nodes that produced no plot —
`characterize_task`, `shortlist_candidates`, `selection_audit`, every gate and every
reasoning step — left no trace at all, even though they are the steps a recipe exists for.

Here the **recipe DAG is the dashboard skeleton**: one section per node, in DAG order,
whether or not that node drew a picture, and *including nodes that never ran* (rendered as
such, so a partial run reads as a partial run rather than a short report).

Everything in this module is reconstructed from the database plus the workspace, never from
in-memory workflow state, so `reporting.rebuild_dashboard` produces the identical sections
after the fact.

Section evidence, in descending order of trustworthiness:
  1. metrics captured by the orchestrator's kernel probe (ground truth scalars)
  2. `gads_emit_insight()` payloads emitted by the node's own code
  3. files the node created (attributed by task id, not by regex over prose)
  4. the tail of the node's stdout
  5. the Synthesizer's prose for that section — the only LLM-authored part, and optional
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

# Tasks whose `assigned_to` is one of these are orchestration stages, not recipe nodes.
ORCHESTRATION_AGENTS = {
    "Planner", "Router", "SpecDrafter", "PlanCritique", "CompletenessVerifier",
    "Synthesizer", "Critique", "System", "DataAnalyzer", "DataSampler",
}

# Blocks the RecipeCompiler appends to a node's intent before handing it to the Coder.
# They are prompt scaffolding, not a description of the work, so they are stripped for
# display. Kept as a deterministic strip (the markers are emitted by us) rather than a
# guess at where the intent ends.
_APPENDED_BLOCKS = re.compile(r"\n\n\[(?:SPEC HINTS|RECIPE INVARIANTS)\b.*?\]\s*$",
                              re.DOTALL)
_METRICS_IN_SUMMARY = re.compile(r"\|\s*Metrics captured:\s*(.+)$")
_FILES_IN_SUMMARY = re.compile(r"[\w\d\-]+\.(?:json|html|png|csv|parquet|md)", re.IGNORECASE)

STDOUT_TAIL_CHARS = 2500


def clean_intent(description: str) -> str:
    """Strip the compiler's appended prompt blocks from a task description."""
    text = description or ""
    prev = None
    while prev != text:                       # both blocks can be appended
        prev = text
        text = _APPENDED_BLOCKS.sub("", text).strip()
    return text


def prettify_node_id(node_id: str) -> str:
    return (node_id or "").replace("_", " ").replace("-", " ").strip().title()


def _first_sentence(text: str, limit: int = 220) -> str:
    """A one-line summary for a node that declares none. Recipe intents are paragraphs and
    sometimes open with a code block, so this is a display convenience, not a parser."""
    text = " ".join((text or "").split())
    if not text:
        return ""
    m = re.search(r"(?<=[.!?])\s", text)
    if m and m.start() < limit:
        return text[:m.start() + 1].strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].strip() + " …"


def format_metric(value: Any) -> str:
    """Metrics reach the report as floats even when they are counts (the kernel probe
    JSON-round-trips them), and `16000.0000` rows reads as a defect. Whole floats lose the
    decimals; everything else keeps 4 significant decimals."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() and abs(value) < 1e15 else f"{value:.4f}"
    if isinstance(value, str):
        try:
            return format_metric(float(value))
        except ValueError:
            return value
    return str(value)


def extract_skeleton(tasks: List[Any]) -> List[Dict[str, Any]]:
    """Recover the recipe section skeleton persisted by the RecipeCompiler.

    Stored on the deterministic Planner task rather than re-read from the recipe file so a
    rebuild reproduces the sections of the run *as it was compiled*, even if the recipe has
    since been edited. Returns [] for drafted (non-recipe) plans.
    """
    best: List[Dict[str, Any]] = []
    best_at = None
    for t in tasks:
        if t.assigned_to != "Planner":
            continue
        sections = (t.result_json or {}).get("recipe_sections")
        if not sections:
            continue
        if best_at is None or (t.created_at and t.created_at >= best_at):
            best, best_at = sections, t.created_at
    return best


def _node_id_of(task: Any) -> Optional[str]:
    return (task.postcondition_json or {}).get("recipe_node_id")


def _latest_task_per_node(tasks: List[Any]) -> Dict[str, Any]:
    """One task per recipe node — the attempt whose result the dashboard should show.

    Replans create a fresh Task row per node, so a node can have several. A completed
    attempt always beats a failed one (resume-from-failed-node means the earlier success is
    the state the run actually carried forward); among equals the most recent wins.
    """
    rank = {"completed": 3, "bypassed": 2, "failed": 1, "pending": 0}
    chosen: Dict[str, Any] = {}
    for t in tasks:
        nid = _node_id_of(t)
        if not nid:
            continue
        cur = chosen.get(nid)
        if cur is None:
            chosen[nid] = t
            continue
        better = rank.get(t.status, 0) > rank.get(cur.status, 0)
        same_and_newer = (rank.get(t.status, 0) == rank.get(cur.status, 0)
                          and (t.created_at or datetime.min) >= (cur.created_at or datetime.min))
        if better or same_and_newer:
            chosen[nid] = t
    return chosen


def _dedupe_drafted(tasks: List[Any]) -> List[Any]:
    """One task per *step* of a drafted plan.

    A drafted plan has no node ids, so its steps are the tasks themselves — but a replan
    materialises a fresh row per step and the old rows stay in the table. Keyed on the
    description (which a replan reproduces near-verbatim for the steps it did not change)
    with the same completed-beats-failed preference as the recipe lane, so a re-planned run
    shows one section per step rather than one per attempt.
    """
    rank = {"completed": 3, "bypassed": 2, "failed": 1, "pending": 0}
    order: List[str] = []
    chosen: Dict[str, Any] = {}
    for t in sorted(tasks, key=lambda t: t.created_at or datetime.min):
        key = " ".join((t.description or "").split())[:160].lower()
        cur = chosen.get(key)
        if cur is None:
            order.append(key)
            chosen[key] = t
        elif rank.get(t.status, 0) >= rank.get(cur.status, 0):
            chosen[key] = t
    return [chosen[k] for k in order]


def _metrics_from_task(task: Any) -> Dict[str, str]:
    """Metrics this node produced. Prefers the structured record; falls back to parsing the
    orchestrator summary so runs predating that record still show their numbers."""
    res = task.result_json or {}
    captured = res.get("metrics_captured")
    if isinstance(captured, dict) and captured:
        return {k: format_metric(v) for k, v in captured.items()}
    m = _METRICS_IN_SUMMARY.search(res.get("orchestrator_summary", "") or "")
    if not m:
        return {}
    out: Dict[str, str] = {}
    for part in m.group(1).split(","):
        if "=" in part:
            k, _, v = part.partition("=")
            out[k.strip()] = format_metric(v.strip())
    return out


def _files_from_task(task: Any) -> List[str]:
    """Files this node created. The structured record is authoritative; projects that ran
    before it existed fall back to the filenames named in the orchestrator summary, which
    the executor writes from its own before/after workspace scan. That keeps the section
    view working retroactively across the existing run corpus."""
    res = task.result_json or {}
    files = res.get("artifact_files")
    if isinstance(files, list) and files:
        return [str(f) for f in files]
    summary = res.get("orchestrator_summary", "") or ""
    seen, out = set(), []
    for fn in _FILES_IN_SUMMARY.findall(summary):
        if fn.lower() not in seen:
            seen.add(fn.lower())
            out.append(fn)
    return out


def _split_insights(raw: List[Any]) -> tuple:
    """Separate a node's interpretive insights from its structural telemetry.

    The executor appends the 'verification floor' probe — a scan of every DataFrame alive in
    the kernel — to the same list the model's own `gads_emit_insight()` calls land in. That
    is right for contract validation and wrong for a report: the probe is cumulative, so
    every node re-reports every upstream frame and the one sentence the node actually
    reasoned its way to is buried. Flagged entries (`is_floor`) become state telemetry
    instead, shown compactly and only where a variable first appears.
    """
    insights, state = [], []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        (state if item.get("is_floor") else insights).append(item)
    return insights, state


def _fallback_kind(model_used: str) -> Optional[str]:
    if str(model_used).startswith("native_fallback:"):
        return "native"
    if str(model_used).startswith("cloud_fallback:"):
        return "cloud"
    return None


def build_sections(tasks: List[Any], skeleton: Optional[List[Dict[str, Any]]] = None
                   ) -> List[Dict[str, Any]]:
    """Build the ordered dashboard sections for a project.

    `skeleton` is the compiled recipe DAG (see `extract_skeleton`). When it is absent the
    plan was LLM-drafted and has no node ids, so the executed tasks themselves become the
    sections, in plan order — the same mechanism, degraded to what a drafted plan can offer.
    """
    skeleton = skeleton or extract_skeleton(tasks)
    by_node = _latest_task_per_node(tasks)
    sections: List[Dict[str, Any]] = []
    seen_state: set = set()   # DataFrames already reported, so each is shown once

    if skeleton:
        pairs = [(s, by_node.get(s.get("id"))) for s in skeleton]
    else:
        exec_tasks = _dedupe_drafted(
            [t for t in tasks if t.assigned_to not in ORCHESTRATION_AGENTS])
        pairs = [({"id": f"step_{i + 1}",
                   "title": f"Step {i + 1}",
                   "intent": clean_intent(t.description)}, t)
                 for i, t in enumerate(exec_tasks)]

    for idx, (node, task) in enumerate(pairs, start=1):
        node_id = node.get("id") or f"step_{idx}"
        report_meta = node.get("report") or {}
        intent = node.get("intent") or (clean_intent(task.description) if task else "")

        section: Dict[str, Any] = {
            "node_id": node_id,
            "index": idx,
            "title": report_meta.get("title") or node.get("title") or prettify_node_id(node_id),
            "summary": report_meta.get("summary") or _first_sentence(intent),
            "intent": intent,
            "produces": node.get("produces") or [],
            "status": "not_executed",
            "model_used": "",
            "fallback": None,
            "metrics": {},
            "insights": [],
            "state": [],
            "files": [],
            "stdout": "",
            "error": None,
            "cards": [],
            "note": "",
            "task_id": None,
            "forced_collapsed": bool(report_meta.get("collapsed")),
        }

        if task is not None:
            res = task.result_json or {}
            model_used = res.get("model_used", "") or task.assigned_to
            node_insights, node_state = _split_insights(res.get("semantic_insights"))
            # Only frames this node introduced — an unchanged upstream frame is not news.
            fresh_state = [st for st in node_state
                           if st.get("artifact") and st["artifact"] not in seen_state]
            seen_state.update(st["artifact"] for st in fresh_state)
            section.update({
                "state": fresh_state,
                "status": task.status,
                "model_used": model_used,
                "fallback": _fallback_kind(model_used),
                "metrics": _metrics_from_task(task),
                "insights": node_insights,
                "files": _files_from_task(task),
                "stdout": (res.get("stdout") or "").strip()[-STDOUT_TAIL_CHARS:],
                "error": task.error,
                "task_id": str(task.id),
            })

        sections.append(section)

    return sections


def attach_cards(sections: List[Dict[str, Any]], cards: List[Dict[str, Any]]
                 ) -> List[Dict[str, Any]]:
    """Distribute dashboard cards over the sections; return the ones that fit nowhere.

    Attribution is by the `task_id` stamped on the artifact at creation. Older projects have
    no such stamp, so filenames recorded against the node fall back to a name match. An
    artifact that matches nothing is never dropped — the caller renders the remainder in a
    trailing section, so the dashboard can lose structure but not evidence.
    """
    by_task = {s["task_id"]: s for s in sections if s.get("task_id")}
    by_file: Dict[str, Dict[str, Any]] = {}
    for s in sections:
        for f in s.get("files") or []:
            by_file.setdefault(str(f).lower(), s)

    orphans: List[Dict[str, Any]] = []
    for card in cards:
        target = by_task.get(card.get("task_id"))
        if target is None:
            key = (card.get("source_filename") or "").lower()
            target = by_file.get(key) if key else None
        if target is None:
            orphans.append(card)
        else:
            target["cards"].append(card)
    return orphans


def apply_section_notes(sections: List[Dict[str, Any]], notes: Optional[List[Any]]) -> None:
    """Merge the Synthesizer's per-section prose in. Matching is by node id, then by title,
    then positional — the prose is a nicety and a mismatch must never blank a section."""
    if not notes:
        return
    by_id = {s["node_id"].lower(): s for s in sections}
    by_title = {s["title"].lower(): s for s in sections}
    for i, note in enumerate(notes):
        data = note.model_dump() if hasattr(note, "model_dump") else dict(note or {})
        text = (data.get("text") or "").strip()
        if not text:
            continue
        key = str(data.get("node_id") or "").strip().lower()
        target = by_id.get(key) or by_title.get(key)
        if target is None and key:
            target = next((s for s in sections if key in s["node_id"].lower()), None)
        if target is None and i < len(sections) and not key:
            target = sections[i]
        if target is not None and not target["note"]:
            target["note"] = text


def finalize(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Decide each section's presentation weight.

    A node that produced no cards, metrics, insights or prose is plumbing (loading a split,
    binding a variable): it still gets a section — the run is not honestly described without
    it — but a compact, collapsed one, so the reasoning and result nodes carry the page.
    """
    for s in sections:
        has_substance = bool(s["cards"] or s["metrics"] or s["insights"] or s["note"])
        s["collapsed"] = s["forced_collapsed"] or not has_substance
        s["has_substance"] = has_substance
    return sections


def format_for_prompt(sections: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    """Render the section skeleton for the Synthesizer's context.

    Deliberately evidence-only: each step is listed with what it actually produced, so the
    model writes commentary on a structure it cannot change. Steps that never ran are listed
    too — the model is told to report them as gaps rather than quietly omit them.
    """
    if not sections:
        return ""
    lines = ["### PIPELINE SECTIONS (the report's structure — one note per entry, in order)"]
    for s in sections:
        status = "NOT EXECUTED" if s["status"] == "not_executed" else s["status"].upper()
        lines.append(f"\n[{s['index']}] node_id: {s['node_id']}  ({status})")
        lines.append(f"    step: {s['title']} — {s['summary'] or s['intent'][:200]}")
        if s["metrics"]:
            lines.append("    metrics: " + ", ".join(f"{k}={v}" for k, v in s["metrics"].items()))
        for ins in s["insights"][:4]:   # interpretive only — floor telemetry is excluded
            lines.append(f"    insight: {str(ins.get('insight', ''))[:220]}")
        if s["files"]:
            lines.append("    files: " + ", ".join(str(f) for f in s["files"][:8]))
        if s["error"]:
            lines.append(f"    error: {str(s['error'])[:200]}")
        if not (s["metrics"] or s["insights"] or s["files"] or s["error"]):
            lines.append("    evidence: none recorded")
    text = "\n".join(lines)
    return text[:max_chars]
