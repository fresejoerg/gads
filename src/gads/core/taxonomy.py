"""Data-science project taxonomy — vocabulary loader, spec-tag validator, and the
spec coverage index. Backs the `taxonomy:` frontmatter block (see
approach_docs/018_ds_project_taxonomy.md) and the Studio Taxonomy view.

Kept out of server.py; the vocabulary source of truth is
``src/gads/knowledge/taxonomy.yaml``.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # src/gads/core -> repo
_VOCAB_PATH = os.path.join(os.path.dirname(_HERE), "knowledge", "taxonomy.yaml")
_SPECS_DIR = os.path.join(_REPO_ROOT, "specs")

# Frontmatter fence — same shape used elsewhere in the codebase.
_FM = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# Dial-rung suffix on spec filenames (execution posture, not a taxonomy facet).
_RUNG_SUFFIX = re.compile(r"_(d[0-5][a-z]*)$", re.IGNORECASE)

# Which facets are single-valued vs list, and which are required.
_FACETS_ONE = ("intent", "domain")
_FACETS_MANY = ("task", "modality", "deliverable", "validation")
_REQUIRED = ("intent", "task", "modality", "domain", "deliverable")


@lru_cache(maxsize=1)
def load_vocab() -> Dict[str, Any]:
    """Parsed taxonomy.yaml (cached). Call ``load_vocab.cache_clear()`` after edits."""
    with open(_VOCAB_PATH) as f:
        return yaml.safe_load(f)


def _value_ok(facet: str, value: str, vocab: Dict[str, Any]) -> bool:
    """True if ``value`` is in the controlled vocab for ``facet``. For ``task``,
    a bare family or any ``family.subtype`` is accepted."""
    fv = vocab.get(facet)
    if not isinstance(fv, dict):
        return False
    if value in fv:
        return True
    if facet == "task" and "." in value:
        fam, sub = value.split(".", 1)
        return fam in fv and sub in (fv.get(fam) or [])
    return False


def validate_tags(block: Any) -> Dict[str, List[str]]:
    """Validate a spec's ``taxonomy:`` block against the vocabulary + schema.

    Returns ``{"errors": [...], "warnings": [...]}``. Errors = required facet
    missing, wrong cardinality, or unknown value. Warnings = soft advisories
    (e.g. no ``validation`` regime declared)."""
    errors: List[str] = []
    warnings: List[str] = []
    vocab = load_vocab()

    if not isinstance(block, dict):
        return {"errors": ["taxonomy: must be a mapping"], "warnings": []}

    # rung must never leak into the taxonomy block
    for k in ("rung", "dial", "delegation"):
        if k in block:
            errors.append(f"'{k}' is not a taxonomy facet (delegation rung is a separate axis)")

    for facet in _FACETS_ONE:
        val = block.get(facet)
        if val is None:
            if facet in _REQUIRED:
                errors.append(f"missing required facet '{facet}'")
            continue
        if isinstance(val, list):
            errors.append(f"'{facet}' must be a single value, got a list")
            continue
        if not _value_ok(facet, str(val), vocab):
            errors.append(f"unknown {facet}: {val!r}")

    for facet in _FACETS_MANY:
        val = block.get(facet)
        if val is None:
            if facet in _REQUIRED:
                errors.append(f"missing required facet '{facet}'")
            continue
        vals = val if isinstance(val, list) else [val]
        if facet in _REQUIRED and len(vals) < 1:
            errors.append(f"'{facet}' needs at least one value")
        for v in vals:
            if not _value_ok(facet, str(v), vocab):
                errors.append(f"unknown {facet} value: {v!r}")

    if not block.get("validation"):
        warnings.append("no 'validation' regime declared — recommended so completeness "
                        "can judge methodological appropriateness")
    return {"errors": errors, "warnings": warnings}


def canonicalize(task_type: str) -> Optional[Dict[str, str]]:
    """Map a legacy free-text ``task_type`` string (recipe/Router) to a canonical
    ``{"intent": ..., "task": ...}`` pair via the crosswalk. None if unmapped."""
    if not task_type:
        return None
    return (load_vocab().get("crosswalk") or {}).get(task_type.strip().lower())


def _as_list(v: Any) -> List[str]:
    if v is None:
        return []
    return list(v) if isinstance(v, list) else [v]


def _task_family(task_val: str) -> str:
    return task_val.split(".", 1)[0] if "." in task_val else task_val


def _repair(tax: Dict[str, Any], vocab: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """Drop any values not in the vocab (rather than fail); return cleaned block."""
    warns: List[str] = []
    out: Dict[str, Any] = {}
    for facet in _FACETS_ONE:
        val = tax.get(facet)
        if val is None:
            continue
        if _value_ok(facet, str(val), vocab):
            out[facet] = val
        else:
            warns.append(f"dropped unknown {facet}={val!r}")
    for facet in _FACETS_MANY:
        if facet not in tax:
            continue
        kept = [v for v in _as_list(tax[facet]) if _value_ok(facet, str(v), vocab)]
        dropped = [v for v in _as_list(tax[facet]) if not _value_ok(facet, str(v), vocab)]
        for d in dropped:
            warns.append(f"dropped unknown {facet} value {d!r}")
        if kept:
            out[facet] = kept
    if tax.get("domain_detail"):
        out["domain_detail"] = tax["domain_detail"]
    return out, warns


def derive_run_taxonomy(existing: Any = None, task_type: Optional[str] = None,
                        data_modality: Optional[str] = None,
                        domain_text: Optional[str] = None) -> Dict[str, Any]:
    """Resolve a *run's* taxonomy so every run — ad-hoc included — is classified.

    If ``existing`` is a valid block (spec-launched run) it is honored and repaired.
    Otherwise facets are derived deterministically: intent/task from the Router's
    ``task_type`` (via the crosswalk), modality from ``data_modality``, domain from a
    substring match on ``domain_text``, and deliverable/validation from per-family
    defaults. Returns ``{"taxonomy", "warnings", "source"}``. Never raises for content."""
    vocab = load_vocab()
    fam_defaults = vocab.get("family_defaults", {})
    mod_alias = vocab.get("modality_aliases", {})
    dom_alias = vocab.get("domain_aliases", {})
    warnings: List[str] = []

    tax: Dict[str, Any] = dict(existing) if isinstance(existing, dict) and existing else {}
    source = "spec" if tax else "derived"

    canon = canonicalize(task_type) if task_type else None
    if not tax.get("task") and canon:
        tax["task"] = [canon["task"]]
    if not tax.get("intent") and canon:
        tax["intent"] = canon["intent"]

    tasks = _as_list(tax.get("task"))
    if not tasks:
        tax["task"] = ["analytics.descriptive_stats"]
        tasks = tax["task"]
        if source == "derived":
            warnings.append(f"could not classify task from task_type={task_type!r}; "
                            "defaulted to descriptive analytics")
    fam = _task_family(tasks[0])
    fd = fam_defaults.get(fam, {})

    if not tax.get("intent"):
        tax["intent"] = fd.get("intent") or "descriptive"

    if not tax.get("modality"):
        m = mod_alias.get((data_modality or "").strip().lower())
        tax["modality"] = [m] if m else ["tabular"]

    if not tax.get("domain"):
        dt = (domain_text or "").lower()
        dom = next((v for k, v in dom_alias.items() if k in dt), None)
        tax["domain"] = dom or "general"
    if domain_text and not tax.get("domain_detail"):
        tax["domain_detail"] = domain_text

    if not tax.get("deliverable"):
        tax["deliverable"] = list(fd.get("deliverable") or ["report_narrative"])
    if "validation" not in tax:
        v = fd.get("validation")
        if v:
            tax["validation"] = list(v)

    tax, repaired = _repair(tax, vocab)
    warnings += repaired
    # guarantee required facets survive the repair
    if not tax.get("intent"):
        tax["intent"] = "descriptive"
    if not tax.get("task"):
        tax["task"] = ["analytics.descriptive_stats"]
    if not tax.get("modality"):
        tax["modality"] = ["tabular"]
    if not tax.get("domain"):
        tax["domain"] = "general"
    if not tax.get("deliverable"):
        tax["deliverable"] = ["report_narrative"]
    return {"taxonomy": tax, "warnings": warnings, "source": source}


def render_block(tax: Dict[str, Any]) -> str:
    """Render a resolved taxonomy dict as a frontmatter ``taxonomy:`` block."""
    def flow(seq: List[str]) -> str:
        return "[" + ", ".join(seq) + "]"
    lines = ["taxonomy:"]
    lines.append(f"  intent: {tax['intent']}")
    lines.append(f"  task: {flow(_as_list(tax['task']))}")
    lines.append(f"  modality: {flow(_as_list(tax['modality']))}")
    lines.append(f"  domain: {tax['domain']}")
    if tax.get("domain_detail"):
        lines.append(f'  domain_detail: "{tax["domain_detail"]}"')
    lines.append(f"  deliverable: {flow(_as_list(tax['deliverable']))}")
    if tax.get("validation"):
        lines.append(f"  validation: {flow(_as_list(tax['validation']))}")
    return "\n".join(lines) + "\n"


def inject_into_frontmatter(spec_md: str, tax: Dict[str, Any]) -> str:
    """Insert a ``taxonomy:`` block into a spec's frontmatter (before the closing
    fence). No-op if a taxonomy block is already present or there is no frontmatter."""
    m = _FM.match(spec_md)
    if not m or "taxonomy:" in m.group(1):
        return spec_md
    new_fm = "---\n" + m.group(1).rstrip("\n") + "\n" + render_block(tax) + "---\n"
    return new_fm + spec_md[m.end():]


def _parse_frontmatter(text: str) -> Optional[Dict[str, Any]]:
    m = _FM.match(text)
    if not m:
        return None
    try:
        data = yaml.safe_load(m.group(1))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def spec_index() -> List[Dict[str, Any]]:
    """Every spec's taxonomy tag (best-effort), for the coverage view. Includes
    untagged specs (``tagged: False``) so gaps are visible. ``test_*`` skipped."""
    out: List[Dict[str, Any]] = []
    if not os.path.isdir(_SPECS_DIR):
        return out
    for fn in sorted(os.listdir(_SPECS_DIR)):
        if not fn.endswith(".md") or fn.startswith("test_"):
            continue
        stem = fn[:-3]
        data = _parse_frontmatter(open(os.path.join(_SPECS_DIR, fn)).read()) or {}
        tax = data.get("taxonomy") or {}
        rung_m = _RUNG_SUFFIX.search(stem)
        rec = {
            "spec": stem,
            "name": data.get("name"),
            "recipe_id": data.get("recipe_id"),
            "rung": rung_m.group(1).upper() if rung_m else None,
            "tagged": bool(tax),
            "intent": tax.get("intent"),
            "task": tax.get("task") if isinstance(tax.get("task"), list) else ([tax["task"]] if tax.get("task") else []),
            "modality": tax.get("modality") if isinstance(tax.get("modality"), list) else ([tax["modality"]] if tax.get("modality") else []),
            "domain": tax.get("domain"),
            "deliverable": tax.get("deliverable") if isinstance(tax.get("deliverable"), list) else ([tax["deliverable"]] if tax.get("deliverable") else []),
            "validation": tax.get("validation") if isinstance(tax.get("validation"), list) else ([tax["validation"]] if tax.get("validation") else []),
        }
        out.append(rec)
    return out


def coverage() -> Dict[str, Any]:
    """Intent × Task-family matrix over the spec library + per-axis distributions
    and the empty high-value cells. Dial variants are folded to one 'project' per
    (intent, task-families, domain) group so the counts reflect distinct projects,
    not rung duplicates."""
    vocab = load_vocab()
    specs = [s for s in spec_index() if s["tagged"]]
    intents = list(vocab["intent"].keys())
    families = list(vocab["task"].keys())

    matrix: Dict[str, Dict[str, List[str]]] = {i: {f: [] for f in families} for i in intents}
    seen_projects: set = set()
    intent_dist: Dict[str, int] = {}
    modality_dist: Dict[str, int] = {}
    domain_dist: Dict[str, int] = {}

    for s in specs:
        fam_set = sorted({_task_family(t) for t in s["task"]})
        # distinct-project key ignores the dial rung
        pkey = (s["intent"], tuple(fam_set), s["domain"], tuple(sorted(s["modality"])))
        is_new = pkey not in seen_projects
        seen_projects.add(pkey)
        for fam in fam_set:
            if s["intent"] in matrix and fam in matrix[s["intent"]]:
                matrix[s["intent"]][fam].append(s["spec"])
        if is_new:
            intent_dist[s["intent"]] = intent_dist.get(s["intent"], 0) + 1
            for m in s["modality"]:
                modality_dist[m] = modality_dist.get(m, 0) + 1
            if s["domain"]:
                domain_dist[s["domain"]] = domain_dist.get(s["domain"], 0) + 1

    # which (intent, family) cells are populated
    populated = {(i, f) for i in intents for f in families if matrix[i][f]}

    return {
        "intents": intents,
        "task_families": families,
        "matrix": matrix,
        "distinct_projects": len(seen_projects),
        "total_specs": len(specs),
        "intent_distribution": intent_dist,
        "modality_distribution": modality_dist,
        "domain_distribution": domain_dist,
        "populated_cells": [list(c) for c in sorted(populated)],
    }


def recipe_index(registry) -> List[Dict[str, Any]]:
    """Derive each recipe's taxonomy from its ``applies_when`` (task_type via the
    crosswalk, modality via aliases). A recipe declares no explicit taxonomy block, so
    this is a projection. ``unmapped_task_types`` flags declarations the crosswalk
    doesn't cover (a data-quality signal)."""
    vocab = load_vocab()
    mod_alias = vocab.get("modality_aliases", {})
    out: List[Dict[str, Any]] = []
    for r in registry.recipes.values():
        aw = r.applies_when or {}
        tts = _as_list(aw.get("task_type"))
        mods = _as_list(aw.get("data_modality"))
        intents, tasks, unmapped = set(), set(), []
        for tt in tts:
            c = canonicalize(str(tt))
            if c:
                intents.add(c["intent"]); tasks.add(c["task"])
            else:
                unmapped.append(tt)
        modset = [m for m in (mod_alias.get(str(x).strip().lower()) for x in mods) if m]
        try:
            prov = registry.provenance("recipes", r.id)
        except Exception:
            prov = None
        out.append({
            "id": r.id,
            "provenance": prov,
            "intents": sorted(intents),
            "tasks": sorted(tasks),
            "task_families": sorted({_task_family(t) for t in tasks}),
            "modality": sorted(set(modset)),
            "declared_task_type": tts,
            "unmapped_task_types": unmapped,
        })
    return sorted(out, key=lambda x: x["id"])


def recipe_coverage(registry) -> Dict[str, Any]:
    """Intent × task-family matrix over the recipe library + per-axis distributions,
    mirroring ``coverage()`` for specs. Recipe primary intent = first sorted intent."""
    vocab = load_vocab()
    intents = list(vocab["intent"].keys())
    families = list(vocab["task"].keys())
    recs = recipe_index(registry)

    matrix: Dict[str, Dict[str, List[str]]] = {i: {f: [] for f in families} for i in intents}
    intent_dist: Dict[str, int] = {}
    family_dist: Dict[str, int] = {}
    modality_dist: Dict[str, int] = {}
    unmapped: List[Dict[str, Any]] = []

    for r in recs:
        if r["unmapped_task_types"]:
            unmapped.append({"id": r["id"], "declared": r["unmapped_task_types"]})
        primary = r["intents"][0] if r["intents"] else None
        if primary:
            intent_dist[primary] = intent_dist.get(primary, 0) + 1
        for fam in r["task_families"]:
            family_dist[fam] = family_dist.get(fam, 0) + 1
            for i in r["intents"]:
                if i in matrix and fam in matrix[i]:
                    matrix[i][fam].append(r["id"])
        for m in r["modality"]:
            modality_dist[m] = modality_dist.get(m, 0) + 1

    populated = {(i, f) for i in intents for f in families if matrix[i][f]}
    return {
        "intents": intents,
        "task_families": families,
        "matrix": matrix,
        "total_recipes": len(recs),
        "intent_distribution": intent_dist,
        "family_distribution": family_dist,
        "modality_distribution": modality_dist,
        "unmapped": unmapped,
        "populated_cells": [list(c) for c in sorted(populated)],
    }
