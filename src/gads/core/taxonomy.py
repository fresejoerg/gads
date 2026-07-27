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


def _task_family(task_val: str) -> str:
    return task_val.split(".", 1)[0] if "." in task_val else task_val


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
