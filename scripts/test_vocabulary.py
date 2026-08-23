"""Controlled-vocabulary regression guard (approach_docs/024 §1) — no LLM calls.

The Router's labels, recipe `applies_when` and spec `taxonomy:` blocks are three places
that name the same things. They had silently drifted apart: 11 of 29 recipes declared a
task_type no Router label could ever equal, 6 specs carried terms outside the vocabulary,
and the coverage oracle scored routing against the resulting mismatch. Nothing failed
loudly, because nothing checked.

This asserts that every term in every one of those places resolves through
`taxonomy.yaml`, so a new recipe or spec cannot reintroduce the drift.

    PYTHONPATH=src uv run python scripts/test_vocabulary.py
"""
import os
import re
import sys

sys.path.insert(0, "src")

import yaml

from gads.core import taxonomy as tx
from gads.core.knowledge import KnowledgeRegistry
from gads.core.prompts import FACTORY_DEFAULTS, REQUIRED_VARS

FM = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  — ' + detail) if detail and not ok else ''}")
    if not ok:
        failures.append(f"{label}: {detail}")


def main() -> int:
    vocab = tx.load_vocab()
    registry = KnowledgeRegistry("src/gads/knowledge/recipes")

    print("\n1. taxonomy.yaml is internally consistent")
    bad = [k for k, v in (vocab.get("crosswalk") or {}).items()
           if not tx.canonical_task(v.get("task"))]
    check("every crosswalk target is a valid task term", not bad, str(bad))
    bad = [f for f in (vocab.get("training_task_families") or [])
           if f not in (vocab.get("task") or {})]
    check("every training_task_family is a real task family", not bad, str(bad))
    bad = [f for f in (vocab.get("family_defaults") or {}) if f not in (vocab.get("task") or {})]
    check("every family_default names a real task family", not bad, str(bad))
    bad = [v for v in (vocab.get("modality_aliases") or {}).values()
           if v not in (vocab.get("modality") or {})]
    check("every modality alias targets a real modality", not bad, str(bad))

    print("\n2. every recipe's applies_when resolves to the vocabulary")
    bad_t, bad_m = [], []
    for rid, rec in registry.recipes.items():
        aw = rec.applies_when or {}
        for t in (aw.get("task_type") or []):
            if not tx.canonical_task(t):
                bad_t.append(f"{rid}:{t}")
        for m in (aw.get("data_modality") or []):
            if not tx.canonical_modality(m):
                bad_m.append(f"{rid}:{m}")
    check(f"task_type terms canonicalize ({len(registry.recipes)} recipes)", not bad_t, str(bad_t))
    check("data_modality terms canonicalize", not bad_m, str(bad_m))

    print("\n3. every recipe is reachable — or explicitly pin-only")
    unreachable = []
    for rid, rec in registry.recipes.items():
        aw = rec.applies_when or {}
        if registry.is_pin_only(aw):
            continue
        labels = {"task_type": tx.canonical_task((aw.get("task_type") or [None])[0]),
                  "data_modality": tx.canonical_modality((aw.get("data_modality") or [None])[0])}
        if rid not in [r.id for r in registry.find_matches(labels)]:
            unreachable.append(rid)
    check("each routable recipe is found by the oracle under its own labels",
          not unreachable, str(unreachable))
    pin_only = [r for r in registry.recipes.values() if registry.is_pin_only(r.applies_when)]
    print(f"        ({len(pin_only)} pin-only research instrument(s), withheld from the "
          f"agent catalogue by design)")

    print("\n4. every spec's taxonomy block validates")
    bad_specs = []
    for name in sorted(os.listdir("specs")):
        if not name.endswith(".md"):
            continue
        m = FM.search(open(os.path.join("specs", name)).read())
        if not m:
            continue
        block = (yaml.safe_load(m.group(1)) or {}).get("taxonomy")
        if not block:
            continue
        errs = tx.validate_tags(block)["errors"]
        if errs:
            bad_specs.append(f"{name}: {errs}")
    check("all tagged specs pass validate_tags", not bad_specs, "; ".join(bad_specs))

    print("\n5. the Router prompt still renders from the vocabulary")
    try:
        rendered = FACTORY_DEFAULTS["Router"].format(
            recipes_json="[]",
            task_vocab=tx.render_task_vocabulary(),
            modality_vocab=tx.render_modality_vocabulary(),
        )
        check("Router prompt formats with the derived vocab blocks", "classification.binary" in rendered)
    except KeyError as e:
        check("Router prompt formats with the derived vocab blocks", False, f"missing placeholder {e}")
    check("REQUIRED_VARS['Router'] lists the vocab placeholders",
          {"task_vocab", "modality_vocab"} <= set(REQUIRED_VARS.get("Router", [])))

    print("\n" + "=" * 70)
    if failures:
        print(f"FAILED ({len(failures)})")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All vocabulary checks passed.")
    return 0


sys.exit(main())
