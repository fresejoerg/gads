"""Determine the delegation-dial rung for a spec WITHOUT running it.

Usage:
    PYTHONPATH=src uv run python scripts/dial_rung.py --spec specs/amlb_segment.md
    PYTHONPATH=src uv run python scripts/dial_rung.py --all

Static analysis only: reads the spec frontmatter, resolves a pinned recipe against the
registry, and applies the same rung rules the workflow uses (core/dial.py). Specs
without a pinned recipe are reported as drafted-lane (D0/D1) — at runtime the Router
may still match a recipe and lift the run to D3+; that outcome is recorded in the
ledger, not predictable here.
"""
import argparse
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from gads.core.dial import compiled_plan_dial, drafted_plan_dial  # noqa: E402
from gads.core.knowledge import KnowledgeRegistry  # noqa: E402


def spec_frontmatter(path: Path) -> dict:
    m = re.search(r"^---\s*\n(.*?)\n---\s*\n", path.read_text(), re.DOTALL)
    return yaml.safe_load(m.group(1)) if m else {}


def analyze(spec_path: Path, registry: KnowledgeRegistry) -> dict:
    fm = spec_frontmatter(spec_path)
    recipe_id = fm.get("recipe_id")
    recipe = registry.get_recipe(recipe_id) if recipe_id else None
    if recipe:
        info = compiled_plan_dial([n.model_dump() for n in recipe.dag], selection="pinned")
        info["recipe_id"] = recipe.id
    else:
        info = drafted_plan_dial(fm)
        if recipe_id:
            info["warning"] = f"recipe_id '{recipe_id}' not found in registry"
        else:
            info["note"] = "no pinned recipe — Router may still match one at runtime (would lift to D3+)"
    info["spec"] = spec_path.name
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", help="path to a spec .md file")
    ap.add_argument("--all", action="store_true", help="analyze every spec in specs/")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    registry = KnowledgeRegistry(str(root / "src/gads/knowledge/recipes"))
    targets = sorted((root / "specs").glob("*.md")) if args.all else [Path(args.spec)]
    if not args.all and not args.spec:
        ap.error("--spec or --all required")

    for t in targets:
        info = analyze(t, registry)
        rung = info["rung"]
        detail = ", ".join(f"{k}={v}" for k, v in info["task_rungs"].items()) or info.get("note", "drafted")
        print(f"{info['spec']:42s} {rung}  ({info['selection']}"
              + (f", recipe={info['recipe_id']}" if info.get("recipe_id") else "")
              + f")  [{detail}]"
              + (f"  WARNING: {info['warning']}" if info.get("warning") else ""))


if __name__ == "__main__":
    main()
