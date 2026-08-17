"""Deep validation for knowledge items (recipes, skills, native nodes).

Server-side so every client — the Knowledge Studio SPA, CI, the legacy editor — is
protected uniformly (approach_docs/017 §3). Each validator returns
``{"errors": [...], "warnings": [...]}``: errors block a write, warnings do not.

Pydantic shape checks (Recipe/Skill) live in ``knowledge.py``; this module adds the
structural checks Pydantic can't express — DAG integrity, id collisions, trigger
sanity, capability whitelisting, and Python-hazard scanning for native nodes.
"""
from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional

import yaml

from gads.core.knowledge import Recipe, Skill, _extract_rationale

# Packages the sandbox actually ships (skills/sandbox_environment.md). Unknown
# capabilities aren't fatal — the recipe may target a future kernel — but they warn.
KNOWN_CAPABILITIES = {
    "pandas", "numpy", "polars", "pyarrow", "duckdb",
    "sklearn", "scikit-learn", "joblib", "torch", "lightgbm", "xgboost", "shap",
    "autogluon", "dowhy", "econml", "causalml", "causallearn", "linearmodels",
    "statsmodels", "pymc", "arviz", "bambi", "pycausalimpact", "pgmpy",
    "sentence-transformers", "nltk", "textblob",
    "matplotlib", "seaborn", "plotly", "kaleido", "networkx",
    "lifelines", "scikit-survival", "sksurv", "skore",
    "optuna", "catboost", "implicit", "scipy",
}

# Triggers this short/common would over-fire on ordinary code (the list->tolist() class
# of bug the design doc cites). Substring-y single tokens get a sanity warning.
RISKY_TRIGGER_TOKENS = {
    "list", "map", "set", "str", "sum", "df", "data", "id", "run", "fit", "plot",
    "model", "test", "train", "col", "row", "np", "pd", "type", "print", "value",
}

# Imports/calls that don't belong in a sandbox-injected native node.
NATIVE_HAZARDS = ("subprocess", "socket", "requests", "urllib", "httpx", "pip",
                  "os.system", "eval(", "exec(", "__import__", "shutil.rmtree")

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)", re.DOTALL)


def _split(content: str):
    m = _FRONTMATTER.search(content or "")
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _result(errors: List[str], warnings: List[str]) -> Dict[str, List[str]]:
    return {"errors": errors, "warnings": warnings}


# --------------------------------------------------------------------------- #
# Recipes
# --------------------------------------------------------------------------- #
def validate_recipe(content: str, registry: Any = None, filename: Optional[str] = None) -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    yaml_str, body = _split(content)
    if yaml_str is None:
        return _result(["Missing YAML frontmatter (--- ... ---)."], warnings)
    try:
        data = yaml.safe_load(yaml_str) or {}
    except yaml.YAMLError as e:
        return _result([f"Frontmatter is not valid YAML: {e}"], warnings)

    data["rationale"] = _extract_rationale(body or "", data.get("rationale", ""))
    try:
        recipe = Recipe(**data)
    except Exception as e:
        return _result([f"Schema validation failed: {e}"], warnings)

    # --- DAG integrity ---
    node_ids = [t.id for t in recipe.dag]
    if not node_ids:
        errors.append("Recipe DAG has no nodes.")
    dupes = {n for n in node_ids if node_ids.count(n) > 1}
    if dupes:
        errors.append(f"Duplicate DAG node id(s): {', '.join(sorted(dupes))}.")
    idset = set(node_ids)
    for t in recipe.dag:
        for dep in t.depends_on:
            if dep not in idset:
                errors.append(f"Node '{t.id}' depends_on unknown node '{dep}'.")
    cycle = _find_cycle(recipe.dag)
    if cycle:
        errors.append(f"Cycle in DAG depends_on: {' -> '.join(cycle)}.")

    # --- produces / required_metrics consistency (soft) ---
    produced = {p for t in recipe.dag for p in t.produces}
    for t in recipe.dag:
        for rm in t.required_metrics:
            if rm not in produced:
                warnings.append(f"Node '{t.id}' requires metric '{rm}' that no node lists in produces.")

    # --- capabilities whitelist (soft) ---
    caps = (recipe.requires or {}).get("capabilities", []) or []
    for c in caps:
        if str(c).lower() not in KNOWN_CAPABILITIES:
            warnings.append(f"Capability '{c}' is not in the known sandbox package set.")

    # --- attached skills resolve (soft) ---
    if registry is not None:
        known_skills = set(getattr(registry, "skills", {}).keys())
        for t in recipe.dag:
            for sk in t.attached_skills or []:
                if sk not in known_skills:
                    warnings.append(f"Node '{t.id}' attaches unknown skill '{sk}'.")

    # --- id uniqueness across the registry ---
    _id_collision(errors, registry, "recipe", recipe.id, filename)

    return _result(errors, warnings)


def _find_cycle(dag) -> Optional[List[str]]:
    graph = {t.id: list(t.depends_on) for t in dag}
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in graph}
    stack: List[str] = []

    def dfs(n: str) -> Optional[List[str]]:
        color[n] = GRAY
        stack.append(n)
        for m in graph.get(n, []):
            if m not in color:
                continue
            if color[m] == GRAY:
                return stack[stack.index(m):] + [m]
            if color[m] == WHITE:
                c = dfs(m)
                if c:
                    return c
        color[n] = BLACK
        stack.pop()
        return None

    for n in graph:
        if color[n] == WHITE:
            c = dfs(n)
            if c:
                return c
    return None


# --------------------------------------------------------------------------- #
# Skills
# --------------------------------------------------------------------------- #
def validate_skill(content: str, registry: Any = None, filename: Optional[str] = None) -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    yaml_str, _body = _split(content)
    if yaml_str is None:
        return _result(["Missing YAML frontmatter (--- ... ---)."], warnings)
    try:
        data = yaml.safe_load(yaml_str) or {}
    except yaml.YAMLError as e:
        return _result([f"Frontmatter is not valid YAML: {e}"], warnings)
    try:
        skill = Skill(**data)
    except Exception as e:
        return _result([f"Schema validation failed: {e}"], warnings)

    triggers = [t for t in (skill.triggers or [])]
    if not triggers:
        errors.append("Skill has no triggers; it can never keyword-match.")
    if any(not str(t).strip() for t in triggers):
        errors.append("Skill has an empty trigger string.")
    for t in triggers:
        tok = str(t).strip().lower()
        if tok and " " not in tok and (len(tok) < 4 or tok in RISKY_TRIGGER_TOKENS):
            warnings.append(f"Trigger '{t}' is short/common and may over-fire on ordinary code; "
                            f"prefer a multi-word or distinctive phrase.")
    if not (skill.description or "").strip():
        warnings.append("Skill has no description; it won't be discoverable in the library.")

    _id_collision(errors, registry, "skill", skill.id, filename)
    return _result(errors, warnings)


# --------------------------------------------------------------------------- #
# Native nodes (Python injected into sandbox code) — distinct safety model
# --------------------------------------------------------------------------- #
def validate_native(content: str, registry: Any = None, filename: Optional[str] = None) -> Dict[str, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    try:
        ast.parse(content or "")
    except SyntaxError as e:
        return _result([f"Python syntax error: line {e.lineno}: {e.msg}"], warnings)

    lowered = content or ""
    for hz in NATIVE_HAZARDS:
        if hz in lowered:
            warnings.append(f"Native node references '{hz}', which is unusual for sandbox-injected "
                            f"code — confirm this is intended (it will run in the kernel).")
    # export-contract hint (approach_docs/017 §4): a native module should surface a preamble.
    if "PREAMBLE" not in content and "preamble" not in content:
        warnings.append("Native module exposes no *_PREAMBLE / preamble symbol; the executor "
                        "injects native nodes via their preamble.")
    return _result(errors, warnings)


# --------------------------------------------------------------------------- #
def _id_collision(errors: List[str], registry: Any, kind: str, new_id: str, filename: Optional[str]):
    if registry is None or not new_id:
        return
    if kind == "recipe":
        paths = getattr(registry, "_recipe_filepaths", {}) or {}
        existing = paths.get(new_id)
        if existing and filename and not str(existing).endswith(str(filename)):
            errors.append(f"Recipe id '{new_id}' is already used by {existing}.")
    elif kind == "skill":
        if new_id in (getattr(registry, "skills", {}) or {}) and filename:
            # skills are keyed by id in the registry; a same-id different-file write collides
            errors.append(f"Skill id '{new_id}' already exists; editing must keep the same file.")


VALIDATORS = {"recipe": validate_recipe, "skill": validate_skill, "native": validate_native}


def validate(item_type: str, content: str, registry: Any = None, filename: Optional[str] = None) -> Dict[str, List[str]]:
    fn = VALIDATORS.get(item_type)
    if not fn:
        return _result([f"Unknown item type '{item_type}' (expected recipe|skill|native)."], [])
    return fn(content, registry=registry, filename=filename)
