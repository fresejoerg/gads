import os
import yaml
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

def _extract_rationale(body_str: str, fallback: str = "") -> str:
    """Pull the ## Rationale section from a recipe body.

    Tolerant of recipes that end the section with EOF rather than a trailing `##`
    heading. Falls back to any frontmatter `rationale:` value (passed in) when the
    body has no parseable section, instead of clobbering it with an empty string.
    """
    match = re.search(r"##\s*Rationale\s*\n(.*?)(?:\n\s*##|\Z)", body_str, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        if extracted:
            return extracted
    return (fallback or "").strip()


class RecipeTask(BaseModel):
    id: str
    intent: str
    worker_tier: str
    depends_on: List[str] = []
    produces: List[str] = []
    # NOTE: `postconditions` are advisory contract-design aids that document a node's
    # success criteria for recipe authors. They are NOT evaluated at runtime — the
    # ExecutionHub builds the enforced contract from `produces`/`required_metrics`
    # only. Hard-asserting these arbitrary expressions would cause false failures on
    # forgetful local models, against the project's deliberate soft-fail philosophy.
    postconditions: List[str] = []
    required_metrics: List[str] = []
    # Skill IDs to inject (full body) into the Coder prompt for this node. When set,
    # the RecipeEnforcer uses these verbatim instead of keyword-scoring the intent.
    attached_skills: List[str] = []
    skippable_if: Optional[str] = None
    rationale_required: bool = False

class Recipe(BaseModel):
    id: str
    version: str
    author: str
    applies_when: Dict[str, Any]
    requires: Dict[str, Any]
    dag: List[RecipeTask]
    invariants: List[str] = []
    rationale: str = ""

class Skill(BaseModel):
    id: str
    triggers: List[str]
    description: str = ""
    content: str = ""

class KnowledgeRegistry:
    TYPES = ("recipes", "skills", "native")

    def __init__(self, recipes_dir: str, overlay_root: str = "gads_data/knowledge"):
        # Shipped baseline (version-controlled IP, read-only in the studio).
        self.recipes_dir = recipes_dir
        shipped_root = os.path.dirname(recipes_dir)
        self.skills_dir = os.path.join(shipped_root, "skills")
        self.native_dir = os.path.join(shipped_root, "native")
        self._shipped = {"recipes": self.recipes_dir, "skills": self.skills_dir, "native": self.native_dir}
        # Writable user overlay: shadows shipped by id/filename. gads_data/ is git-ignored,
        # so user-authored/edited items never touch the shipped IP (approach_docs/017 §1).
        self.overlay_root = overlay_root
        self._overlay = {t: os.path.join(overlay_root, t) for t in self.TYPES}

        self.recipes: Dict[str, Recipe] = {}
        self.skills: Dict[str, Skill] = {}
        self._recipe_filepaths: Dict[str, str] = {}
        self._skill_filepaths: Dict[str, str] = {}
        # id -> "shipped" | "overlay" | "overridden", per type.
        self._provenance: Dict[str, Dict[str, str]] = {t: {} for t in self.TYPES}

        # Lazy semantic index (embedding-based skill retrieval). Import is local so a
        # missing fastembed never breaks the registry — semantic lookups just return [].
        from gads.core.skill_semantics import SkillEmbeddingIndex
        self.semantic_index = SkillEmbeddingIndex()
        self.load_recipes()
        self.load_skills()

    # ---- overlay-aware path helpers -------------------------------------- #
    def _dirs(self, item_type: str):
        """(shipped_dir, overlay_dir) for a knowledge type."""
        return self._shipped[item_type], self._overlay[item_type]

    def _resolve_path(self, item_type: str, filename: str) -> Optional[str]:
        """Path to a file by name, overlay shadowing shipped."""
        shp, ov = self._dirs(item_type)
        for base in (ov, shp):
            p = os.path.join(base, filename)
            if os.path.exists(p):
                return p
        return None

    def _list_files(self, item_type: str, ext: str = ".md") -> List[str]:
        shp, ov = self._dirs(item_type)
        names = set()
        for base in (shp, ov):
            if os.path.isdir(base):
                names.update(f for f in os.listdir(base) if f.endswith(ext))
        return sorted(names)

    def _write_overlay(self, item_type: str, filename: str, content: str) -> str:
        _, ov = self._dirs(item_type)
        os.makedirs(ov, exist_ok=True)
        path = os.path.join(ov, filename)
        with open(path, "w") as f:
            f.write(content)
        return path

    def provenance(self, item_type: str, item_id: Optional[str] = None):
        prov = self._provenance.get(item_type, {})
        return prov.get(item_id) if item_id is not None else dict(prov)

    def reset_to_shipped(self, item_type: str, filename: str):
        """Delete the overlay copy of a file, reverting to the shipped version."""
        _, ov = self._dirs(item_type)
        path = os.path.join(ov, filename)
        if not os.path.exists(path):
            raise ValueError("No overlay copy to reset (item is shipped or overlay-only).")
        os.remove(path)
        if item_type == "recipes":
            self.load_recipes()
        elif item_type == "skills":
            self.load_skills()

    def load_recipes(self):
        """Load shipped recipes, then overlay (overlay shadows a shipped id -> 'overridden')."""
        self.recipes = {}
        self._recipe_filepaths = {}
        self._provenance["recipes"] = {}
        shp, ov = self._dirs("recipes")
        for base, prov in ((shp, "shipped"), (ov, "overlay")):
            if not os.path.isdir(base):
                continue
            for filename in sorted(os.listdir(base)):
                if not filename.endswith(".md"):
                    continue
                path = os.path.join(base, filename)
                try:
                    with open(path, "r") as f:
                        content = f.read()
                    match = re.search(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
                    if not match:
                        continue
                    data = yaml.safe_load(match.group(1))
                    # Preserve a frontmatter `rationale:` rather than clobbering with an empty match.
                    data["rationale"] = _extract_rationale(match.group(2), data.get("rationale", ""))
                    recipe = Recipe(**data)
                    was_shipped = recipe.id in self.recipes
                    self.recipes[recipe.id] = recipe
                    self._recipe_filepaths[recipe.id] = path
                    self._provenance["recipes"][recipe.id] = "overridden" if (prov == "overlay" and was_shipped) else prov
                    print(f"  [Registry] Loaded recipe: {recipe.id} (v{recipe.version}) [{self._provenance['recipes'][recipe.id]}]")
                except Exception as e:
                    print(f"  [Registry] Error loading {filename}: {e}")

    def load_skills(self):
        """Load shipped skills, then overlay (overlay shadows a shipped id -> 'overridden')."""
        self.skills = {}
        self._skill_filepaths = {}
        self._provenance["skills"] = {}
        shp, ov = self._dirs("skills")
        for base, prov in ((shp, "shipped"), (ov, "overlay")):
            if not os.path.isdir(base):
                continue
            for filename in sorted(os.listdir(base)):
                if not filename.endswith(".md"):
                    continue
                path = os.path.join(base, filename)
                try:
                    with open(path, "r") as f:
                        content = f.read()
                    match = re.search(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
                    if not match:
                        continue
                    data = yaml.safe_load(match.group(1))
                    data["content"] = match.group(2).strip()
                    skill = Skill(**data)
                    was_shipped = skill.id in self.skills
                    self.skills[skill.id] = skill
                    self._skill_filepaths[skill.id] = path
                    self._provenance["skills"][skill.id] = "overridden" if (prov == "overlay" and was_shipped) else prov
                    print(f"  [Registry] Loaded skill: {skill.id} [{self._provenance['skills'][skill.id]}]")
                except Exception as e:
                    print(f"  [Registry] Error loading skill {filename}: {e}")
        # Keep the semantic index in sync with the skill collection (hot-edits included).
        self.semantic_index.rebuild(list(self.skills.values()))

    def list_items(self, item_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Unified library listing across types with metadata + provenance (studio)."""
        out: List[Dict[str, Any]] = []
        if item_type in (None, "recipe"):
            for rid, rec in self.recipes.items():
                aw = rec.applies_when or {}
                out.append({
                    "type": "recipe", "id": rid,
                    "filename": os.path.basename(self._recipe_filepaths.get(rid, "")),
                    "version": rec.version,
                    "task_type": aw.get("task_type") or [],
                    "data_modality": aw.get("data_modality") or [],
                    "capabilities": (rec.requires or {}).get("capabilities", []) or [],
                    "provenance": self._provenance["recipes"].get(rid),
                })
        if item_type in (None, "skill"):
            for sid, sk in self.skills.items():
                out.append({
                    "type": "skill", "id": sid,
                    "filename": os.path.basename(self._skill_filepaths.get(sid, "")),
                    "description": sk.description, "triggers": sk.triggers,
                    "provenance": self._provenance["skills"].get(sid),
                })
        if item_type in (None, "native"):
            shp, ov = self._dirs("native")
            for fn in self.list_native_files():
                in_ov = os.path.exists(os.path.join(ov, fn))
                in_shp = os.path.exists(os.path.join(shp, fn))
                prov = "overridden" if (in_ov and in_shp) else ("overlay" if in_ov else "shipped")
                out.append({"type": "native", "id": fn, "filename": fn, "provenance": prov})
        return sorted(out, key=lambda i: (i["type"], i["id"]))

    def get_recipe(self, recipe_id: str) -> Optional[Recipe]:
        return self.recipes.get(recipe_id)

    def get_recipe_filepath(self, recipe_id: str) -> Optional[str]:
        return self._recipe_filepaths.get(recipe_id)

    def list_recipes(self) -> List[str]:
        return list(self.recipes.keys())

    def list_recipe_files(self) -> List[str]:
        """Sorted union of shipped + overlay recipe filenames."""
        return self._list_files("recipes")

    def get_raw_recipe(self, filename: str) -> str:
        """Raw content of a recipe file (overlay shadows shipped)."""
        path = self._resolve_path("recipes", filename)
        if not path:
            raise ValueError("Recipe file not found")
        with open(path, "r") as f:
            return f.read()

    def save_raw_recipe(self, filename: str, content: str):
        """Deep-validate and save a recipe to the OVERLAY (never mutates shipped IP)."""
        if not filename.endswith(".md"):
            filename += ".md"
        self._deep_validate("recipe", content, filename)
        self._write_overlay("recipes", filename, content)
        self.load_recipes()

    def list_skill_files(self) -> List[str]:
        """Sorted union of shipped + overlay skill filenames."""
        return self._list_files("skills")

    def get_raw_skill(self, filename: str) -> str:
        """Raw content of a skill file (overlay shadows shipped)."""
        path = self._resolve_path("skills", filename)
        if not path:
            raise ValueError("Skill file not found")
        with open(path, "r") as f:
            return f.read()

    def save_raw_skill(self, filename: str, content: str):
        """Deep-validate and save a skill to the OVERLAY (never mutates shipped IP)."""
        if not filename.endswith(".md"):
            filename += ".md"
        self._deep_validate("skill", content, filename)
        self._write_overlay("skills", filename, content)
        self.load_skills()

    # ---- native nodes (read; overlay-write lands in a later increment) ----- #
    def list_native_files(self) -> List[str]:
        """Sorted union of shipped + overlay native module filenames (.py, excl. dunder)."""
        return [f for f in self._list_files("native", ".py") if not f.startswith("__")]

    def get_raw_native(self, filename: str) -> str:
        path = self._resolve_path("native", filename)
        if not path:
            raise ValueError("Native module not found")
        with open(path, "r") as f:
            return f.read()

    def _deep_validate(self, item_type: str, content: str, filename: str):
        """Run the shared deep validator (approach_docs/017 §3); raise on errors so every
        write path — studio, legacy editor, CI — is protected uniformly."""
        from gads.core.knowledge_validation import validate
        result = validate(item_type, content, registry=self, filename=filename)
        if result["errors"]:
            raise ValueError("; ".join(result["errors"]))

    @staticmethod
    def _trigger_hits(triggers: List[str], desc_lower: str) -> int:
        """Count triggers that match on a word boundary (not bare substring).

        Substring matching caused spurious loads — e.g. the trigger 'list' firing on
        'tolist()' or 'top' on a 'fi_top' variable. Word-boundary matching keeps
        multi-word phrase triggers ('auto ml', 'treatment effect') working while
        eliminating mid-token false positives that dilute a small model's attention.
        """
        hits = 0
        for t in triggers:
            pattern = r"\b" + re.escape(t.lower().strip()) + r"\b"
            if re.search(pattern, desc_lower):
                hits += 1
        return hits

    def find_skills(self, task_description: str) -> List[Skill]:
        """Returns skills that match triggers in the task description (case-insensitive)."""
        desc_lower = task_description.lower()
        return [
            skill for skill in self.skills.values()
            if self._trigger_hits(skill.triggers, desc_lower) > 0
        ]

    def find_skills_scored(self, task_description: str) -> List[tuple]:
        """Returns [(Skill, hit_count)] for skills with at least one trigger match."""
        desc_lower = task_description.lower()
        results = []
        for skill in self.skills.values():
            hits = self._trigger_hits(skill.triggers, desc_lower)
            if hits > 0:
                results.append((skill, hits))
        return results

    def find_skills_semantic(self, task_description: str) -> List[tuple]:
        """Returns [(Skill, cosine_similarity)] via the embedding index, best first.

        Empty when semantic matching is disabled/unavailable. Intended for tasks with
        NO curated attached_skills — callers keep compiled benchmark prompts stable by
        not augmenting curated tasks (see skill_semantics module docstring).
        """
        return [
            (self.skills[sid], score)
            for sid, score in self.semantic_index.find(task_description)
            if sid in self.skills
        ]

    def find_skills_combined(self, task_description: str) -> List[tuple]:
        """Union of keyword and semantic matches as [(Skill, source, score)].

        Keyword matches come first (exact signal beats similarity); a skill matched by
        both is reported once, as keyword.
        """
        keyword = self.find_skills_scored(task_description)
        seen = {s.id for s, _ in keyword}
        combined = [(s, "keyword", float(hits)) for s, hits in keyword]
        combined += [
            (s, "semantic", score)
            for s, score in self.find_skills_semantic(task_description)
            if s.id not in seen
        ]
        return combined

    def get_skills_summary(self) -> List[Dict[str, Any]]:
        """Returns available skills with IDs, descriptions (for agent selection) and
        triggers (for relevance pruning — strip triggers before serializing into prompts)."""
        return [{"id": s.id, "description": s.description, "triggers": s.triggers} for s in self.skills.values()]

    def get_recipes_summary(self) -> List[Dict[str, Any]]:
        """Returns a list of available recipes with their IDs and rationale for agent awareness."""
        return [{"id": r.id, "rationale": r.rationale, "applies_when": r.applies_when} for r in self.recipes.values()]

    def find_matches(self, intent_tags: Dict[str, Any]) -> List[Recipe]:
        """
        Simple matching logic based on task_type and modality.
        In Phase 2, this will be handled by the RouterAgent.
        """
        matches = []
        target_type = intent_tags.get("task_type")
        target_modality = intent_tags.get("data_modality")
        
        for recipe in self.recipes.values():
            applies = recipe.applies_when
            if target_type in applies.get("task_type", []) and \
               target_modality in applies.get("data_modality", []):
                matches.append(recipe)
        return matches
