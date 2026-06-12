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
    def __init__(self, recipes_dir: str):
        self.recipes_dir = recipes_dir
        self.skills_dir = os.path.join(os.path.dirname(recipes_dir), "skills")
        self.recipes: Dict[str, Recipe] = {}
        self.skills: Dict[str, Skill] = {}
        self._recipe_filepaths: Dict[str, str] = {}
        self.load_recipes()
        self.load_skills()

    def load_recipes(self):
        # ... rest of load_recipes logic ...
        if not os.path.exists(self.recipes_dir):
            print(f"  [Registry] Warning: Recipes directory not found: {self.recipes_dir}")
            return

        for filename in os.listdir(self.recipes_dir):
            if filename.endswith(".md"):
                path = os.path.join(self.recipes_dir, filename)
                try:
                    with open(path, "r") as f:
                        content = f.read()

                    # Extract YAML frontmatter (between --- and ---)
                    match = re.search(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
                    if match:
                        yaml_str = match.group(1)
                        body_str = match.group(2)

                        data = yaml.safe_load(yaml_str)
                        # Extract Rationale from body; preserve any frontmatter
                        # `rationale:` rather than overwriting it with an empty match.
                        data["rationale"] = _extract_rationale(body_str, data.get("rationale", ""))

                        recipe = Recipe(**data)
                        self.recipes[recipe.id] = recipe
                        self._recipe_filepaths[recipe.id] = path
                        print(f"  [Registry] Loaded recipe: {recipe.id} (v{recipe.version})")
                except Exception as e:
                    print(f"  [Registry] Error loading {filename}: {e}")

    def load_skills(self):
        """Parses all .md files in the skills directory and extracts YAML frontmatter."""
        if not os.path.exists(self.skills_dir):
            print(f"  [Registry] Warning: Skills directory not found: {self.skills_dir}")
            return

        self.skills = {}
        for filename in os.listdir(self.skills_dir):
            if filename.endswith(".md"):
                path = os.path.join(self.skills_dir, filename)
                try:
                    with open(path, "r") as f:
                        content = f.read()
                    
                    match = re.search(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
                    if match:
                        yaml_str = match.group(1)
                        body_str = match.group(2).strip()
                        
                        data = yaml.safe_load(yaml_str)
                        data["content"] = body_str
                        
                        skill = Skill(**data)
                        self.skills[skill.id] = skill
                        print(f"  [Registry] Loaded skill: {skill.id}")
                except Exception as e:
                    print(f"  [Registry] Error loading skill {filename}: {e}")

    def get_recipe(self, recipe_id: str) -> Optional[Recipe]:
        return self.recipes.get(recipe_id)

    def get_recipe_filepath(self, recipe_id: str) -> Optional[str]:
        return self._recipe_filepaths.get(recipe_id)

    def list_recipes(self) -> List[str]:
        return list(self.recipes.keys())

    def list_recipe_files(self) -> List[str]:
        """Returns sorted list of .md filenames in the recipes directory."""
        if not os.path.exists(self.recipes_dir): return []
        return sorted([f for f in os.listdir(self.recipes_dir) if f.endswith(".md")])

    def get_raw_recipe(self, filename: str) -> str:
        """Returns the raw content of a recipe file."""
        path = os.path.join(self.recipes_dir, filename)
        if not os.path.exists(path): raise ValueError("Recipe file not found")
        with open(path, "r") as f:
            return f.read()

    def save_raw_recipe(self, filename: str, content: str):
        """Validates and saves a raw recipe file."""
        # 1. Validation
        match = re.search(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
        if not match:
            raise ValueError("Invalid format: Missing YAML frontmatter (--- ... ---)")
        
        try:
            yaml_str = match.group(1)
            body_str = match.group(2)
            data = yaml.safe_load(yaml_str)

            # Extract Rationale for validation (preserve frontmatter fallback)
            data["rationale"] = _extract_rationale(body_str, data.get("rationale", ""))

            # Validate with Pydantic
            Recipe(**data)
        except Exception as e:
            raise ValueError(f"Validation failed: {str(e)}")

        # 2. Save
        if not filename.endswith(".md"): filename += ".md"
        path = os.path.join(self.recipes_dir, filename)
        with open(path, "w") as f:
            f.write(content)
        
        # 3. Reload
        self.load_recipes()

    def list_skill_files(self) -> List[str]:
        """Returns sorted list of .md filenames in the skills directory."""
        if not os.path.exists(self.skills_dir): return []
        return sorted([f for f in os.listdir(self.skills_dir) if f.endswith(".md")])

    def get_raw_skill(self, filename: str) -> str:
        """Returns the raw content of a skill file."""
        path = os.path.join(self.skills_dir, filename)
        if not os.path.exists(path): raise ValueError("Skill file not found")
        with open(path, "r") as f:
            return f.read()

    def save_raw_skill(self, filename: str, content: str):
        """Validates and saves a raw skill file."""
        match = re.search(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
        if not match:
            raise ValueError("Invalid format: Missing YAML frontmatter (--- ... ---)")
        
        try:
            yaml_str = match.group(1)
            data = yaml.safe_load(yaml_str)
            Skill(**data)
        except Exception as e:
            raise ValueError(f"Validation failed: {str(e)}")

        if not filename.endswith(".md"): filename += ".md"
        path = os.path.join(self.skills_dir, filename)
        with open(path, "w") as f:
            f.write(content)
        
        self.load_skills()

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

    def get_skills_summary(self) -> List[Dict[str, Any]]:
        """Returns a list of available skills with their IDs and triggers for agent awareness."""
        return [{"id": s.id, "triggers": s.triggers} for s in self.skills.values()]

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
