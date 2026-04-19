import os
import yaml
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class RecipeTask(BaseModel):
    id: str
    intent: str
    worker_tier: str
    depends_on: List[str] = []
    produces: List[str] = []
    postconditions: List[str] = []
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

class KnowledgeRegistry:
    def __init__(self, recipes_dir: str):
        self.recipes_dir = recipes_dir
        self.recipes: Dict[str, Recipe] = {}
        self.load_recipes()

    def load_recipes(self):
        """Parses all .md files in the recipes directory and extracts YAML frontmatter."""
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
                        # Extract Rationale from body for the Recipe model
                        rationale_match = re.search(r"# .*?\n\n## Rationale\n(.*?)\n\n##", body_str, re.DOTALL)
                        data["rationale"] = rationale_match.group(1).strip() if rationale_match else ""
                        
                        recipe = Recipe(**data)
                        self.recipes[recipe.id] = recipe
                        print(f"  [Registry] Loaded recipe: {recipe.id} (v{recipe.version})")
                except Exception as e:
                    print(f"  [Registry] Error loading {filename}: {e}")

    def get_recipe(self, recipe_id: str) -> Optional[Recipe]:
        return self.recipes.get(recipe_id)

    def list_recipes(self) -> List[str]:
        return list(self.keys())

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
