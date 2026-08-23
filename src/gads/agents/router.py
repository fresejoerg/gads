from typing import Dict, Any, Optional, List
import json
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent
from gads.core.prompts import prompt_registry
from gads.core import taxonomy as tx

class RouterInput(BaseModel):
    objective: str
    available_recipes: List[Dict[str, Any]] = []

# The label vocabulary is DERIVED from taxonomy.yaml, never restated here. Three
# vocabularies had drifted apart — the spec taxonomy, this enum, and recipe
# `applies_when` — and the enum was the narrowest of the three: 11 of 29 recipes
# declared a task_type no Router label could equal, so their coverage was
# unreachable by classification (approach_docs/024 §1).
class RouterOutput(BaseModel):
    task_type: str = Field(description=(
        "The canonical task, from the controlled vocabulary in the system prompt "
        "(`family` or `family.subtype`, e.g. 'classification.binary', "
        "'regression.survival'). 'unknown' if nothing fits."))
    data_modality: str = Field(description=(
        "The canonical modality, from the controlled vocabulary in the system prompt "
        "(e.g. 'tabular', 'relational', 'text'). 'unknown' if nothing fits."))
    matched_recipe_id: Optional[str] = Field(None, description="The ID of a recipe that perfectly matches the methodology required for the objective.")
    confidence: float = Field(description="Score between 0.0 and 1.0")
    reasoning: Optional[str] = Field(None, description="Brief justification for the classification and recipe choice.")

class DataScienceRouter(BaseAgent[RouterInput, RouterOutput]):
    def __init__(self, model: str = "local_model"):
        # Use a cheaper/faster model for classification as per Opus suggestion
        super().__init__(
            name="Router",
            model=model,
            system_prompt=prompt_registry.get_prompt("Router"),
            output_schema=RouterOutput
        )

    async def run(self, input_data: RouterInput, stream_callback=None, **kwargs) -> Any:
        # Refresh prompt from registry
        base_prompt = prompt_registry.get_prompt(self.name)
        recipes_str = json.dumps(input_data.available_recipes, indent=2)
        formatted_prompt = base_prompt.format(
            recipes_json=recipes_str,
            task_vocab=tx.render_task_vocabulary(),
            modality_vocab=tx.render_modality_vocabulary(),
        )

        # Router output is compact JSON (~200 tokens), but reasoning local models
        # (gemma emits reasoning_content that counts against the completion budget)
        # need headroom before the answer. 1024 guillotined legitimate routing CoT on
        # dense objectives → finish_reason='length' → fatal halt in the drafted lane.
        # 4096 fits reasoning + JSON while still failing fast on a true runaway (8192).
        kwargs.setdefault("max_tokens", 4096)
        # Use super().run to get streaming support
        res = await super().run(
            f"OBJECTIVE: {input_data.objective}",
            system_prompt=formatted_prompt,
            **kwargs
        )
        
        # Validate matched_recipe_id against available recipes to prevent hallucination
        if res and res.content:
            valid_ids = {r.get("id") for r in input_data.available_recipes if r.get("id")}
            if res.content.matched_recipe_id and res.content.matched_recipe_id not in valid_ids:
                print(f"  [Router] Hallucinated matched_recipe_id '{res.content.matched_recipe_id}' reset to None.", flush=True)
                res.content.matched_recipe_id = None

            # Normalize the labels to the controlled vocabulary. A model that answers with
            # a near-miss synonym ('survival_analysis', 'timeseries') is right about the
            # task and should not be punished for the wording; a label that canonicalizes
            # to nothing becomes 'unknown' rather than silently poisoning the coverage
            # oracle, the taxonomy tagging and the SampleBudget threshold downstream.
            for field, canon in (("task_type", tx.canonical_task),
                                 ("data_modality", tx.canonical_modality)):
                raw = getattr(res.content, field)
                fixed = canon(raw) or "unknown"
                if fixed != raw:
                    print(f"  [Router] {field} {raw!r} -> {fixed!r} (controlled vocabulary).", flush=True)
                    setattr(res.content, field, fixed)

        return res

