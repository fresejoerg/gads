from typing import Dict, Any, Optional, List
import json
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent
from gads.core.prompts import prompt_registry

class RouterInput(BaseModel):
    objective: str
    available_recipes: List[Dict[str, Any]] = []

class RouterOutput(BaseModel):
    task_type: str = Field(description="One of: binary_classification, regression, time_series, eda, clustering, thematic_analysis, semantic_search, causal_inference, or 'unknown'")
    data_modality: str = Field(description="One of: tabular, image, text, unstructured_text, audio, or 'unknown'")
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
        formatted_prompt = base_prompt.format(recipes_json=recipes_str)

        # Router output is compact JSON (~200 tokens) — cap to fail fast on repetition loops
        kwargs.setdefault("max_tokens", 1024)
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
                
        return res

