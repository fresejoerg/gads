from typing import Dict, Any, Optional, List
import json
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent
from gads.core.prompts import prompt_registry

class RouterInput(BaseModel):
    objective: str
    available_recipes: List[Dict[str, Any]] = []

class RouterOutput(BaseModel):
    task_type: str = Field(description="One of: binary_classification, regression, time_series, eda, clustering, thematic_analysis, semantic_search, or 'unknown'")
    data_modality: str = Field(description="One of: tabular, image, text, unstructured_text, audio, or 'unknown'")
    matched_recipe_id: Optional[str] = Field(None, description="The ID of a recipe that perfectly matches the methodology required for the objective.")
    confidence: float = Field(description="Score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief justification for the classification and recipe choice.")

class DataScienceRouter(BaseAgent[RouterInput, RouterOutput]):
    def __init__(self, model: str = "claude-haiku-4.5"):
        # Use a cheaper/faster model for classification as per Opus suggestion
        super().__init__(
            name="Router",
            model=model,
            system_prompt=prompt_registry.get_prompt("Router"),
            output_schema=RouterOutput
        )

    async def run(self, input_data: RouterInput, stream_callback=None, **kwargs) -> Any:
        # Refresh prompt from registry
        self.system_prompt = prompt_registry.get_prompt(self.name)
        
        recipes_str = json.dumps(input_data.available_recipes, indent=2)
        formatted_prompt = self.system_prompt.format(recipes_json=recipes_str)

        messages = [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": f"OBJECTIVE: {input_data.objective}"}
        ]
        
        from gads.core.llm import get_structured_completion
        content = await get_structured_completion(
            model=self.model,
            response_model=self.output_schema,
            messages=messages,
            stream_callback=stream_callback,
            **kwargs
        )
        
        from gads.agents.base import AgentResponse
        return AgentResponse(content=content, model_used=self.model)
