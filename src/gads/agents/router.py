from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent

class RouterInput(BaseModel):
    objective: str

class RouterOutput(BaseModel):
    task_type: str = Field(description="One of: binary_classification, regression, time_series, eda, clustering, thematic_analysis, or 'unknown'")
    data_modality: str = Field(description="One of: tabular, image, text, unstructured_text, audio, or 'unknown'")
    confidence: float = Field(description="Score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief justification for the classification.")

ROUTER_SYSTEM_PROMPT = """
You are a Senior Data Science Architect. 
Your goal is to categorize a user's technical objective into a specific `task_type` and `data_modality`.

### GUIDELINES:
1. **Binary Classification**: Predict a choice between two outcomes.
2. **Thematic Analysis**: Extract human-meaningful patterns/themes from text and analyze distributions or metadata correlations.
3. **EDA**: General exploratory analysis.
4. **Tabular**: Structured data (CSV, SQL).
5. **Unstructured Text**: Raw text, reviews, feedback, documents.

### EXAMPLES:
- "Extract themes from reviews" -> {task_type: 'thematic_analysis', data_modality: 'unstructured_text'}
- "Predict Titanic survival" -> {task_type: 'binary_classification', data_modality: 'tabular'}

### FORMATTING RULE:
You MUST return a valid JSON object matching the requested schema. 
Do NOT include any metadata, schema definitions, or 'properties' wrappers.
"""

class DataScienceRouter(BaseAgent[RouterInput, RouterOutput]):
    def __init__(self, model: str = "claude-haiku-4.5"):
        # Use a cheaper/faster model for classification as per Opus suggestion
        super().__init__(
            name="Router",
            model=model,
            system_prompt=ROUTER_SYSTEM_PROMPT,
            output_schema=RouterOutput
        )

    async def run(self, input_data: RouterInput) -> Any:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"OBJECTIVE: {input_data.objective}"}
        ]
        
        from gads.core.llm import get_structured_completion
        content = await get_structured_completion(
            model=self.model,
            response_model=self.output_schema,
            messages=messages
        )
        
        from gads.agents.base import AgentResponse
        return AgentResponse(content=content, model_used=self.model)
