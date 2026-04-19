from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent

class RouterInput(BaseModel):
    objective: str

class RouterOutput(BaseModel):
    task_type: str = Field(description="One of: binary_classification, regression, time_series, eda, clustering, or 'unknown'")
    data_modality: str = Field(description="One of: tabular, image, text, audio, or 'unknown'")
    confidence: float = Field(description="Score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief justification for the classification.")

ROUTER_SYSTEM_PROMPT = """
You are a Senior Data Science Architect. 
Your goal is to categorize a user's technical objective into a specific `task_type` and `data_modality`.

### GUIDELINES:
1. **Binary Classification**: Use if the user wants to predict a choice between two outcomes (e.g., churn, survival, fraudulent/legitimate).
2. **Regression**: Use if the target is a continuous numerical value (e.g., price, temperature, sales count).
3. **Tabular**: Use if the data is structured (CSV, SQL, Excel, DataFrames).
4. **Unknown**: If the request is non-technical or doesn't fit a pattern, set both to 'unknown'.

### EXAMPLES:
- "Predict Titanic survival" -> {task_type: 'binary_classification', data_modality: 'tabular'}
- "Forecast house prices" -> {task_type: 'regression', data_modality: 'tabular'}
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
