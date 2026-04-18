from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent
import json

class PlannerTask(BaseModel):
    """Simplified task model for the Planner to output."""
    description: str
    assigned_to: str
    postcondition: Dict[str, Any] = Field(
        description="Structural contract for success. E.g., {'output_type': 'dataframe', 'min_rows': 1, 'required_columns': ['name']}"
    )

class PlannerInput(BaseModel):
    objective: str
    available_models_hierarchy: Dict[str, Any]

class PlannerOutput(BaseModel):
    steps: List[PlannerTask] = Field(description="A list of discrete steps with postcondition contracts.")

PLANNER_SYSTEM_PROMPT = """
You are the Lead Project Manager of a high-end Data Science team. 
Your goal is to decompose a user's request into a list of tasks, delegate each to the **LOWEST FEASIBLE MODEL TIER**, and define a **POSTCONDITION CONTRACT** for every step.

### 1. CAPABILITY RUBRIC
Score every task across these 4 dimensions (Low, Med, High):
- **Reasoning Depth**: Novel decomposition, multi-hop logic, or architectural decisions.
- **Context Breadth**: Need for 100K+ tokens of context or very long memory.
- **Output Fidelity**: Zero-tolerance for syntax errors (e.g., complex pandas code).
- **Domain Specificity**: Obscure Python libraries or deep mathematical expertise.

### 2. SELECTION RULES
- IF any dimension is **HIGH** → Delegate to **Tier 1**.
- ELIF two or more dimensions are **MEDIUM** → Delegate to **Tier 2**.
- ELIF all dimensions are **LOW** but structured output is needed → Delegate to **Tier 3**.
- ELIF the task is purely mechanical (regex, format conversion, boilerplate) → Delegate to **Tier 4**.

### 3. HIERARCHY ANCHORS (PREFER LOCAL)
- **Tier 4 (Local)**: "Extract emails", "Format this list as JSON", "Lower-case all strings".
- **Tier 3 (Haiku/Lite)**: "Summarize this 1-page text", "Categorize these 10 items".
- **Tier 2 (Sonnet/Flash)**: "Clean this messy CSV with pandas", "Create a Seaborn plot".
- **Tier 1 (Pro/Opus)**: "Design the entire pipeline", "Debug a multi-file race condition".

### 4. OUTPUT FORMAT
You MUST provide a list of steps. For each task:
- Set `assigned_to` to the EXACT verbatim model ID from the 'models' list in the chosen Tier (e.g. 'claude-haiku-4.5', NOT 'T3').
- You MUST select a model that is explicitly listed in the hierarchy below.
- define a structural contract for every task to detect "silent failures."
Example contracts:
- DataFrame: {{"output_type": "dataframe", "min_rows": 5, "required_columns": ["x", "y"]}}
- List: {{"output_type": "list", "min_items": 1}}
- Text: {{"output_type": "string", "contains": "keyword"}}

## AVAILABLE_MODELS_HIERARCHY:
{hierarchy_json}
"""

class DataSciencePlanner(BaseAgent[PlannerInput, PlannerOutput]):
    def __init__(self, model: str = "claude-opus-4.7"):
        super().__init__(
            name="Planner",
            model=model,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            output_schema=PlannerOutput
        )

    async def run(self, input_data: PlannerInput) -> Any:
        hierarchy_str = json.dumps(input_data.available_models_hierarchy, indent=2)
        formatted_prompt = self.system_prompt.format(hierarchy_json=hierarchy_str)
        
        messages = [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": f"USER OBJECTIVE: {input_data.objective}"}
        ]
        
        from gads.core.llm import get_structured_completion
        content = await get_structured_completion(
            model=self.model,
            response_model=self.output_schema,
            messages=messages
        )
        
        from gads.agents.base import AgentResponse
        return AgentResponse(content=content, model_used=self.model)
