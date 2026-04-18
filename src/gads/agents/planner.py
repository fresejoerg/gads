from typing import List
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent
from gads.core.state import Task

class PlannerInput(BaseModel):
    objective: str

class PlannerOutput(BaseModel):
    steps: List[Task] = Field(description="A list of discrete steps to complete the objective.")

PLANNER_SYSTEM_PROMPT = """
You are the Project Manager of a high-end Data Science team. 
Your goal is to take a user's request and break it down into a clear, executable plan.
Available sub-agents:
1. NLPExtractor: For extracting entities or structured data from text.
2. DataViz: For creating charts and visualizations.
3. CodeRunner: For generic data processing or model training.

Assign each task to the most appropriate sub-agent using their exact name.
Assign 'local_model' to simple extraction tasks and 'claude-3.5-sonnet' or 'gemini-3.1' for complex planning or viz tasks.
"""

class DataSciencePlanner(BaseAgent[PlannerInput, PlannerOutput]):
    def __init__(self, model: str = "claude-opus-4.7"):
        super().__init__(
            name="Planner",
            model=model,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            output_schema=PlannerOutput
        )
