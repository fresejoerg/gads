from typing import List
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent
from gads.core.models import Task

class PlannerInput(BaseModel):
    objective: str

class PlannerOutput(BaseModel):
    steps: List[Task] = Field(description="A list of discrete steps to complete the objective.")

PLANNER_SYSTEM_PROMPT = """
You are a precise and literal Project Manager. 
Your goal is to decompose a user's request into the MINIMUM number of executable steps.

STRICT RULES:
1. NO OVER-ENGINEERING: Do NOT add steps for extraction, analysis, or visualization UNLESS the user explicitly asked for them.
2. LITERAL ADHERENCE: If the user only asks for a 'dataframe' and a 'story', then ONLY plan a 'dataframe' step and a 'story' step.
3. SUBJECT MATTER: Adhere strictly to the user's topic (e.g., flowers, people, cities).
4. AGENT ASSIGNMENT:
   - Use 'CodeGenerator' for any coding/dataframe/plotting tasks.
   - Use 'Synthesizer' for the final narrative story or summary.
   - Use 'NLPExtractor' ONLY if the user explicitly asks to 'extract' specific entities.
"""

class DataSciencePlanner(BaseAgent[PlannerInput, PlannerOutput]):
    def __init__(self, model: str = "claude-opus-4.7"):
        super().__init__(
            name="Planner",
            model=model,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            output_schema=PlannerOutput
        )
 Here is the updated code:
from typing import List
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent
from gads.core.models import Task

class PlannerInput(BaseModel):
    objective: str

class PlannerOutput(BaseModel):
    steps: List[Task] = Field(description="A list of discrete steps to complete the objective.")

PLANNER_SYSTEM_PROMPT = """
You are a precise and literal Project Manager. 
Your goal is to decompose a user's request into the MINIMUM number of executable steps.

STRICT RULES:
1. NO OVER-ENGINEERING: Do NOT add steps for extraction, analysis, or visualization UNLESS the user explicitly asked for them.
2. LITERAL ADHERENCE: If the user only asks for a 'dataframe' and a 'story', then ONLY plan a 'dataframe' step and a 'story' step.
3. SUBJECT MATTER: Adhere strictly to the user's topic (e.g., flowers, people, cities).
4. AGENT ASSIGNMENT:
   - Use 'CodeGenerator' for any coding/dataframe/plotting tasks.
   - Use 'Synthesizer' for the final narrative story or summary.
   - Use 'NLPExtractor' ONLY if the user explicitly asks to 'extract' specific entities.
"""

class DataSciencePlanner(BaseAgent[PlannerInput, PlannerOutput]):
    def __init__(self, model: str = "claude-opus-4.7"):
        super().__init__(
            name="Planner",
            model=model,
            system_prompt=PLANNER_SYSTEM_PROMPT,
            output_schema=PlannerOutput
        )
