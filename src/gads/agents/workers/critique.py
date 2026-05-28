from typing import List, Optional, Any
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent
from gads.core.prompts import prompt_registry

class CritiqueInput(BaseModel):
    objective: str
    context_artifacts: str # Serialized summary of what happened
    synthesis_narrative: str
    synthesis_takeaways: List[str]
    dashboard_html: str

class CritiqueOutput(BaseModel):
    is_approved: bool = Field(description="True if the synthesis is consistent, logical, and follows quality standards.")
    critique_feedback: str = Field(description="Actionable feedback if not approved; 'Looks good' if approved.")
    redundant_artifacts: List[str] = Field(default_factory=list, description="List of filenames of redundant or low-quality plots to remove.")

class CritiqueAgent(BaseAgent[CritiqueInput, CritiqueOutput]):
    def __init__(self, model: str = "local_model"):
        super().__init__(
            name="Critique",
            model=model,
            system_prompt=prompt_registry.get_prompt("Critique"),
            output_schema=CritiqueOutput
        )

    async def run(self, input_data: CritiqueInput, **kwargs) -> Any:
        # Refresh prompt from registry
        # Set the prompt for the Pydantic AI agent
        self.agent._system_prompts = (self.system_prompt,)

        user_content = (
            f"USER OBJECTIVE: {input_data.objective}\n\n"
            f"AGENT SYNTHESIS:\nNarrative: {input_data.synthesis_narrative}\n"
            f"Takeaways: {input_data.synthesis_takeaways}\n\n"
            f"TASK OUTPUTS AND RAW ARTIFACTS:\n{input_data.context_artifacts}\n\n"
            f"FINAL DASHBOARD HTML (Draft):\n{input_data.dashboard_html}"
        )

        # Use super().run
        return await super().run(
            user_content,
            **kwargs
        )

