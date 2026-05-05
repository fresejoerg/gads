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
    def __init__(self, model: str = "claude-sonnet-4.6"):
        super().__init__(
            name="Critique",
            model=model,
            system_prompt=prompt_registry.get_prompt("Critique"),
            output_schema=CritiqueOutput
        )

    async def run(self, input_data: CritiqueInput, **kwargs) -> Any:
        # Refresh prompt from registry
        self.system_prompt = prompt_registry.get_prompt(self.name)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"USER OBJECTIVE: {input_data.objective}\n\nAGENT SYNTHESIS:\nNarrative: {input_data.synthesis_narrative}\nTakeaways: {input_data.synthesis_takeaways}\n\nTASK OUTPUTS AND RAW ARTIFACTS:\n{input_data.context_artifacts}\n\nFINAL DASHBOARD HTML (Draft):\n{input_data.dashboard_html}"}
        ]
        
        from gads.core.llm import get_structured_completion
        content = await get_structured_completion(
            model=self.model,
            response_model=self.output_schema,
            messages=messages,
            **kwargs
        )
        
        from gads.agents.base import AgentResponse
        return AgentResponse(content=content, model_used=self.model)
