from typing import List, Optional, Any
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent

class SynthesizerInput(BaseModel):
    objective: str
    context_artifacts: str # Serialized summary of what happened
    existing_narrative: Optional[str] = None
    existing_takeaways: Optional[List[str]] = None

class SynthesizerOutput(BaseModel):
    narrative: str = Field(description="The final human-friendly response or story.")
    key_takeaways: List[str] = Field(description="Bullet points of the most important findings.")

SYNTHESIZER_SYSTEM_PROMPT = """
You are the Lead Data Scientist and Storyteller.
Your goal is to take the raw results from various sub-agents (code outputs, data extractions, visualizations) 
and synthesize them into a compelling, easy-to-understand narrative for the user.

RULES:
1. Be professional but engaging.
2. Focus on the 'WHY' and 'SO WHAT' of the data.
3. Refer to specific findings or visualizations mentioned in the context using 'Figure N' designations (e.g., 'As seen in Figure 1...').
4. GROUNDING: Refer to actual artifact filenames when appropriate (e.g., 'the correlation matrix saved in Figure 2 (correlation.html)...').
5. AMENDMENTS: If the user is asking a follow-up question, you will receive the EXISTING NARRATIVE and EXISTING TAKEAWAYS. You MUST seamlessly integrate the new findings into the existing story. Expand the report. DO NOT delete or ignore the previous findings.
6. If there were errors, explain them simply.
7. CRITICAL: You MUST return a valid JSON object containing ONLY the 'narrative' and 'key_takeaways' fields. 
   Do NOT include any metadata, schema definitions, or 'properties' wrappers.

## PREVIOUS REPORT STATE
{previous_state}
"""

class SynthesizerAgent(BaseAgent[SynthesizerInput, SynthesizerOutput]):
    def __init__(self, model: str = "claude-opus-4.7"):
        super().__init__(
            name="Synthesizer",
            model=model,
            system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
            output_schema=SynthesizerOutput
        )

    async def run(self, input_data: SynthesizerInput, **kwargs) -> Any:
        prev_state = "No previous report."
        if input_data.existing_narrative:
            prev_state = f"--- EXISTING NARRATIVE ---\n{input_data.existing_narrative}\n\n--- EXISTING TAKEAWAYS ---\n" + "\n".join([f"- {t}" for t in (input_data.existing_takeaways or [])])
            
        formatted_prompt = self.system_prompt.format(previous_state=prev_state)
        
        messages = [
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": f"USER OBJECTIVE: {input_data.objective}\n\nARTIFACTS AND OUTPUTS:\n{input_data.context_artifacts}"}
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
