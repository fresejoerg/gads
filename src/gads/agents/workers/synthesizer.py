from typing import List, Optional
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent

class SynthesizerInput(BaseModel):
    objective: str
    context_artifacts: str # Serialized summary of what happened

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
5. If there were errors, explain them simply.
6. CRITICAL: You MUST return a valid JSON object containing ONLY the 'narrative' and 'key_takeaways' fields. 
   Do NOT include any metadata, schema definitions, or 'properties' wrappers.
"""

class SynthesizerAgent(BaseAgent[SynthesizerInput, SynthesizerOutput]):
    def __init__(self, model: str = "claude-opus-4.7"):
        super().__init__(
            name="Synthesizer",
            model=model,
            system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
            output_schema=SynthesizerOutput
        )
