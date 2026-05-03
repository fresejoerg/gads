from typing import List
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent
from gads.core.prompts import prompt_registry

class Entity(BaseModel):
    name: str
    category: str = Field(description="e.g., PERSON, ORG, GPE, DATE")
    context: str = Field(description="The sentence or phrase where this entity was found.")

class NLPExtractorOutput(BaseModel):
    entities: List[Entity]

class NLPExtractorInput(BaseModel):
    text: str

class NLPExtractorAgent(BaseAgent[NLPExtractorInput, NLPExtractorOutput]):
    def __init__(self, model: str = "local_model"):
        super().__init__(
            name="NLPExtractor",
            model=model,
            system_prompt=prompt_registry.get_prompt("NLPExtractor"),
            output_schema=NLPExtractorOutput
        )
