from typing import List
from pydantic import BaseModel, Field
from gads.agents.base import BaseAgent

class Entity(BaseModel):
    name: str
    category: str = Field(description="e.g., PERSON, ORG, GPE, DATE")
    context: str = Field(description="The sentence or phrase where this entity was found.")

class NLPExtractorOutput(BaseModel):
    entities: List[Entity]

class NLPExtractorInput(BaseModel):
    text: str

NLP_SYSTEM_PROMPT = """
You are a precise NLP Extraction specialist. 
Your task is to identify key entities in the provided text and categorize them.
Be extremely literal and only extract what is present in the text.
"""

class NLPExtractorAgent(BaseAgent[NLPExtractorInput, NLPExtractorOutput]):
    def __init__(self, model: str = "local_model"):
        super().__init__(
            name="NLPExtractor",
            model=model,
            system_prompt=NLP_SYSTEM_PROMPT,
            output_schema=NLPExtractorOutput
        )
