import os
import instructor
from litellm import acompletion
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-1234")

# Configure instructor with LiteLLM async completion
client = instructor.patch(create=acompletion, mode=instructor.Mode.JSON)

async def get_structured_completion(model: str, response_model, messages: list, **kwargs):
    """
    Wrapper around instructor/litellm to get validated Pydantic objects.
    Routes through the local proxy.
    """
    # Ensure base_url and api_key are passed if not already in kwargs
    if "base_url" not in kwargs:
        kwargs["base_url"] = LITELLM_BASE_URL
    if "api_key" not in kwargs:
        kwargs["api_key"] = LITELLM_MASTER_KEY
        
    return await client(
        model=model,
        response_model=response_model,
        messages=messages,
        custom_llm_provider="openai", # Force openai proxy compatibility
        **kwargs
    )
