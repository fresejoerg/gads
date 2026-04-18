import os
import instructor
from litellm import completion, acompletion
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-1234")

# Configure instructor with LiteLLM async completion
# Use instructor.patch for explicit control over the completion function
# MD_JSON is most compatible with local models
client = instructor.patch(create=acompletion, mode=instructor.Mode.MD_JSON)

async def get_structured_completion(model: str, response_model, messages: list, **kwargs):
    """
    Wrapper around instructor/litellm to get validated Pydantic objects.
    Routes through the local proxy.
    """
    return await client(
        model=model,
        response_model=response_model,
        messages=messages,
        base_url=LITELLM_BASE_URL,
        api_key=LITELLM_MASTER_KEY,
        custom_llm_provider="openai",
        **kwargs
    )
