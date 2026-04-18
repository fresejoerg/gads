import httpx
import os
from typing import Dict, List, Any
from dotenv import load_dotenv

load_dotenv()

LITELLM_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-1234")

# Hardcoded rules for mapping model names to Tiers
TIER_MAPPING = {
    "T1": ["claude-opus-4.7", "gemini-3.1-pro-preview"],
    "T2": ["claude-sonnet-4.6", "gemini-3.1-flash-preview"],
    "T3": ["claude-haiku-4.5", "gemini-3.1-flash-lite-preview"],
    "T4": ["local_model"]
}

TIER_DESCRIPTIONS = {
    "T1": "Architect tier. Use for complex reasoning, multi-step planning, and novel problem solving.",
    "T2": "Coder tier. Use for complex Python logic, data visualization, and statistical modeling.",
    "T3": "Worker tier. Use for rapid text processing, summarization, and formatting.",
    "T4": "Local tier. Use for simple, mechanical tasks like regex extraction or basic boilerplate."
}

async def get_model_hierarchy() -> Dict[str, Any]:
    """Fetches models from LiteLLM and organizes them into capability tiers."""
    print(f"  [Registry] Fetching models from: {LITELLM_URL}", flush=True)
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{LITELLM_URL}/models",
                headers={"Authorization": f"Bearer {LITELLM_KEY}"}
            )
            resp.raise_for_status()
            available_models = [m["id"] for m in resp.json()["data"]]
            
            hierarchy = {}
            for tier, keywords in TIER_MAPPING.items():
                matching_models = [m for m in available_models if m in keywords]
                if matching_models:
                    hierarchy[tier] = {
                        "description": TIER_DESCRIPTIONS[tier],
                        "models": matching_models
                    }
            
            return hierarchy
    except Exception as e:
        print(f"Registry Error: {e}")
        # Fallback
        return {"T4": {"description": "Fallback", "models": ["local_model"]}}
