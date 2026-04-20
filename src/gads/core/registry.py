import httpx
import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()

LITELLM_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-1234")
GADS_LOCAL_ONLY = os.getenv("GADS_LOCAL_ONLY", "false").lower() == "true"

# Hardcoded rules for mapping model names to Tiers
# Gemini/Local are always index 0 to ensure they are the primary choice.
TIER_MAPPING = {
    "T1": ["gemini-3.1-pro-preview", "claude-opus-4.7"],
    "T2": ["gemini-3-flash-preview", "claude-sonnet-4.6"],
    "T3": ["gemini-3.1-flash-lite-preview", "claude-haiku-4.5"],
    "T4": ["local_model"]
}

TIER_ORDER = ["T4", "T3", "T2", "T1"]

TIER_DESCRIPTIONS = {
    "T1": "Architect tier. Use for complex reasoning, multi-step planning, and novel problem solving.",
    "T2": "Coder tier. Use for complex Python logic, data visualization, and statistical modeling.",
    "T3": "Worker tier. Use for rapid text processing, summarization, and formatting.",
    "T4": "Local tier. Use for simple, mechanical tasks like regex extraction or basic boilerplate."
}

async def get_model_hierarchy() -> Dict[str, Any]:
    """Fetches models from LiteLLM and organizes them into capability tiers."""
    if GADS_LOCAL_ONLY:
        print("  [Registry] 🏠 LOCAL ONLY MODE ENABLED. Overriding all tiers to 'local_model'.")
        return {
            tier: {
                "description": TIER_DESCRIPTIONS[tier],
                "models": ["local_model"]
            } for tier in TIER_ORDER
        }

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
                # Preservation of order: if gemini is in TIER_MAPPING, it will be first in matching_models
                matching_models = [m for m in keywords if m in available_models]
                if matching_models:
                    hierarchy[tier] = {
                        "description": TIER_DESCRIPTIONS[tier],
                        "models": matching_models
                    }
            
            return hierarchy
    except Exception as e:
        print(f"Registry Error: {e}")
        return {"T4": {"description": "Fallback", "models": ["local_model"]}}

def get_next_model_dynamic(current_model: str, hierarchy: Dict[str, Any]) -> Optional[str]:
    """
    Deterministic Escalation Logic:
    1. Try the next model in the SAME tier (e.g. Gemini -> Claude).
    2. If no more models in tier, move to the first model of the NEXT tier.
    """
    current_tier = None
    current_tier_models = []
    
    for tier, data in hierarchy.items():
        if current_model in data["models"]:
            current_tier = tier
            current_tier_models = data["models"]
            break
            
    if not current_tier:
        return hierarchy.get("T4", {}).get("models", [None])[0]

    # 1. Try to find a fallback model in the SAME tier (Claude fallback)
    try:
        current_idx = current_tier_models.index(current_model)
        if current_idx + 1 < len(current_tier_models):
            return current_tier_models[current_idx + 1]
    except ValueError:
        pass

    # 2. No more in current tier? Move to first model of NEXT tier
    try:
        tier_idx = TIER_ORDER.index(current_tier)
        if tier_idx + 1 < len(TIER_ORDER):
            next_tier = TIER_ORDER[tier_idx + 1]
            if next_tier in hierarchy:
                # Return the first model in the next tier (which is Gemini)
                return hierarchy[next_tier]["models"][0]
    except ValueError:
        pass
        
    return None
