import httpx
import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()

LITELLM_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-1234")

# Allow runtime toggling of local only mode
_LOCAL_ONLY = os.getenv("GADS_LOCAL_ONLY", "false").lower() == "true"
_RANDOM_ROUTING = False

def set_local_only(enabled: bool):
    global _LOCAL_ONLY
    _LOCAL_ONLY = enabled

def get_local_only() -> bool:
    return _LOCAL_ONLY

def set_random_routing(enabled: bool):
    global _RANDOM_ROUTING
    _RANDOM_ROUTING = enabled

def get_random_routing() -> bool:
    return _RANDOM_ROUTING

# Hardcoded rules for mapping model names to Tiers
# Gemini/Local are always index 0 to ensure they are the primary choice.
TIER_MAPPING = {
    "T1": ["gemini-3.1-pro-preview", "claude-opus-4.7", "gpt-5.5", "kimi-k2.6", "kimi-k2-thinking", "kimi-k2-thinking-turbo"],
    "T2": ["gemini-3-flash-preview", "claude-sonnet-4.6", "gpt-5.4", "kimi-k2-0905-preview", "kimi-k2-turbo-preview"],
    "T3": ["gemini-3.1-flash-lite-preview", "claude-haiku-4.5", "gpt-5.4-mini", "kimi-k2-0711-preview"],
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
    if get_local_only():
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
                if not get_local_only() and tier == "T4":
                    continue
                
                # Preservation of order: if gemini is in TIER_MAPPING, it will be first in matching_models
                matching_models = [m for m in keywords if m in available_models]
                if matching_models:
                    if get_random_routing():
                        import random
                        random.shuffle(matching_models)
                        
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
    Randomized Escalation Logic:
    1. If other models exist in the SAME tier, randomize between them (excluding current).
    2. If no more models in tier, move to the first model of the NEXT tier.
    """
    import random
    
    current_tier = None
    current_tier_models = []
    
    for tier, data in hierarchy.items():
        if current_model in data["models"]:
            current_tier = tier
            current_tier_models = data["models"]
            break
            
    if not current_tier:
        return hierarchy.get("T4", {}).get("models", [None])[0]

    # 1. Try to find a fallback model in the SAME tier (Randomize remaining)
    other_models_in_tier = [m for m in current_tier_models if m != current_model]
    if other_models_in_tier:
        return random.choice(other_models_in_tier)

    # 2. No more in current tier? Move to first model of NEXT tier
    try:
        tier_idx = TIER_ORDER.index(current_tier)
        if tier_idx + 1 < len(TIER_ORDER):
            next_tier = TIER_ORDER[tier_idx + 1]
            if next_tier in hierarchy:
                # Return the first model in the next tier (which is usually a Gemini)
                return hierarchy[next_tier]["models"][0]
    except ValueError:
        pass
        
    return None
