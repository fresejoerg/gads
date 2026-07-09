import httpx
import os
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

load_dotenv()

LITELLM_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-1234")

# ---------------------------------------------------------------------------
# ROUTING MODES — how workflow stages map to models.
#
#   cloud        — tiered cloud hierarchy (T3→T2→T1) with the escalation ladder.
#   local        — every stage on local_model. No escalation (T4 isolation).
#   hybrid       — plan construction (SpecDrafter/Router/Planner/PlanCritique)
#                  and report writing (Synthesizer/Critique) on cloud tiers;
#                  everything else (execution tasks, CompletenessVerifier) on
#                  local_model. Cloud stages keep the ladder; local tasks never
#                  escalate to cloud (isolation mandate unchanged).
#   cloud_pinned — one operator-chosen cloud model for EVERY stage. No
#                  escalation ladder by design: a failure retries on the same
#                  model (inner Coder retries) or fails.
#
# `local_only` is kept as a legacy alias: true ≡ mode "local", false ≡ "cloud".
# ---------------------------------------------------------------------------
VALID_ROUTING_MODES = ("cloud", "local", "hybrid", "cloud_pinned")

def _initial_mode() -> str:
    mode = os.getenv("GADS_ROUTING_MODE", "").strip().lower()
    if mode in VALID_ROUTING_MODES:
        return mode
    return "local" if os.getenv("GADS_LOCAL_ONLY", "false").lower() == "true" else "cloud"

_ROUTING_MODE = _initial_mode()
_PINNED_MODEL: Optional[str] = os.getenv("GADS_PINNED_MODEL", "").strip() or None
_RANDOM_ROUTING = False

# Stages that stay on cloud tiers in hybrid mode. Everything not listed here
# (Coder execution tasks, CompletenessVerifier, any future stage) goes local.
HYBRID_CLOUD_STAGES = {"SpecDrafter", "Router", "Planner", "PlanCritique", "Synthesizer", "Critique"}

def set_routing_mode(mode: str, pinned_model: Optional[str] = None):
    global _ROUTING_MODE, _PINNED_MODEL
    mode = (mode or "").strip().lower()
    if mode not in VALID_ROUTING_MODES:
        raise ValueError(f"Unknown routing mode '{mode}'. Valid: {VALID_ROUTING_MODES}")
    if mode == "cloud_pinned":
        if not pinned_model or pinned_model == "local_model":
            raise ValueError("cloud_pinned mode requires a cloud pinned_model (use mode 'local' for local_model).")
        _PINNED_MODEL = pinned_model
    elif pinned_model:
        _PINNED_MODEL = pinned_model
    _ROUTING_MODE = mode
    print(f"  [Registry] Routing mode set to '{_ROUTING_MODE}'"
          + (f" (pinned: {_PINNED_MODEL})" if _ROUTING_MODE == "cloud_pinned" else ""), flush=True)

def get_routing_mode() -> str:
    return _ROUTING_MODE

def get_pinned_model() -> Optional[str]:
    return _PINNED_MODEL

def set_local_only(enabled: bool):
    """Legacy alias for old clients/UI: maps onto routing modes."""
    set_routing_mode("local" if enabled else "cloud")

def get_local_only() -> bool:
    """Legacy alias: true only in fully-local mode."""
    return _ROUTING_MODE == "local"

def set_random_routing(enabled: bool):
    global _RANDOM_ROUTING
    _RANDOM_ROUTING = enabled

def get_random_routing() -> bool:
    return _RANDOM_ROUTING

def resolve_stage_model(stage: str, tier_default: str) -> str:
    """Central mode-aware model choice for a workflow stage.

    `tier_default` is what the tier/hierarchy logic picked; this applies the
    routing-mode override on top. Every stage and every execution-task
    assignment must route through here so a mode change covers the whole
    pipeline.
    """
    if _ROUTING_MODE == "local":
        return "local_model"
    if _ROUTING_MODE == "cloud_pinned" and _PINNED_MODEL:
        return _PINNED_MODEL
    if _ROUTING_MODE == "hybrid" and stage not in HYBRID_CLOUD_STAGES:
        return "local_model"
    return tier_default

# Hardcoded rules for mapping model names to Tiers
# Gemini/Local are always index 0 to ensure they are the primary choice.
TIER_MAPPING = {
    "T1": ["gemini-3.1-pro-preview", "claude-opus-4.8", "claude-fable-5", "gpt-5.5", "kimi-k2.6"],
    "T2": ["gemini-3.5-flash", "claude-sonnet-4.6", "gpt-5.4", "kimi-k2.6"],
    "T3": ["gemini-3.1-flash-lite-preview", "claude-haiku-4.5", "gpt-5.4-mini", "kimi-k2.6"],
    "T4": ["local_model"]
}

# Escalation path: only includes cloud tiers. 
# local_model (T4) is isolated and never escalates to cloud tiers.
TIER_ORDER = ["T3", "T2", "T1"]

TIER_DESCRIPTIONS = {
    "T1": "Architect tier. Use for complex reasoning, multi-step planning, and novel problem solving.",
    "T2": "Coder tier. Use for complex Python logic, data visualization, and statistical modeling.",
    "T3": "Worker tier. Use for rapid text processing, summarization, and formatting.",
    "T4": "Local tier. Use for simple, mechanical tasks. NEVER escalates to cloud tiers."
}

async def get_available_models() -> List[str]:
    """Raw model list from LiteLLM (for the UI's pinned-model picker etc.)."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{LITELLM_URL}/models",
            headers={"Authorization": f"Bearer {LITELLM_KEY}"}
        )
        resp.raise_for_status()
        return [m["id"] for m in resp.json()["data"]]

async def get_model_hierarchy() -> Dict[str, Any]:
    """Fetches models from LiteLLM and organizes them into capability tiers."""
    if get_routing_mode() == "local":
        print("  [Registry] 🏠 LOCAL ONLY MODE ENABLED. Overriding all tiers to 'local_model'.")
        # In local mode, everything is pinned to the local model.
        # No escalation path exists because all tiers lead to the same model.
        return {
            tier: {
                "description": TIER_DESCRIPTIONS.get(tier, "Local execution only."),
                "models": ["local_model"]
            } for tier in ["T4", "T3", "T2", "T1"]
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

            if get_routing_mode() == "cloud_pinned":
                pinned = get_pinned_model()
                if pinned not in available_models:
                    raise RuntimeError(
                        f"cloud_pinned mode: pinned model '{pinned}' is not served by "
                        f"LiteLLM (available: {available_models}). Refusing to start."
                    )
                print(f"  [Registry] 📌 CLOUD PINNED MODE. All tiers → '{pinned}'. No escalation ladder.", flush=True)
                return {
                    tier: {
                        "description": TIER_DESCRIPTIONS.get(tier, ""),
                        "models": [pinned]
                    } for tier in ["T3", "T2", "T1"]
                }

            hierarchy = {}
            for tier, keywords in TIER_MAPPING.items():
                # HARD MANDATE: local_model (T4) is reachable ONLY in local_only mode.
                # Cloud runs must never plan onto or fall back to the local model, so
                # T4 is excluded from the hierarchy the Planner/sanitizer/enforcer see.
                if tier == "T4":
                    continue
                # Preservation of order: if gemini is in TIER_MAPPING, it will be first in matching_models
                matching_models = [m for m in keywords if m in available_models]
                if matching_models:
                    if get_random_routing():
                        import random
                        random.shuffle(matching_models)

                    hierarchy[tier] = {
                        "description": TIER_DESCRIPTIONS.get(tier, ""),
                        "models": matching_models
                    }

            if not hierarchy:
                raise RuntimeError(
                    "No cloud models available from LiteLLM. Refusing to fall back to "
                    "local_model in cloud mode — enable local_only to run locally."
                )
            return hierarchy
    except Exception as e:
        # Fail loudly: silently degrading a cloud run onto local_model violates the
        # local-isolation mandate. The workflow marks itself failed (trace labeled
        # outcome:failed) instead of quietly running on the local model.
        print(f"Registry Error: {e}")
        raise

def get_next_model_dynamic(current_model: str, hierarchy: Dict[str, Any], aggressive: bool = False) -> Optional[str]:
    """
    Escalation Logic:
    1. If LOCAL ONLY MODE is active: No escalation is permitted.
    2. If current model is 'local_model': No escalation to cloud models is permitted.
       (This also covers hybrid mode's local execution tasks.)
    3. If CLOUD PINNED mode: No escalation ladder by design — retry-or-fail on the pinned model.
    4. If cloud/hybrid-cloud: Try other models in same tier, then move to better tiers (T3 -> T2 -> T1).
    """
    # HARD MANDATE: No escalation in local mode, FROM local model, or in pinned mode
    if get_local_only() or current_model == "local_model" or get_routing_mode() == "cloud_pinned":
        return None

    import random
    
    current_tier = None
    current_tier_models = []
    
    for tier, data in hierarchy.items():
        if current_model in data["models"]:
            current_tier = tier
            current_tier_models = data["models"]
            break
            
    if not current_tier:
        # Fallback to T3 if current model is unknown
        return hierarchy.get("T3", {}).get("models", [None])[0]

    # 1. Non-aggressive: Try to find a fallback model in the SAME tier
    if not aggressive:
        other_models_in_tier = [m for m in current_tier_models if m != current_model]
        if other_models_in_tier:
            return random.choice(other_models_in_tier)

    # 2. Aggressive OR no more in current tier? Move to first model of NEXT tier
    try:
        # TIER_ORDER is ["T3", "T2", "T1"]
        if current_tier in TIER_ORDER:
            tier_idx = TIER_ORDER.index(current_tier)
            if tier_idx + 1 < len(TIER_ORDER):
                next_tier = TIER_ORDER[tier_idx + 1]
                # Skip tiers that aren't in the actual hierarchy
                while next_tier not in hierarchy and tier_idx + 1 < len(TIER_ORDER):
                    tier_idx += 1
                    next_tier = TIER_ORDER[tier_idx + 1]
                    
                if next_tier in hierarchy:
                    return hierarchy[next_tier]["models"][0]
    except (ValueError, IndexError):
        pass
        
    return None
