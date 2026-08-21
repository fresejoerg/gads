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

# Local-execution retry-exhaustion fallback (approach_docs/019). When a task exhausts all
# retries, optionally rescue the node instead of failing → replanning:
#   none              — fail the task (default; the boundary stays visible)
#   native            — invoke the node's fallback_native deterministically (no replan)
#   cloud             — escalate the one task to a cloud model once (gated exception to
#                       "local never escalates" — post-exhaustion, opt-in only)
#   native_then_cloud — native if the node has one, else cloud
VALID_LOCAL_FALLBACKS = ("none", "native", "cloud", "native_then_cloud")

def _initial_fallback() -> str:
    fb = os.getenv("GADS_LOCAL_FALLBACK", "").strip().lower()
    return fb if fb in VALID_LOCAL_FALLBACKS else "none"

_LOCAL_FALLBACK = _initial_fallback()

def set_local_fallback(mode: str):
    global _LOCAL_FALLBACK
    mode = (mode or "").strip().lower()
    if mode not in VALID_LOCAL_FALLBACKS:
        raise ValueError(f"Unknown local_fallback '{mode}'. Valid: {VALID_LOCAL_FALLBACKS}")
    _LOCAL_FALLBACK = mode
    print(f"  [Registry] Local fallback set to '{_LOCAL_FALLBACK}'", flush=True)

def get_local_fallback() -> str:
    return _LOCAL_FALLBACK


# ——— RUN MODE: what a run is FOR ———————————————————————————————————————————————
# "research"   — model-first. Every node with a native still gets attempted by the model,
#                and the native is only a post-exhaustion safety net. This is what makes
#                pass@model meaningful: if the native ran first there would be nothing to
#                measure. Costs tokens and carries the failed-attempt risks (state drift,
#                kernel poisoning) on nodes whose answer is invariant anyway.
# "production" — native-first. A node that declares a native uses it directly and the model
#                is never asked. Deterministic, reproducible and cheaper on exactly the nodes
#                where a model adds nothing (one defensible answer), while judgment nodes
#                that declare no native are still model-generated.
#
# The distinction is deliberate: the same node can be worth measuring in a benchmark and
# worth short-circuiting in a real analysis. See approach_docs/019.
VALID_RUN_MODES = ("research", "production")


def _initial_run_mode() -> str:
    rm = os.getenv("GADS_RUN_MODE", "").strip().lower()
    return rm if rm in VALID_RUN_MODES else "research"


_RUN_MODE = _initial_run_mode()


def set_run_mode(mode: str):
    global _RUN_MODE
    mode = (mode or "").strip().lower()
    if mode not in VALID_RUN_MODES:
        raise ValueError(f"Unknown run_mode '{mode}'. Valid: {VALID_RUN_MODES}")
    _RUN_MODE = mode
    print(f"  [Registry] Run mode set to '{_RUN_MODE}'", flush=True)


def get_run_mode() -> str:
    return _RUN_MODE

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

# Hardcoded rules for mapping model names to Tiers. Latest generation per provider,
# placed by capability; Gemini/Local are kept at index 0 so they are the primary
# (cheapest-first) choice. Currency per provider (2026-07-22):
#   OpenAI    — GPT-5.6 generation only (sol/terra/luna); 5.4/5.5 retired.
#   Anthropic — Claude 5 line (Fable 5, Sonnet 5) + Opus 4.8 (T1) + Haiku 4.5 (T3).
#   Gemini    — 3.6-flash (T2), 3.5-flash-lite (T3); Pro is 3.1-pro-preview (latest Pro).
#   Kimi      — k3 flagship (T1) + k2.7-code / k2.7-code-highspeed (k2.x/2.6 sunset).
# The live hierarchy intersects this with LiteLLM's served models, so any newer ID
# must also be added to the MyLocalStack gateway to become reachable; until then
# intra-tier fallback covers interim 404s and the mapping self-heals.
TIER_MAPPING = {
    "T1": ["gemini-3.1-pro-preview", "claude-opus-4.8", "claude-fable-5", "gpt-5.6-sol", "kimi-k3"],
    "T2": ["gemini-3.7-flash", "claude-sonnet-5", "gpt-5.6-terra", "kimi-k2.7-code"],
    "T3": ["gemini-3.5-flash-lite", "claude-haiku-4.5", "gpt-5.6-luna", "kimi-k2.7-code-highspeed"],
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

async def get_model_hierarchy(force_cloud: bool = False) -> Dict[str, Any]:
    """Fetches models from LiteLLM and organizes them into capability tiers.

    `force_cloud=True` builds the real cloud tiers even in local mode — used ONLY by the
    opt-in local retry-exhaustion cloud fallback (approach_docs/019), which needs a cloud
    model to escalate to after the local model has exhausted its retries.
    """
    if get_routing_mode() == "local" and not force_cloud:
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
