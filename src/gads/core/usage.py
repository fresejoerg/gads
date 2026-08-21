"""Per-task token and cost accounting.

Every LLM call in a run belongs to exactly one task — `llm.trace_context` already carries
the `task_id` so Langfuse can parent the span correctly — so the same context is enough to
attribute tokens and spend to the recipe node that caused them. That is what lets the
dashboard report what each section of the report cost to produce.

Deliberately **cumulative per task**: a node that took six Coder attempts spent six
generations' worth of tokens, and the honest number for that section is the sum. Reporting
a single call's usage would make an expensive, heavily-retried node look cheap — precisely
the node an efficiency-boundary study needs to see.

Two sources of cost, in order of authority:
  1. `response_cost` from LiteLLM's `_hidden_params` — the proxy's own calculation. Present
     on non-streamed calls and used verbatim.
  2. tokens x the per-token prices the proxy publishes at `/model/info`. The streaming path
     returns usage but no cost, and streaming is how every Coder generation runs, so
     without this the most expensive calls in the system would report nothing.

A model the proxy does not price (and `local_model`, which it prices at zero) yields
`cost_usd = 0.0` or `None` rather than a guess; `cost_source` records which path was taken
so a number is never mistaken for something it isn't.

State is in-process and per-run: `server` snapshots it into the task's `result_json` when
the task completes, and the dashboard reads it back from there.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

_BY_TASK: Dict[str, Dict[str, Any]] = {}

# model -> (input_cost_per_token, output_cost_per_token); populated lazily from the proxy.
_PRICES: Dict[str, tuple] = {}
_PRICES_LOADED = False


def _blank() -> Dict[str, Any]:
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "reasoning_tokens": 0, "total_tokens": 0, "cost_usd": 0.0,
            "models": [], "cost_source": None, "unpriced_calls": 0}


async def _ensure_prices() -> None:
    """Fetch per-token pricing once per process. Best-effort: without it the streaming
    path simply reports tokens and no cost, which is degraded but not wrong."""
    global _PRICES_LOADED
    if _PRICES_LOADED:
        return
    _PRICES_LOADED = True          # set first: one failed attempt must not retry per call
    try:
        import httpx
        base = os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1").rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3].rstrip("/")
        key = os.getenv("LITELLM_MASTER_KEY", "sk-1234")
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{base}/model/info",
                                    headers={"Authorization": f"Bearer {key}"})
            resp.raise_for_status()
            rows = resp.json().get("data", [])
        for row in rows:
            name = row.get("model_name")
            info = row.get("model_info") or {}
            i, o = info.get("input_cost_per_token"), info.get("output_cost_per_token")
            if name and i is not None and o is not None:
                _PRICES[name] = (float(i), float(o))
        print(f"  [Usage] Loaded pricing for {len(_PRICES)} models.", flush=True)
    except Exception as e:
        print(f"  [Usage] Pricing unavailable ({e}); cost will be reported only for "
              f"calls where the proxy returns it.", flush=True)


def _current_task_id() -> Optional[str]:
    try:
        from gads.core.llm import trace_context
        ctx = trace_context.get()
        if ctx:
            tid = ctx.get("task_id")
            return str(tid) if tid else None
    except Exception:
        pass
    return None


async def record_call(model: str, usage: Any, response_cost: Optional[float] = None,
                      task_id: Optional[str] = None) -> None:
    """Attribute one completion's tokens and cost to its task. Never raises: accounting
    must not be able to fail a workflow."""
    try:
        tid = task_id or _current_task_id()
        if not tid:
            return

        prompt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        total = int(getattr(usage, "total_tokens", 0) or 0) if usage else (prompt + completion)
        reasoning = 0
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        if details is not None:
            reasoning = int(getattr(details, "reasoning_tokens", 0) or 0)

        cost: Optional[float] = None
        source = None
        if response_cost is not None:
            cost, source = float(response_cost), "proxy"
        else:
            await _ensure_prices()
            price = _PRICES.get(model)
            if price is not None:
                cost = prompt * price[0] + completion * price[1]
                source = "computed"

        rec = _BY_TASK.setdefault(tid, _blank())
        rec["calls"] += 1
        rec["prompt_tokens"] += prompt
        rec["completion_tokens"] += completion
        rec["reasoning_tokens"] += reasoning
        rec["total_tokens"] += total
        if cost is None:
            rec["unpriced_calls"] += 1
        else:
            rec["cost_usd"] = round(rec["cost_usd"] + cost, 8)
        if model and model not in rec["models"]:
            rec["models"].append(model)
        # "mixed" is worth surfacing: it means part of the number is the proxy's and part
        # is ours, so the two should not be compared to the cent.
        if source and rec["cost_source"] not in (None, source):
            rec["cost_source"] = "mixed"
        elif source:
            rec["cost_source"] = source
    except Exception:
        return


def snapshot(task_id: Any) -> Optional[Dict[str, Any]]:
    """The accumulated usage for one task, or None if it made no LLM calls (a node
    completed by a native fallback legitimately has none)."""
    rec = _BY_TASK.get(str(task_id))
    return dict(rec) if rec else None


def discard(task_id: Any) -> None:
    _BY_TASK.pop(str(task_id), None)


def aggregate(snapshots: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """Roll several task snapshots into a run total for the dashboard header."""
    out = _blank()
    sources = set()
    for s in snapshots:
        if not s:
            continue
        for k in ("calls", "prompt_tokens", "completion_tokens", "reasoning_tokens",
                  "total_tokens", "unpriced_calls"):
            out[k] += int(s.get(k) or 0)
        out["cost_usd"] = round(out["cost_usd"] + float(s.get("cost_usd") or 0.0), 8)
        for m in s.get("models") or []:
            if m not in out["models"]:
                out["models"].append(m)
        if s.get("cost_source"):
            sources.add(s["cost_source"])
    out["cost_source"] = ("mixed" if len(sources) > 1 else (sources.pop() if sources else None))
    return out


def format_cost(cost: Optional[float]) -> str:
    """Costs here span five orders of magnitude — a Router call is fractions of a cent, a
    full run is dollars — so a fixed 2dp would render most sections as '$0.00'."""
    if cost is None:
        return "n/a"
    if cost == 0:
        return "$0.00"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def format_tokens(n: Optional[int]) -> str:
    if not n:
        return "0"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)
