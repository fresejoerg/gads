"""
GADS Task Error Ledger — persistent, cross-run memory of how recipe steps fail.

Append-only JSONL (deliberately NOT Postgres: the GADS Postgres tables are wiped on a
local-stack restart, which would erase a learning asset that must accumulate across every
invocation). Each failed recipe-step attempt appends one `error` event; a step that
succeeds after having failed appends a `resolution` event. Aggregation happens on read and
powers two things:

  1. a first-attempt "COMMON PITFALLS" prior injected into the Coder (preempt known
     recipe-step mistakes before they happen), and
  2. an async recipe/skill-hardening report (which steps fail most, and how).

Scope: recipe-pinned runs only (a stable recipe_id + a node key derived from the step's
intent). Transient / infra / data-specific errors are filtered out so the ledger reflects
recipe-STRUCTURAL failure modes — the kind worth hardening a recipe or skill against.

Best-effort: every write is wrapped so a ledger failure can never break task execution.
"""
import json
import os
import re
import time
from collections import Counter
from typing import Optional

LEDGER_PATH = os.environ.get("GADS_ERROR_LEDGER", "research/error_ledger.jsonl")

# Error classes that are NOT recipe-structural (infra / transport / resource / workflow),
# so they must not pollute the pitfalls prior or the hardening signal.
_TRANSIENT_ENAMES = {
    "TimeoutError", "ConnectionError", "ChannelError", "Cancelled",
    "KeyboardInterrupt", "MemoryError", "BrokenPipeError", "OSError",
}
_TRANSIENT_PATTERNS = (
    "exceed_context_size", "context size", "context length", "context window",
    "channel error", "connection error", "timed out", "timeout", "rate limit",
    "max planning attempts",          # workflow-level, not a codegen fault
    "cancelled by user", "badrequesterror",
)


# Curated remedies for recurring, well-understood structural errors. Keyed by a substring
# match against "<ename> <evalue>" (lowercased). These turn a SYMPTOM ('X has no attribute
# iloc') into an ACTIONABLE fix — the piece a small model needs to actually self-correct.
# The async-hardening loop can grow this map from ledger observations over time.
_REMEDIES = (
    ("numpy.ndarray' object has no attribute 'iloc",
     "predict_proba/predict and sklearn transforms return NUMPY ARRAYS, not DataFrames — "
     "index positionally: `proba[:, 1]`, never `.iloc[:, 1]`."),
    ("'numpy.ndarray' object has no attribute 'columns",
     "The transformed matrix is a numpy array with no .columns; capture feature names with "
     "`preprocessor.get_feature_names_out()` BEFORE transforming."),
    ("could not convert string to float",
     "Encode categorical columns (OneHotEncoder / OrdinalEncoder) before fitting a numeric "
     "estimator; do not pass raw string columns."),
    ("'>=' not supported between instances of",
     "You are comparing incompatible types (e.g. a float against a dict) — unwrap the value "
     "you actually mean to compare before the operator."),
)


def remedy_for(ename: str, evalue) -> Optional[str]:
    """Return a curated actionable fix for a known structural error, or None."""
    low = f"{ename} {evalue}".lower()
    for pattern, fix in _REMEDIES:
        if pattern in low:
            return fix
    return None


def normalize_error_reason(ename: str, evalue) -> str:
    """Collapse an error to a stable 'reason' key: strip line numbers, hex addresses, and
    other varying values so e.g. 'unexpected indent at line 394' and '...line 427' match."""
    s = re.sub(r"0x[0-9a-fA-F]+", "H", str(evalue))
    s = re.sub(r"\d+", "N", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return f"{ename}:{s[:120]}"


def is_structural_error(ename: str, evalue) -> bool:
    """True only for deterministic, recipe-structural errors worth learning from."""
    if ename in _TRANSIENT_ENAMES:
        return False
    low = f"{ename} {evalue}".lower()
    return not any(p in low for p in _TRANSIENT_PATTERNS)


def node_slug(task_description: Optional[str]) -> str:
    """Stable per-recipe-node key derived from the step's intent text (recipe runs reuse
    the same intent verbatim across invocations, so this is stable run-to-run)."""
    s = re.sub(r"[^a-z0-9]+", "_", (task_description or "").lower()).strip("_")
    return s[:60] or "unknown"


def _append(event: dict) -> None:
    try:
        d = os.path.dirname(LEDGER_PATH)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(LEDGER_PATH, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception:
        pass  # best-effort; never break a task on a logging failure


def record_error(recipe_id, recipe_version, task_description, ename, evalue, model_str=None) -> None:
    """Append one structural-error event for a recipe step. No-op off-recipe or transient."""
    if not recipe_id:
        return
    if not is_structural_error(ename, evalue):
        return
    _append({
        "ts": time.time(), "type": "error",
        "recipe_id": recipe_id, "recipe_version": recipe_version,
        "node": node_slug(task_description),
        "node_intent": (task_description or "")[:200],
        "ename": ename, "reason": normalize_error_reason(ename, evalue),
        "message": str(evalue)[:300], "fix": remedy_for(ename, evalue), "model": model_str,
    })


def record_resolution(recipe_id, recipe_version, task_description, model_str=None) -> None:
    """Append a resolution event: a step that ultimately SUCCEEDED after failing. Lets the
    hardening view distinguish 'recurring but recoverable' from 'hard dead end'."""
    if not recipe_id:
        return
    _append({
        "ts": time.time(), "type": "resolution",
        "recipe_id": recipe_id, "recipe_version": recipe_version,
        "node": node_slug(task_description), "model": model_str,
    })


def _load() -> list:
    events = []
    if not os.path.exists(LEDGER_PATH):
        return events
    try:
        with open(LEDGER_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return events


def common_pitfalls(recipe_id, task_description, top_n: int = 3, min_count: int = 2) -> Optional[str]:
    """Ranked recurring structural errors for a recipe step, formatted for first-attempt
    injection into the Coder prompt. Returns None if nothing clears the frequency bar."""
    if not recipe_id:
        return None
    node = node_slug(task_description)
    reasons: Counter = Counter()
    samples: dict = {}
    fixes: dict = {}
    for e in _load():
        if (e.get("type") == "error" and e.get("recipe_id") == recipe_id
                and e.get("node") == node):
            r = e.get("reason")
            reasons[r] += 1
            samples.setdefault(r, e.get("message", ""))
            if e.get("fix") and r not in fixes:
                fixes[r] = e["fix"]
    top = [(r, c) for r, c in reasons.most_common(top_n) if c >= min_count]
    if not top:
        return None
    lines = []
    for r, c in top:
        line = f"- (seen {c}×) {samples.get(r, '')[:160]}"
        if fixes.get(r):
            line += f"\n    FIX: {fixes[r]}"      # actionable remedy, not just the symptom
        lines.append(line)
    return "\n".join(lines)


def aggregate():
    """For the hardening report: (recipe_id, node) -> {reason: count}, resolution counts,
    a sample message per reason, and the node intent. Pure read; no side effects."""
    err: Counter = Counter()
    res: Counter = Counter()
    samples: dict = {}
    intents: dict = {}
    for e in _load():
        key = (e.get("recipe_id"), e.get("node"))
        if e.get("type") == "error":
            err[(key, e.get("reason"))] += 1
            samples.setdefault((key, e.get("reason")), e.get("message", ""))
            intents[key] = e.get("node_intent", "")
        elif e.get("type") == "resolution":
            res[key] += 1
    return err, res, samples, intents
