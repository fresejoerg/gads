"""aah rung-3 grounding: the governed semantic layer.

Self-contained (needs only duckdb + aah_star.py, both alongside the CSVs). Each metric
is ONE governed definition — the right grain, the right population filter, the right
math. The agent SELECTS a metric via query_metric(...) instead of writing (and possibly
mis-writing) the SQL. The definitions bake in the business rules: user-count / per-user
metrics exclude internal/test accounts; MRR divides annual plans by 12 over active subs
only; power user = 5+ moments in a day; activation = first value moment within 7 days.

Usage in the sandbox:
    from aah_metrics import build_layer, query_metric, list_metrics
    layer = build_layer(".")                      # connects duckdb + loads metrics
    print(list_metrics(layer))                    # catalog the agent can read
    mrr = query_metric(layer, "mrr")              # -> 2685.08
    au  = query_metric(layer, "active_users", period="last_week")
    apac = query_metric(layer, "value_moments", filters={"region": "APAC"}, start="2026-05-01", end="2026-06-30")

Provenance: ported from d-n-ust/ai-analytics-harness (MIT). See research/harness/aah/.
"""
from __future__ import annotations
import datetime as dt

from aah_star import connect

ANALYSIS_DATE = dt.date(2026, 7, 16)   # fixed "today"
DATA_END = dt.date(2026, 7, 12)        # last complete ISO week (Sun)
NAMED_PERIODS = ("last_week", "prev_week", "last_month", "last_quarter", "ytd", "all")

# --- the governed metric catalog (verbatim from semantic_layer.yml) --------- #
METRICS: dict[str, dict] = {
    "value_moments": {
        "description": "Number of value moments (completed habits) in the period. Raw activity volume — includes all users.",
        "base": "agg_active_days", "agg": "sum(moments)", "time_column": "active_date",
        "dimensions": ["region", "platform", "channel", "country"], "supports_internal_filter": True},
    "new_signups": {
        "description": "Number of user signups in the period (raw — includes all accounts).",
        "base": "dim_users", "agg": "count(*)", "time_column": "signup_date",
        "dimensions": ["region", "channel", "platform"], "supports_internal_filter": True},
    "active_users": {
        "description": "Distinct real users with at least one value moment in the period (excludes internal/test accounts).",
        "base": "agg_active_days", "agg": "count(distinct user_id)", "time_column": "active_date",
        "dimensions": ["region", "platform", "channel", "country"],
        "default_filters": ["NOT is_internal"], "supports_internal_filter": True},
    "power_users": {
        "description": "Distinct real users who logged 5+ value moments in a single day during the period (excludes internal/test).",
        "base": "agg_active_days", "agg": "count(distinct case when moments >= 5 then user_id end)",
        "time_column": "active_date", "dimensions": ["region", "platform", "channel", "country"],
        "default_filters": ["NOT is_internal"], "supports_internal_filter": True},
    "days_per_user": {
        "description": "Average active days per active user in the period (frequency; excludes internal/test).",
        "base": "agg_active_days", "agg": "count(*) * 1.0 / nullif(count(distinct user_id), 0)",
        "time_column": "active_date", "dimensions": ["region", "platform", "channel", "country"],
        "default_filters": ["NOT is_internal"], "supports_internal_filter": True},
    "moments_per_day": {
        "description": "Average value moments per active day (depth; excludes internal/test).",
        "base": "agg_active_days", "agg": "sum(moments) * 1.0 / nullif(count(*), 0)",
        "time_column": "active_date", "dimensions": ["region", "platform", "channel", "country"],
        "default_filters": ["NOT is_internal"], "supports_internal_filter": True},
    "reminder_open_rate": {
        "description": "Share of active days on which the user opened a reminder (a retention driver; excludes internal/test).",
        "base": "agg_active_days", "agg": "avg(had_reminder)", "time_column": "active_date",
        "dimensions": ["region", "platform", "channel", "country"],
        "default_filters": ["NOT is_internal"], "supports_internal_filter": True},
    "activation_rate": {
        "description": "Share of the signup cohort that reached a first value moment within 7 days (excludes internal/test).",
        "base": "agg_user_activation", "agg": "avg(CASE WHEN activated THEN 1 ELSE 0 END)",
        "time_column": "signup_date", "dimensions": ["region", "channel", "platform"],
        "default_filters": ["NOT is_internal"], "supports_internal_filter": True},
    "mrr": {
        "description": "Monthly recurring revenue from currently-active subscriptions (annual plans divided by 12).",
        "base": "fct_subscriptions",
        "agg": "sum(CASE WHEN plan = 'annual' THEN billed_amount / 12.0 ELSE billed_amount END)",
        "default_filters": ["is_active"], "time_column": None, "dimensions": ["plan"]},
    "active_subscriptions": {
        "description": "Count of currently-active subscriptions.",
        "base": "fct_subscriptions", "agg": "count(*)", "default_filters": ["is_active"],
        "time_column": None, "dimensions": ["plan"]},
    "paying_users": {
        "description": "Distinct users with an active subscription.",
        "base": "fct_subscriptions", "agg": "count(distinct user_id)", "default_filters": ["is_active"],
        "time_column": None, "dimensions": ["plan"]},
    "arpu": {
        "description": "Average MRR per paying user (annual plans divided by 12).",
        "base": "fct_subscriptions",
        "agg": "sum(CASE WHEN plan = 'annual' THEN billed_amount / 12.0 ELSE billed_amount END) * 1.0 / nullif(count(distinct user_id), 0)",
        "default_filters": ["is_active"], "time_column": None, "dimensions": ["plan"]},
    "marketing_spend": {
        "description": "Total marketing spend in the period.",
        "base": "fct_marketing_spend", "agg": "sum(spend)", "time_column": "spend_date",
        "dimensions": ["channel"]},
    "referrals": {
        "description": "Number of referrals created in the period.",
        "base": "fct_referrals", "agg": "count(*)", "time_column": "referred_ts",
        "filterable": ["status"]},
}


class SemanticError(Exception):
    """Unknown metric / dimension / filter, or a period on a point-in-time metric."""


class SemanticLayer:
    def __init__(self, con, metrics=METRICS):
        self.con = con
        self.metrics = metrics


def _last_complete_week(today: dt.date) -> tuple[dt.date, dt.date]:
    sunday = today - dt.timedelta(days=today.weekday() + 1)
    return sunday - dt.timedelta(days=6), sunday


def resolve_period(name):
    if name in (None, "all"):
        return (None, None)
    today = ANALYSIS_DATE
    if name == "last_week":
        return _last_complete_week(today)
    if name == "prev_week":
        start, end = _last_complete_week(today)
        return start - dt.timedelta(days=7), end - dt.timedelta(days=7)
    if name == "last_month":
        first_this = today.replace(day=1)
        last_prev = first_this - dt.timedelta(days=1)
        return last_prev.replace(day=1), last_prev
    if name == "last_quarter":
        q = (today.month - 1) // 3
        first_this_q = dt.date(today.year, q * 3 + 1, 1)
        last_prev_q = first_this_q - dt.timedelta(days=1)
        return last_prev_q.replace(month=(last_prev_q.month - 2), day=1), last_prev_q
    if name == "ytd":
        return dt.date(today.year, 1, 1), DATA_END
    raise ValueError(f"unknown period {name!r}; use one of {NAMED_PERIODS} or explicit dates")


def _literal(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _allowed_filters(m: dict) -> set:
    allowed = set(m.get("dimensions", [])) | set(m.get("filterable", []))
    if m.get("supports_internal_filter"):
        allowed.add("is_internal")
    return allowed


def compile_metric(layer, name, group_by=None, filters=None, time_grain=None,
                   start=None, end=None, period=None) -> str:
    if name not in layer.metrics:
        raise SemanticError(f"unknown metric {name!r}. Available: {', '.join(layer.metrics)}")
    m = layer.metrics[name]
    time_col = m.get("time_column")
    select, group = [], []
    # Point-in-time metrics (mrr, arpu, active_subscriptions, paying_users) have no time
    # column: a period/grain is meaningless for them, so silently ignore it and return the
    # as-of value rather than raising (a raise would abort the whole answer task).
    if time_grain and time_col:
        select.append(f"date_trunc('{time_grain}', {time_col})::date AS period")
        group.append("period")
    for d in group_by or []:
        if d not in m.get("dimensions", []):
            raise SemanticError(f"cannot group {name!r} by {d!r}. Dimensions: {m.get('dimensions', [])}")
        select.append(d)
        group.append(d)
    select.append(f"{m['agg']} AS value")
    where = list(m.get("default_filters", []))
    if period is not None:
        start, end = resolve_period(period)
    if start and time_col:
        where.append(f"{time_col} >= DATE '{start}'")
    if end and time_col:
        where.append(f"{time_col} <= DATE '{end}'")
    allowed = _allowed_filters(m)
    for col, val in (filters or {}).items():
        if col not in allowed:
            raise SemanticError(f"cannot filter {name!r} by {col!r}. Allowed: {sorted(allowed)}")
        if isinstance(val, (list, tuple)):
            where.append(f"{col} IN ({', '.join(_literal(v) for v in val)})")
        else:
            where.append(f"{col} = {_literal(val)}")
    sql = f"SELECT {', '.join(select)} FROM {m['base']}"
    if where:
        sql += " WHERE " + " AND ".join(where)
    if group:
        sql += " GROUP BY " + ", ".join(group) + " ORDER BY " + ", ".join(group)
    return sql


_DEFAULT_LAYER = None


def build_layer(data_dir: str = ".") -> SemanticLayer:
    return SemanticLayer(connect(data_dir))


def get_layer(data_dir: str = ".") -> SemanticLayer:
    """The cached default layer (built from the working dir on first use). Lets the
    agent call query_metric('mrr') without threading a layer object around."""
    global _DEFAULT_LAYER
    if _DEFAULT_LAYER is None:
        _DEFAULT_LAYER = build_layer(data_dir)
    return _DEFAULT_LAYER


def _resolve(a, b):
    """Accept either (metric) / (metric, layer) OR (layer, metric) — whichever the
    caller reaches for. Returns (layer, metric_name)."""
    if isinstance(a, SemanticLayer):
        return a, b
    if isinstance(b, SemanticLayer):
        return b, a
    return get_layer(), a


def __getattr__(name):
    # PEP 562: `from aah_metrics import layer` builds and returns the default layer.
    if name == "layer":
        return get_layer()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def query_metric(a, b=None, **kw):
    """Return one scalar for a metric. Call as query_metric('mrr'),
    query_metric('active_users', period='last_week'), or query_metric(layer, 'mrr')."""
    _, rows = query_rows(a, b, **kw)
    if not rows or rows[0][-1] is None:
        return None
    return float(rows[0][-1])


def query_rows(a, b=None, **kw):
    """Rows for a metric (grouped/time-grained). Same flexible signature as query_metric."""
    layer, name = _resolve(a, b)
    cur = layer.con.execute(compile_metric(layer, name, **kw))
    return [d[0] for d in cur.description], cur.fetchall()


def list_metrics(layer=None) -> str:
    layer = layer if isinstance(layer, SemanticLayer) else get_layer()
    lines = ["Governed metrics (call query_metric(layer, name, ...) with these names):"]
    for name, m in layer.metrics.items():
        bits = [f"- {name}: {m['description']}"]
        if m.get("dimensions"):
            bits.append(f"    dimensions (group_by / filter): {', '.join(m['dimensions'])}")
        if m.get("supports_internal_filter"):
            bits.append("    excludes internal/test by default")
        if m.get("time_column"):
            bits.append("    time-filterable (period= or start=/end=) and grainable (time_grain=week|month|day)")
        else:
            bits.append("    point-in-time (as of now); no period filter")
        lines.append("\n".join(bits))
    lines.append(f"\nNamed periods: {', '.join(NAMED_PERIODS)} (or explicit start=/end= 'YYYY-MM-DD').")
    return "\n".join(lines)


if __name__ == "__main__":
    layer = build_layer()
    for name in ("mrr", "arpu", "active_users", "power_users", "value_moments"):
        print(f"{name:16s} = {query_metric(layer, name)}")
