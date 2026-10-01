"""Workflow tracing over OpenTelemetry, exported to MLflow (replaces the Langfuse v2 client).

The LiteLLM proxy (1.105+) only speaks OTLP, which the self-hosted Langfuse v2 server cannot
ingest, so the run tree now lives in MLflow: GADS exports its stage spans to MLflow's OTLP
endpoint, and `llm.py` sends a W3C `traceparent` naming the current stage span with every
model call, so the proxy's own spans (`litellm_request`, carrying messages, tokens, cost and
`llm.py`'s metadata under `metadata.requester_metadata`) nest under that stage.

The API mirrors the slice of the Langfuse client `server.py` used — `start_trace(...)`,
`trace.span(name, metadata)`, `.id`, `.end(output=)`, `.update(...)` — so the orchestration
code did not have to be restructured. Spans are started with an explicit parent, never as the
"current" span: stage spans outlive their lexical scope and overlap across awaits.

The trace id IS the project UUID (as the Langfuse trace id was), so a project is found in
MLflow as `tr-<project uuid hex>` without a lookup table.

The proxy must export to the SAME MLflow experiment (`OTEL_HEADERS` on the litellm service);
if the two ids differ, the proxy spans land in another experiment and the run tree silently
has no model calls under its stages.
"""
import json
import os
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from opentelemetry import trace as otel_trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

# Imported (via llm.py) before server.py's own load_dotenv() runs.
load_dotenv()

OTEL_ENDPOINT = os.getenv("GADS_OTEL_ENDPOINT", "http://localhost:5000/v1/traces")
MLFLOW_EXPERIMENT_ID = os.getenv("GADS_MLFLOW_EXPERIMENT_ID", "")

# Span outputs are model dumps (plans, code, stdout); cap them so one Coder task cannot
# bloat the trace store.
_MAX_OUTPUT_CHARS = 32_000

_forced_trace_id: ContextVar[Optional[int]] = ContextVar("gads_forced_trace_id", default=None)


class _ProjectTraceIdGenerator(RandomIdGenerator):
    """Use the project UUID as the trace id when one is set, random otherwise.

    OTEL has no per-span trace-id argument; a synthetic parent SpanContext would make the
    root a child of a span that never exists (MLflow renders it as missing).
    """

    def generate_trace_id(self) -> int:
        forced = _forced_trace_id.get()
        return forced if forced is not None else super().generate_trace_id()


def _build_provider() -> TracerProvider:
    provider = TracerProvider(
        resource=Resource.create({"service.name": "gads-backend"}),
        id_generator=_ProjectTraceIdGenerator(),
    )
    headers = {"x-mlflow-experiment-id": MLFLOW_EXPERIMENT_ID} if MLFLOW_EXPERIMENT_ID else {}
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=OTEL_ENDPOINT, headers=headers)))
    return provider


_provider = _build_provider()
_tracer = _provider.get_tracer("gads")


def _dumps(value: Any) -> str:
    try:
        text = json.dumps(value, default=str)
    except Exception:
        text = str(value)
    return text if len(text) <= _MAX_OUTPUT_CHARS else text[:_MAX_OUTPUT_CHARS] + "…[truncated]"


def _set_attrs(span, prefix: str, values: Optional[Dict[str, Any]]):
    for k, v in (values or {}).items():
        if v is None:
            continue
        span.set_attribute(f"{prefix}{k}", v if isinstance(v, (str, bool, int, float)) else _dumps(v))


class Span:
    def __init__(self, otel_span):
        self._span = otel_span
        self._ended = False
        self.id = format(otel_span.get_span_context().span_id, "016x")

    def update(self, output: Any = None, metadata: Optional[Dict[str, Any]] = None):
        if self._ended:
            return
        if output is not None:
            self._span.set_attribute("mlflow.spanOutputs", _dumps(output))
        _set_attrs(self._span, "gads.", metadata)

    def end(self, output: Any = None):
        # Idempotent: several stages end their span on more than one exit path.
        if self._ended:
            return
        self.update(output=output)
        self._span.end()
        self._ended = True


class Trace(Span):
    """The root span of one workflow run; stage spans are its direct children."""

    def __init__(self, otel_span):
        super().__init__(otel_span)
        self.trace_id = format(otel_span.get_span_context().trace_id, "032x")

    def span(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> Span:
        child = _tracer.start_span(name, context=otel_trace.set_span_in_context(self._span))
        _set_attrs(child, "gads.", metadata)
        return Span(child)

    def update(self, output: Any = None, metadata: Optional[Dict[str, Any]] = None, tags: Optional[list] = None):
        if tags and not self._ended:
            self._span.set_attribute("gads.tags", list(tags))
        super().update(output=output, metadata=metadata)


def start_trace(project_id: uuid.UUID, name: str, metadata: Optional[Dict[str, Any]] = None) -> Trace:
    token = _forced_trace_id.set(project_id.int)
    try:
        root = _tracer.start_span(name, context=otel_trace.set_span_in_context(otel_trace.INVALID_SPAN))
    finally:
        _forced_trace_id.reset(token)
    _set_attrs(root, "gads.", {"project_id": str(project_id), **(metadata or {})})
    return Trace(root)


def traceparent(trace_id: Optional[str], span_id: Optional[str]) -> Optional[str]:
    """W3C traceparent for a model call made under `span_id` (sampled)."""
    if not trace_id or not span_id:
        return None
    return f"00-{trace_id}-{span_id}-01"


def flush(timeout_millis: int = 10_000):
    _provider.force_flush(timeout_millis)
