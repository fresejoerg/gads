"""Regression guard for the deployment seams (core/config.py, 032 phase 0). No LLM, no DB.

Checks that every machine-specific path and URL is an override, not a constant:
1. `/home/joergf` appears in src/ only in core/config.py.
2. With GADS_WORKSPACE_ROOT / GADS_DATASETS_ROOT / GADS_SANDBOX_URL / GADS_ROUTING_LEDGER /
   GADS_HEADER_IMAGE overridden, every consumer resolves to the override (fresh interpreter
   per case, since several are import-time constants).
3. Tracing with GADS_OTEL_ENDPOINT empty exports nothing but still yields non-zero ids.
4. The sandbox factory returns fresh LocalDockerBackend instances that satisfy SandboxBackend,
   and rejects an unknown backend name; GADS_DEPLOYMENT is validated.

    PYTHONPATH=src uv run python scripts/test_config_seams.py
"""
import json
import os
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
FAILS = []
# Empty cwd so the repo .env (load_dotenv) cannot supply the values under test; server.py
# import writes gads_data/ into it, hence a real temp dir.
TMP_CWD = tempfile.mkdtemp(prefix="gads_seams_")


def check(cond, label):
    print(f"  {'✓' if cond else '✗'} {label}")
    if not cond:
        FAILS.append(label)


def run_py(code: str, env_over: dict) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GADS_")}
    # server.py raises at import without a DB URL; a dummy one never connects at import.
    env.update({"PYTHONPATH": str(SRC), "GADS_DATABASE_URL": "postgresql://x:y@127.0.0.1:1/none"})
    env.update(env_over)
    return subprocess.run([sys.executable, "-c", code], env=env, cwd=TMP_CWD,
                          capture_output=True, text=True, timeout=180)


print("1. no hardcoded home paths outside core/config.py")
offenders = [str(p.relative_to(ROOT)) for p in SRC.rglob("*.py")
             if "/home/joergf" in p.read_text(encoding="utf-8") and p.name != "config.py"]
check(not offenders, f"only config.py mentions /home/joergf (offenders: {offenders})")

print("2. overrides reach every consumer")
OVR = {
    "GADS_WORKSPACE_ROOT": "/tmp/seam/ws",
    "GADS_DATASETS_ROOT": "/tmp/seam/data",
    "GADS_SANDBOX_URL": "http://sandbox.test:9999/",
    "GADS_ROUTING_LEDGER": "/tmp/seam/routing.jsonl",
    "GADS_HEADER_IMAGE": "/tmp/seam/header.png",
}
probe = r"""
import json, uuid, asyncio, inspect
from gads.core import config
from gads.tools.sandbox import make_sandbox, SandboxClient
from gads.core.handover import HandoverManager
from gads.core.executor import ExecutionManager
from gads.core import dial
from gads.eval import record
import gads.core.server as server
sb = make_sandbox()
out = {
  "config.ws": config.WORKSPACE_ROOT, "config.ds": config.DATASETS_ROOT,
  "config.img": config.header_image(), "config.wsdir": config.workspace_dir("p1"),
  "server.ws": server.WORKSPACE_ROOT, "server.ds": server.DATASETS_ROOT,
  "record.ws": str(record.WORKSPACE_ROOT), "dial.routing": dial.ROUTING_LEDGER_PATH,
  "sandbox.url": sb.base_url, "SandboxClient.url": SandboxClient().base_url,
  "handover.url": HandoverManager().sandbox.base_url,
  "executor.url": ExecutionManager().sandbox.base_url,
  "handover.ws_default": inspect.signature(HandoverManager.create_bundle).parameters["workspace_root"].default,
}
print("JSON:" + json.dumps(out))
"""
r = run_py(probe, OVR)
line = next((l for l in r.stdout.splitlines() if l.startswith("JSON:")), None)
check(line is not None, "consumers import with overrides" + ("" if line else f"\n{r.stderr[-1500:]}"))
if line:
    got = json.loads(line[5:])
    check(got["config.ws"] == got["server.ws"] == got["record.ws"] == "/tmp/seam/ws", f"workspace root: {got['server.ws']}, {got['record.ws']}")
    check(got["config.wsdir"] == "/tmp/seam/ws/p1", "workspace_dir joins project id")
    check(got["config.ds"] == got["server.ds"] == "/tmp/seam/data", f"datasets root: {got['server.ds']}")
    check(got["config.img"] == "/tmp/seam/header.png", "header image")
    check(got["dial.routing"] == "/tmp/seam/routing.jsonl", "routing ledger")
    urls = {got[k] for k in ("sandbox.url", "SandboxClient.url", "handover.url", "executor.url")}
    check(urls == {"http://sandbox.test:9999"}, f"sandbox URL everywhere (trailing / stripped): {urls}")
    check(got["handover.ws_default"] is None, "handover workspace_root resolved at call time")
# Streamlit can't be imported headless; check it reads the same config names.
ui = (SRC / "gads/ui/streamlit_app.py").read_text(encoding="utf-8")
check(all(s in ui for s in ("config.WORKSPACE_ROOT", "config.header_image()", "config.DATASETS_ROOT")),
      "streamlit_app reads workspace/header/datasets from config")

print("2b. defaults unchanged when nothing is set")
r = run_py("from gads.core import config; print('JSON:'+__import__('json').dumps([config.WORKSPACE_ROOT, config.DATASETS_ROOT, config.SANDBOX_URL, config.DEPLOYMENT]))", {})
line = next((l for l in r.stdout.splitlines() if l.startswith("JSON:")), None)
check(line and json.loads(line[5:]) == ["/home/joergf/projects/MyLocalStack/data/workspaces",
                                         "/home/joergf/datasets", "http://localhost:8000", "local"],
      f"local defaults: {line}")

print("3. tracing off still yields real ids")
tr = r"""
import uuid, json
from gads.core import tracing
pid = uuid.uuid4()
t = tracing.start_trace(pid, "seam-test"); s = t.span("child"); s.end(); t.end(); tracing.flush(1000)
procs = getattr(tracing._provider._active_span_processor, "_span_processors", ())
print("JSON:" + json.dumps({"enabled": tracing.ENABLED, "procs": len(procs), "trace": t.trace_id, "want": pid.hex, "span": s.id}))
"""
r = run_py(tr, {"GADS_OTEL_ENDPOINT": ""})
line = next((l for l in r.stdout.splitlines() if l.startswith("JSON:")), None)
check(line is not None, "tracing imports with endpoint empty" + ("" if line else f"\n{r.stderr[-1500:]}"))
if line:
    got = json.loads(line[5:])
    check(got["enabled"] is False and got["procs"] == 0, "no exporter when endpoint empty")
    check(got["trace"] == got["want"], "trace id is still the project UUID")
    check(got["span"] != "0" * 16, "span id non-zero")
r = run_py(tr, {"GADS_OTEL_ENDPOINT": "http://127.0.0.1:1/v1/traces"})
line = next((l for l in r.stdout.splitlines() if l.startswith("JSON:")), None)
check(line and json.loads(line[5:])["procs"] == 1, "exporter attached when endpoint set")

print("4. sandbox factory + deployment flag")
r = run_py(r"""
from gads.tools.sandbox import make_sandbox, SandboxBackend, LocalDockerBackend
a, b = make_sandbox(), make_sandbox()
print("OK" if (a is not b and a.client is not b.client and isinstance(a, SandboxBackend)
               and type(a) is LocalDockerBackend) else "BAD")
""", {})
check(r.stdout.strip() == "OK", f"fresh LocalDockerBackend per call, satisfies protocol {r.stderr[-300:]}")
r = run_py("from gads.tools.sandbox import make_sandbox; make_sandbox()", {"GADS_SANDBOX_BACKEND": "modal"})
check(r.returncode != 0 and "GADS_SANDBOX_BACKEND" in r.stderr, "unknown backend rejected")
r = run_py("import gads.core.config", {"GADS_DEPLOYMENT": "cloud"})
check(r.returncode != 0 and "GADS_DEPLOYMENT" in r.stderr, "invalid GADS_DEPLOYMENT rejected")
r = run_py("from gads.core import config; print(config.DEPLOYMENT)", {"GADS_DEPLOYMENT": "Hosted"})
check(r.stdout.strip() == "hosted", "GADS_DEPLOYMENT normalized")

shutil.rmtree(TMP_CWD, ignore_errors=True)
print("\nall checks passed" if not FAILS else f"\n{len(FAILS)} check(s) FAILED")
sys.exit(1 if FAILS else 0)
