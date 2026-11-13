"""Regression guard for bring-your-own-key credentials (HOSTED-2a). No paid LLM calls.

Default run: a STUB proxy (local HTTP server) and the scratch DB `gads_tenancy_test`, so
nothing reaches a provider:
  catalog parsing (Kimi/local not BYOK-eligible) · encryption + write-only API · per-model key
  routing in llm.py (body api_key; the proxy's own key unchanged) · FAIL CLOSED with no HTTP call ·
  per-user hierarchy · cloud_pinned needs a key · launch refused without keys.

`--live` (opt-in: each run leaves a few failed-call traces on the gateway key in MLflow): against
the REAL proxy, at (almost) zero cost because every tester call is rejected:
  * a body key is still honoured (a proxy upgrade that banned it would fail here loudly);
  * the exact validate_key body is accepted for each provider (3 tiny gateway calls), and the
    real validate_key rejects a bogus key;
  * a full workflow launched by a tester whose stored key is invalid fails at its model stages,
    never on the gateway key, and the key appears in no task row, ledger or scorecard.
    Ledgers/scorecards go to a temp dir.

    PYTHONPATH=src uv run python scripts/test_byok.py
"""
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
LIVE = "--live" in sys.argv
ENV_FILE = dotenv_values(ROOT / ".env")
TMP = Path(tempfile.mkdtemp(prefix="gads_byok_"))
(TMP / "ws").mkdir(); (TMP / "ds").mkdir()
FAILS = []


def check(cond, label):
    print(f"  {'✓' if cond else '✗'} {label}")
    if not cond:
        FAILS.append(label)


# ——— stub proxy ————————————————————————————————————————————————————————————————————————
STUB_CATALOG = [
    {"model_name": "gemini-3.5-flash-lite", "model_info": {"litellm_provider": "gemini"}, "litellm_params": {"model": "gemini/x"}},
    {"model_name": "gemini-3.8-flash", "model_info": {"litellm_provider": "gemini"}, "litellm_params": {"model": "gemini/y"}},
    {"model_name": "claude-haiku-5.5", "model_info": {"litellm_provider": "anthropic"}, "litellm_params": {"model": "anthropic/h"}},
    {"model_name": "claude-sonnet-5.5", "model_info": {"litellm_provider": "anthropic"}, "litellm_params": {"model": "anthropic/s"}},
    {"model_name": "gpt-6-luna", "model_info": {"litellm_provider": "openai"}, "litellm_params": {"model": "openai/l"}},
    {"model_name": "kimi-k3.1-pro", "model_info": {"litellm_provider": "openai"}, "litellm_params": {"model": "openai/k", "api_base": "https://api.moonshot.ai/v1"}},
    {"model_name": "local_model", "model_info": {"litellm_provider": "openai"}, "litellm_params": {"model": "openai/local", "api_base": "http://host:1234/v1"}},
]
STUB_REQUESTS = []


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/model/info"):
            return self._json({"data": STUB_CATALOG})
        if self.path.startswith("/v1/models"):
            return self._json({"data": [{"id": m["model_name"]} for m in STUB_CATALOG]})
        self._json({}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
        STUB_REQUESTS.append({"auth": self.headers.get("authorization"), "body": body})
        return self._json({"id": "x", "object": "chat.completion", "model": body.get("model"),
                           "choices": [{"index": 0, "finish_reason": "stop",
                                        "message": {"role": "assistant", "content": "{\"ok\": true}"}}],
                           "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})


def start_stub():
    srv = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


real_db = ENV_FILE.get("GADS_DATABASE_URL") or os.environ.get("GADS_DATABASE_URL", "")
SCRATCH_URL = real_db.rsplit("/", 1)[0] + "/gads_tenancy_test"
assert SCRATCH_URL != real_db
from cryptography.fernet import Fernet  # noqa: E402
common = {
    "GADS_DATABASE_URL": SCRATCH_URL, "GADS_AUTH": "dev_header", "GADS_DEPLOYMENT": "local",
    "GADS_ADMIN_EMAIL": "admin@test.local", "GADS_WORKSPACE_ROOT": str(TMP / "ws"),
    "GADS_DATASETS_ROOT": str(TMP / "ds"), "GADS_OTEL_ENDPOINT": "", "GADS_DISTILL_CAPTURE": "false",
    "GADS_SANDBOX_URL": "http://127.0.0.1:9", "GADS_SECRET_KEY": Fernet.generate_key().decode(),
    "GADS_EVALS_DIR": str(TMP / "evals"), "GADS_DIAL_LEDGER": str(TMP / "dial.jsonl"),
    "GADS_ROUTING_LEDGER": str(TMP / "routing.jsonl"), "GADS_ERROR_LEDGER": str(TMP / "errors.jsonl"),
    "GADS_ROUTING_MODE": "cloud",
}
if LIVE:
    os.environ.update(common)  # real LITELLM_BASE_URL / key from .env via load_dotenv
else:
    stub = start_stub()
    os.environ.update({**common, "LITELLM_BASE_URL": f"http://127.0.0.1:{stub.server_port}/v1",
                       "LITELLM_MASTER_KEY": "proxy-virtual-key"})

from sqlmodel import SQLModel, Session, select  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from gads.core.database import engine, init_db  # noqa: E402
from gads.core.models import ProviderKey, Project, Task  # noqa: E402
from gads.core import credentials, registry, tenancy  # noqa: E402
import gads.core.server as server  # noqa: E402
from gads.core import llm  # noqa: E402

assert "gads_tenancy_test" in str(engine.url)
SQLModel.metadata.drop_all(engine)
init_db()
client = TestClient(server.app)
H = lambda e: {"X-GADS-User": e}  # noqa: E731
ADMIN, A, B = "admin@test.local", "alice@test.local", "bob@test.local"


def live_part():
    print("L1. the real proxy still honours a body key")
    import httpx
    base = (os.getenv("LITELLM_BASE_URL") or "http://localhost:4000/v1").rstrip("/")
    r = httpx.post(f"{base}/chat/completions", timeout=60,
                   headers={"Authorization": f"Bearer {os.environ['LITELLM_MASTER_KEY']}"},
                   json={"model": "gemini-3.5-flash-lite", "max_tokens": 5,
                         "messages": [{"role": "user", "content": "Say OK"}],
                         "api_key": "AIzaInvalidBogusKey000000000000000000"})
    check(r.status_code in (400, 401) and "API key not valid" in r.text,
          f"bogus body key -> provider auth error ({r.status_code})")

    print("L1b. validate_key: its body works for every provider; a bogus key is rejected")
    for prov, model in credentials.VALIDATION_MODELS.items():
        r = httpx.post(f"{base}/chat/completions", timeout=60,
                       headers={"Authorization": f"Bearer {os.environ['LITELLM_MASTER_KEY']}"},
                       json={"model": model, "messages": [{"role": "user", "content": "Reply with OK"}], "max_tokens": 5})
        check(r.status_code == 200, f"{prov}: validation body accepted with the gateway key ({r.status_code})")
    try:
        asyncio.run(credentials.validate_key("anthropic", "sk-ant-api03-InvalidBogusKey0000000000"))
        check(False, "real validate_key rejects a bogus key")
    except ValueError as e:
        check("Bogus" not in str(e), f"real validate_key rejects a bogus key without echoing it ({e})")

    print("L2. a tester's workflow with an invalid stored key never runs on the gateway key")
    tenancy.get_or_create_user(A)
    uid = tenancy.get_or_create_user(A).id
    credentials.store_key(uid, "gemini", "AIzaInvalidBogusKey000000000000000000")  # skips validation on purpose
    r = client.post("/projects", headers=H(A), json={"name": "byok-negative", "objective": "Say hello."})
    check(r.status_code == 200, f"launch accepted ({r.status_code})")
    pid = r.json()["project"]["id"]
    with Session(engine) as s:
        p = s.get(Project, uuid.UUID(pid))
        tasks = s.exec(select(Task).where(Task.project_id == p.id)).all()
        blob = " ".join(f"{t.description} {t.error or ''} {json.dumps(t.result_json or {})}" for t in tasks)
        narrative = p.narrative or ""
    import re as _re
    models = sorted(set(_re.findall(r"\b(gemini-[\w.-]+|claude-[\w.-]+|gpt-[\w.-]+|kimi-[\w.-]+|local_model)\b", blob)))
    check(models and all(m.startswith("gemini") for m in models),
          f"only gemini models were used ({models})")
    stages = {"SpecDrafter", "Router", "Planner", "PlanCritique", "Synthesizer", "Critique"}
    check(not any(t.status == "completed" and t.assigned_to in stages for t in tasks),
          "no model stage completed")
    check("API key not valid" in blob or "AuthenticationError" in blob or "HALTED" in narrative or "rejected" in blob.lower(),
          "the run failed on the provider's auth error")
    canary = "InvalidBogusKey"
    files = [f for f in TMP.rglob("*") if f.is_file() and f.suffix in (".json", ".jsonl", ".md", ".txt")]
    leaked = [str(f.relative_to(TMP)) for f in files if canary in f.read_text(errors="ignore")]
    check(canary not in blob and canary not in narrative and not leaked,
          f"the key is in no task row, narrative, ledger or scorecard (scanned {len(files)} files; leaked: {leaked})")


if LIVE:
    live_part()
    SQLModel.metadata.drop_all(engine)
    import shutil
    shutil.rmtree(TMP, ignore_errors=True)
    print("\nall checks passed" if not FAILS else f"\n{len(FAILS)} check(s) FAILED")
    sys.exit(1 if FAILS else 0)

print("1. model catalog")
cat = credentials.model_providers(force=True)
check(cat["gemini-3.8-flash"] == "gemini" and cat["claude-haiku-5.5"] == "anthropic" and cat["gpt-6-luna"] == "openai",
      "provider from litellm_provider")
check(cat["kimi-k3.1-pro"] is None, "Kimi (openai-compatible, custom api_base) is NOT BYOK-eligible")
check(cat["local_model"] is None, "local_model is not BYOK-eligible")
mixed = credentials.parse_catalog([STUB_CATALOG[4], {**STUB_CATALOG[5], "model_name": "gpt-6-luna"}])
check(mixed["gpt-6-luna"] is None, "a model with any non-eligible deployment is not eligible")

print("2. encryption + write-only API")
raw_g, raw_a = "AIzaGeminiTestKey1234567890abcd", "sk-ant-test-key-ABCDEFGH9876"
orig_validate = credentials.validate_key
async def fake_validate(provider, raw):
    if "bad" in raw:
        raise ValueError(f"The {provider} key was rejected (HTTP 400)")
credentials.validate_key = fake_validate
r = client.put("/me/keys/gemini", headers=H(A), json={"api_key": raw_g})
check(r.status_code == 200 and r.json() == {"provider": "gemini", "last4": "abcd"}, f"store gemini key ({r.status_code})")
check(client.put("/me/keys/anthropic", headers=H(A), json={"api_key": raw_a}).status_code == 200, "store anthropic key")
check(client.put("/me/keys/openai", headers=H(A), json={"api_key": "sk-bad-key-000"}).status_code == 400, "rejected key not stored")
check(client.put("/me/keys/kimi", headers=H(A), json={"api_key": "whatever123"}).status_code == 400, "unknown provider refused")
g = client.get("/me/keys", headers=H(A))
check(raw_g not in g.text and raw_a not in g.text and {k["provider"] for k in g.json()["keys"]} == {"anthropic", "gemini"},
      "GET /me/keys shows provider + last4, never the key")
with Session(engine) as s:
    rows = s.exec(select(ProviderKey)).all()
check(rows and all(raw_g not in r.ciphertext and raw_a not in r.ciphertext for r in rows), "ciphertext only in the DB")
check("abcd" not in client.get("/me/keys", headers=H(B)).text, "another user does not see your keys")
uid_a = tenancy.get_or_create_user(A).id
check(credentials.load_keys(uid_a) == {"gemini": raw_g, "anthropic": raw_a}, "decrypts for a run snapshot")
saved = os.environ.pop("GADS_SECRET_KEY")
r = client.put("/me/keys/gemini", headers=H(A), json={"api_key": raw_g})
check(r.status_code == 503, "storing refused without GADS_SECRET_KEY")
os.environ["GADS_SECRET_KEY"] = saved
check(client.delete("/me/keys/anthropic", headers=H(A)).status_code == 200 and
      [k["provider"] for k in client.get("/me/keys", headers=H(A)).json()["keys"]] == ["gemini"], "delete")
client.put("/me/keys/anthropic", headers=H(A), json={"api_key": raw_a})
credentials.validate_key = orig_validate

print("3. launch refused without keys")
with Session(engine) as s:
    before = len(s.exec(select(Project)).all())
r = client.post("/projects", headers=H(B), json={"name": "nokey", "objective": "do something"})
with Session(engine) as s:
    after = len(s.exec(select(Project)).all())
check(r.status_code == 400 and "model key" in r.text and before == after, "tester with no key: 400, nothing created")
check(client.post("/projects", headers=H(B), json={"name": "draft", "objective": ""}).status_code == 200,
      "a draft project (no objective, e.g. for an upload) is still allowed")

print("4. per-call key routing in llm.py (fail closed)")
creds = credentials.run_credentials_for_owner(tenancy.get_or_create_user(A))
check(creds is not None and set(creds.keys) == {"gemini", "anthropic"}, "tester snapshot = their keys")
check(credentials.run_credentials_for_owner(tenancy.get_or_create_user(ADMIN)) is credentials.GATEWAY, "admin snapshot = GATEWAY marker")
check(raw_g not in repr(creds) and raw_a not in repr(creds), "RunCredentials repr hides the keys")
def prep(model):
    kw = {}
    llm._prepare_llm_kwargs(model, [{"role": "user", "content": "x"}], kw)
    return kw
with credentials.credentials_scope(creds):
    kw = prep("gemini-3.8-flash")
    check(kw["extra_body"]["api_key"] == raw_g and kw["api_key"] == "proxy-virtual-key",
          "gemini call: body key = tester's gemini key, auth = proxy virtual key")
    check(prep("claude-sonnet-5.5")["extra_body"]["api_key"] == raw_a, "claude call: tester's anthropic key")
    for m, why in [("gpt-6-luna", "no openai key"), ("kimi-k3.1-pro", "custom endpoint"), ("local_model", "local")]:
        try:
            prep(m); check(False, f"{m} fails closed ({why})")
        except credentials.MissingProviderKey:
            check(True, f"{m} fails closed ({why})")
with credentials.credentials_scope(credentials.GATEWAY):
    check("api_key" not in prep("gemini-3.8-flash").get("extra_body", {}), "GATEWAY run: no body key")
try:
    prep("gemini-3.8-flash"); check(False, "no credentials context in multi-user mode fails closed")
except credentials.MissingProviderKey:
    check(True, "no credentials context in multi-user mode fails closed (no silent gateway)")

async def one_call(model):
    from pydantic import BaseModel
    class Out(BaseModel):
        ok: bool
    return await llm.get_structured_completion(model, Out, [{"role": "user", "content": "x"}])
STUB_REQUESTS.clear()
with credentials.credentials_scope(creds):
    try:
        asyncio.run(one_call("gpt-6-luna")); check(False, "missing key raises before any HTTP")
    except credentials.MissingProviderKey:
        check(not STUB_REQUESTS, "missing key raises before any HTTP request")
    try:
        asyncio.run(one_call("gemini-3.8-flash"))
    except Exception as e:
        print("    (structured parse:", type(e).__name__, ")")
sent = [r for r in STUB_REQUESTS if r["body"].get("model") == "gemini-3.8-flash"]
check(sent and all(r["body"].get("api_key") == raw_g and r["auth"] == "Bearer proxy-virtual-key" for r in sent),
      f"on the wire: body api_key = tester key, Authorization = proxy key ({len(sent)} request(s))")

print("5. per-user hierarchy")
with credentials.credentials_scope(creds), registry.run_settings_scope({**registry.global_settings(), "routing_mode": "cloud"}):
    h = asyncio.run(registry.get_model_hierarchy())
allm = {m for t in h.values() for m in t["models"]}
check(allm and all(m.startswith(("gemini", "claude")) for m in allm), f"only keyed providers' models: {sorted(allm)}")
only_a = credentials.RunCredentials(keys={"anthropic": raw_a}, providers=creds.providers)
with credentials.credentials_scope(only_a), registry.run_settings_scope({**registry.global_settings(), "routing_mode": "cloud"}):
    h = asyncio.run(registry.get_model_hierarchy())
check({m for t in h.values() for m in t["models"]} <= {"claude-haiku-5.5", "claude-sonnet-5.5"}, "anthropic-only tester: Claude-only tiers")
with credentials.credentials_scope(only_a), registry.run_settings_scope(
        {**registry.global_settings(), "routing_mode": "cloud_pinned", "pinned_model": "gemini-3.8-flash"}):
    try:
        asyncio.run(registry.get_model_hierarchy()); check(False, "pinning an unkeyed model refused")
    except RuntimeError:
        check(True, "pinning a model you hold no key for is refused")
with registry.run_settings_scope({**registry.global_settings(), "routing_mode": "cloud"}):
    h = asyncio.run(registry.get_model_hierarchy())
check("gpt-6-luna" in {m for t in h.values() for m in t["models"]}, "admin (no BYOK scope): unfiltered")

print("6. workflow wrapper carries the owner's credentials")
seen = {}
real = server.run_agent_workflow
async def stub_run(project_id, objective, instruction_id=None):
    c = credentials.current()
    seen[str(project_id)] = c if c is credentials.GATEWAY or c is None else sorted(c.keys)
server.run_agent_workflow = stub_run
pa = client.post("/projects", headers=H(A), json={"name": "a", "objective": ""}).json()["project"]["id"]
padm = client.post("/projects", headers=H(ADMIN), json={"name": "adm", "objective": ""}).json()["project"]["id"]
async def both():
    await asyncio.gather(server.run_agent_workflow_wrapper(uuid.UUID(pa), "x"),
                         server.run_agent_workflow_wrapper(uuid.UUID(padm), "x"))
asyncio.run(both())
server.run_agent_workflow = real
check(seen[pa] == ["anthropic", "gemini"] and seen[padm] is credentials.GATEWAY, f"tester run = their keys, admin run = GATEWAY {seen}")

print("6b. keys come from the project OWNER, whoever launches")
pb_ = client.post("/projects", headers=H(B), json={"name": "bob-draft", "objective": ""}).json()["project"]["id"]
r = client.post("/projects", headers=H(ADMIN), json={"name": "x", "objective": "go", "existing_project_id": pb_})
check(r.status_code == 400 and "model key" in r.text, "admin resuming a keyless tester's project -> 400 (owner has no key)")
r = client.post(f"/projects/{pb_}/followup", headers=H(ADMIN), json={"objective": "go"})
check(r.status_code == 400 and "model key" in r.text, "admin follow-up on a keyless tester's project -> 400")
check(credentials.current() is None, "no leak outside the run")

print("7. escalation terminates (no ping-pong inside a tier)")
hier = {"T3": {"models": ["a3"]}, "T2": {"models": ["b2", "c2"]}, "T1": {"models": ["d1", "e1"]}}
with registry.run_settings_scope({**registry.global_settings(), "routing_mode": "cloud"}):
    tried, m, seq = set(), "a3", ["a3"]
    for _ in range(20):
        tried.add(m)
        m = registry.get_next_model_dynamic(m, hier, exclude=tried)
        if m is None:
            break
        seq.append(m)
    check(m is None and sorted(seq) == ["a3", "b2", "c2", "d1", "e1"],
          f"each model tried once, then None: {seq}")
    check(registry.get_next_model_dynamic("d1", hier) == "e1", "without exclude: still escalates within the tier")
    check(registry.get_next_model_dynamic("b2", hier, exclude={"c2"}) == "d1", "tier exhausted -> next tier")

SQLModel.metadata.drop_all(engine)

print("(live part not run; it is opt-in: scripts/test_byok.py --live)")
import shutil  # noqa: E402
shutil.rmtree(TMP, ignore_errors=True)
print("\nall checks passed" if not FAILS else f"\n{len(FAILS)} check(s) FAILED")
sys.exit(1 if FAILS else 0)
