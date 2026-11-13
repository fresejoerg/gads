"""Regression guard for tenancy (HOSTED-1, approach_docs/032 §4). No LLM calls.

Runs against a SCRATCH Postgres database, never the research DB (test users and projects
would otherwise show up in eval_run.py --all and the ledgers). The scratch DB is wiped and
recreated at the start of each run; workspaces and datasets live in a temp dir.

    docker exec litellm_db psql -U llmuser -d gads -c "create database gads_tenancy_test"  # once
    PYTHONPATH=src uv run python scripts/test_tenancy.py

Covers: every route has a policy (default deny); identity (dev_header, 401 without it);
project / task / file / stream / ws isolation between two testers; admin sees all; the two
unsafe handlers (download_file symlinks, register-external roots); per-user settings validation;
per-run settings isolation under concurrency; the hosted guard on unverified auth modes.
"""
import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
TMP = Path(tempfile.mkdtemp(prefix="gads_tenancy_"))
WS_ROOT, DS_ROOT = TMP / "workspaces", TMP / "datasets"
WS_ROOT.mkdir(); DS_ROOT.mkdir()

# Scratch DB = the research DB's server and credentials, different database name.
_real = os.environ.get("GADS_DATABASE_URL") or dotenv_values(ROOT / ".env").get("GADS_DATABASE_URL", "")
if not _real:
    sys.exit("GADS_DATABASE_URL not found (env or .env)")
SCRATCH_URL = _real.rsplit("/", 1)[0] + "/gads_tenancy_test"
assert SCRATCH_URL != _real
os.environ.update({
    "GADS_DATABASE_URL": SCRATCH_URL, "GADS_AUTH": "dev_header", "GADS_DEPLOYMENT": "local",
    "GADS_ADMIN_EMAIL": "admin@test.local", "GADS_WORKSPACE_ROOT": str(WS_ROOT),
    "GADS_DATASETS_ROOT": str(DS_ROOT), "GADS_OTEL_ENDPOINT": "",
    "GADS_ROUTING_MODE": "local", "GADS_DISTILL_CAPTURE": "false",
    # A dead sandbox: handlers that probe (register-external's DataAnalyzer) must never reach
    # the real one, which would create kernels and workspace dirs for these test projects.
    "GADS_SANDBOX_URL": "http://127.0.0.1:9",
})

from sqlmodel import SQLModel, Session, select  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from gads.core.database import engine, init_db  # noqa: E402
from gads.core import models  # noqa: E402,F401  (registers tables)
from gads.core.models import Project, Task, User  # noqa: E402
from gads.core.execution_hub import ExecutionHub  # noqa: E402
from gads.core import registry, tenancy  # noqa: E402
import gads.core.server as server  # noqa: E402

FAILS = []


def check(cond, label):
    print(f"  {'✓' if cond else '✗'} {label}")
    if not cond:
        FAILS.append(label)


assert "gads_tenancy_test" in str(engine.url), engine.url
SQLModel.metadata.drop_all(engine)
init_db()

ADMIN, A, B = "admin@test.local", "alice@test.local", "bob@test.local"
H = lambda email: {"X-GADS-User": email}  # noqa: E731
client = TestClient(server.app)  # no context manager: startup loops (dispatcher, watchdog) stay off

print("1. route policy table covers every route (default deny)")
keys = {k for r in server.app.routes if hasattr(r, "path") and getattr(r, "endpoint", None)
        and not r.path.startswith(("/docs", "/openapi", "/redoc")) for k in tenancy.route_key(r)}
check(not (keys - set(tenancy.ROUTE_POLICY)), f"no unlisted routes: {sorted(keys - set(tenancy.ROUTE_POLICY))}")
check(not (set(tenancy.ROUTE_POLICY) - keys), f"no stale policies: {sorted(set(tenancy.ROUTE_POLICY) - keys)}")
try:
    tenancy.check_policy(tenancy.admin_user(), None, {})
    check(False, "unlisted policy denied")
except Exception as e:
    check(getattr(e, "status_code", None) == 403, "unlisted policy denied (403) even for admin")

print("2. identity")
check(client.get("/health").status_code == 200, "health is public")
check(client.get("/projects").status_code == 401, "no X-GADS-User -> 401")
me = client.get("/me", headers=H(A)).json()
check(me.get("role") == "tester" and me.get("email") == A, "first sight creates a tester")
check(client.get("/me", headers=H(ADMIN)).json().get("role") == "admin", "GADS_ADMIN_EMAIL is admin")

print("3. projects")
def make_project(email, name):
    r = client.post("/projects", headers=H(email), json={"name": name, "objective": ""})
    return r.json()["project"]["id"]
pa, pb = make_project(A, "alice-proj"), make_project(B, "bob-proj")
with Session(engine) as s:
    uid = {u.email: u.id for u in s.exec(select(User)).all()}
    check(str(s.get(Project, uuid.UUID(pa)).owner_id) == str(uid[A]), "new project owned by its creator")
la = {p["id"] for p in client.get("/projects", headers=H(A)).json()}
check(la == {pa}, "list shows only own projects")
check({pa, pb} <= {p["id"] for p in client.get("/projects", headers=H(ADMIN)).json()}, "admin lists all")
for method, path in [("get", f"/projects/{pb}"), ("get", f"/projects/{pb}/notes"),
                     ("post", f"/projects/{pb}/cancel"), ("get", f"/projects/{pb}/kernel"),
                     ("delete", f"/projects/{pb}"), ("post", f"/projects/{pb}/finish")]:
    r = getattr(client, method)(path, headers=H(A))
    check(r.status_code == 404, f"{method.upper()} {path.replace(pb, '<bob>')} as alice -> 404 (got {r.status_code})")
check(client.get(f"/projects/{pb}", headers=H(ADMIN)).status_code == 200, "admin opens bob's project")
r = client.post("/projects", headers=H(A), json={"name": "x", "objective": "", "existing_project_id": pb})
check(r.status_code == 404, "resume someone else's project via existing_project_id -> 404")
check(client.get(f"/projects/{uuid.uuid4()}", headers=H(A)).status_code == 404, "unknown id -> 404 (no oracle)")
r = client.get("/taxonomy/runs", headers=H(A))
check(r.status_code == 200 and all(x["project_id"] == pa for x in r.json()), "taxonomy/runs filtered")

print("4. admin-only routes")
for method, path, body in [("post", "/config", {"routing_mode": "cloud"}), ("get", "/specs", None),
                           ("post", "/projects/from-spec", {"filename": "x.md"}),
                           ("post", "/maintenance/cleanup", None),
                           ("post", f"/projects/{pa}/register-external", {"path": str(DS_ROOT / 'd.csv')}),
                           ("get", "/knowledge/recipes/x/evidence", None)]:
    r = getattr(client, method)(path, headers=H(A), **({"json": body} if body is not None else {}))
    check(r.status_code == 403, f"{method.upper()} {path.replace(pa, '<alice>')} as tester -> 403 (got {r.status_code})")
check(client.get("/recipes", headers=H(A)).status_code == 200, "knowledge reads are open")

print("5. streams")
with Session(engine) as s:
    tb = Task(project_id=uuid.UUID(pb), description="bob secret task", assigned_to="x")
    ta = Task(project_id=uuid.UUID(pa), description="alice task", assigned_to="x")
    s.add(tb); s.add(ta); s.commit(); s.refresh(tb); s.refresh(ta)
    tb_id, ta_id = str(tb.id), str(ta.id)
check(client.get(f"/tasks/{tb_id}/stream", headers=H(A)).status_code == 404, "other user's task stream -> 404")
check(client.get(f"/tasks/{ta_id}/stream", headers=H(A)).status_code == 200, "own task stream -> 200")

print("6. /ws replay + broadcast filtering")
with Session(engine) as s:
    hub = ExecutionHub(s)
    for ev, payload in [("SEED", {}), ("TASK_CREATED", {"task_id": ta_id, "description": "alice task"}),
                        ("TASK_CREATED", {"task_id": tb_id, "description": "bob secret task"}),
                        ("STATE_UPDATED", {"project_id": pb, "state": {}}),
                        ("STATE_UPDATED", {"project_id": pa, "state": {}}),
                        ("WORKFLOW_FINAL_RESULT", {"narrative": "unattributed"})]:
        hub.create_outbox_event(ev, payload); s.commit()
with client.websocket_connect("/ws?last_seq=1", headers=H(A)) as ws:
    got = [ws.receive_json(), ws.receive_json()]
seen = [g["payload"].get("task_id") or g["payload"].get("project_id") for g in got]
check(seen == [ta_id, pa], f"alice replay = her task + her project only (got {seen})")
with client.websocket_connect("/ws?last_seq=1", headers=H(ADMIN)) as ws:
    got = [ws.receive_json() for _ in range(5)]
check(len(got) == 5, "admin replay gets everything incl. unattributed")
try:
    with client.websocket_connect("/ws?last_seq=1") as ws:
        ws.receive_json()
    check(False, "/ws without identity refused")
except Exception as e:
    check(type(e).__name__ in ("WebSocketDisconnect", "WebSocketDenialResponse") or "401" in str(e) or "403" in str(e),
          f"/ws without identity refused ({type(e).__name__})")


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send_json(self, m):
        self.sent.append(m)


async def _broadcast():
    from gads.core.bus import EventBus
    bus = EventBus()
    wa, wb, wanon = FakeWS(), FakeWS(), FakeWS()
    bus.active_connections = {wa: tenancy.get_or_create_user(A), wb: tenancy.get_or_create_user(B), wanon: None}
    await bus.broadcast({"seq": 99, "type": "TASK_COMPLETED", "payload": {"task_id": tb_id}})
    return wa.sent, wb.sent, wanon.sent
sa, sb, sn = asyncio.run(_broadcast())
check(not sa and len(sb) == 1 and not sn, "broadcast of bob's event reaches only bob")

print("7. files: symlinks and roots")
wa_dir, wb_dir = WS_ROOT / pa, WS_ROOT / pb
wa_dir.mkdir(exist_ok=True); wb_dir.mkdir(exist_ok=True)
(wa_dir / "ok.txt").write_text("mine")
(wb_dir / "secret.txt").write_text("bob")
(DS_ROOT / "d.csv").write_text("a,b\n1,2\n")
os.symlink(wb_dir / "secret.txt", wa_dir / "evil.txt")
os.symlink(DS_ROOT / "d.csv", wa_dir / "d.csv")
outside = TMP / "outside.txt"; outside.write_text("host file")
os.symlink(outside, wa_dir / "host.txt")
evil_sibling = WS_ROOT / f"{pa}-evil"; evil_sibling.mkdir(); (evil_sibling / "x.txt").write_text("x")
def dl(email, pid, f):
    return client.get(f"/projects/{pid}/files/{f}", headers=H(email)).status_code
check(dl(A, pa, "ok.txt") == 200, "own file 200")
check(dl(A, pa, "d.csv") == 200, "dataset symlink (datasets root) 200")
check(dl(A, pa, "evil.txt") == 403, "symlink into bob's workspace 403")
check(dl(A, pa, "host.txt") == 403, "symlink to an arbitrary host file 403")
check(dl(A, pa, f"../{pa}-evil/x.txt") in (403, 404), "sibling-prefix traversal denied")
check(dl(A, pb, "secret.txt") == 404, "bob's file via bob's project 404")
check(dl(ADMIN, pa, "evil.txt") == 200, "admin may follow a link into a workspace it owns (all)")
r = client.post(f"/projects/{pa}/register-external", headers=H(ADMIN), json={"path": str(outside)})
check(r.status_code == 403, "register-external outside datasets root -> 403 even for admin")
r = client.post(f"/projects/{pa}/register-external", headers=H(ADMIN), json={"path": str(wb_dir / 'secret.txt')})
check(r.status_code == 403, "register-external of another workspace -> 403")
r = client.post(f"/projects/{pa}/register-external", headers=H(ADMIN), json={"path": str(DS_ROOT / 'd.csv')})
check(r.status_code == 200, f"register-external inside datasets root -> 200 ({r.status_code})")

print("8. per-user settings")
r = client.put("/me/settings", headers=H(A), json={"routing_mode": "local"})
check(r.status_code == 400, "tester cannot choose local routing")
r = client.put("/me/settings", headers=H(A), json={"pinned_model": "local_model", "routing_mode": "cloud_pinned"})
check(r.status_code == 400, "tester cannot pin local_model")
r = client.put("/me/settings", headers=H(A), json={"routing_mode": "cloud_pinned", "pinned_model": "gemini-3.8-flash"})
check(r.status_code == 200 and r.json()["effective"]["routing_mode"] == "cloud_pinned", f"tester pins a cloud model ({r.status_code} {r.text[:300]})")
eff_b = client.get("/me/settings", headers=H(B)).json()["effective"]
check(eff_b["routing_mode"] == "cloud", f"tester never inherits the global 'local' default (got {eff_b['routing_mode']})")
check(registry.get_routing_mode() == "local", "global layer untouched by user settings")
check(server.run_settings_for(uuid.UUID(pa))["pinned_model"] == "gemini-3.8-flash", "run snapshot = owner's settings")
r = client.put("/me/settings", headers=H(A), json={"pinned_model": None, "routing_mode": "cloud"})
check(r.status_code == 200 and "pinned_model" not in r.json()["overrides"], "null clears an override")
r = client.put("/me/settings", headers=H(ADMIN), json={"routing_mode": "hybrid"})
check(r.status_code == 200, "admin may choose hybrid")

print("9. concurrent runs keep their own settings")
async def run(settings, stage, out, key):
    with registry.run_settings_scope(settings):
        await asyncio.sleep(0.01)
        a = registry.resolve_stage_model(stage, "tier-default")
        await asyncio.sleep(0.02)
        out[key] = (a, registry.resolve_stage_model(stage, "tier-default"),
                    await asyncio.to_thread(registry.get_routing_mode))
async def both():
    out = {}
    g = registry.global_settings()
    await asyncio.gather(
        run({**g, "routing_mode": "cloud_pinned", "pinned_model": "pinned-x"}, "Coder", out, "a"),
        run({**g, "routing_mode": "local"}, "Coder", out, "b"),
        run({**g, "routing_mode": "cloud"}, "Coder", out, "c"))
    return out
out = asyncio.run(both())
check(out["a"] == ("pinned-x", "pinned-x", "cloud_pinned"), f"run A pinned throughout {out['a']}")
check(out["b"] == ("local_model", "local_model", "local"), f"run B local throughout {out['b']}")
check(out["c"] == ("tier-default", "tier-default", "cloud"), f"run C cloud throughout {out['c']}")
check(registry.current_run_settings() is None and registry.get_routing_mode() == "local", "no leak outside the scope")
with registry.run_settings_scope({**registry.global_settings(), "routing_mode": "cloud"}):
    registry.set_routing_mode("hybrid")  # a POST /config mid-run
    check(registry.get_routing_mode() == "cloud", "mid-run global change does not reach a running workflow")
registry.set_routing_mode("local")

print("9b. workflow wrapper: each run under its owner's snapshot, global flip mid-run ignored")
async def wrapper_runs():
    seen = {}
    real = server.run_agent_workflow
    async def stub(project_id, objective, instruction_id=None):
        first = registry.resolve_stage_model("Coder", "tier-default")
        await asyncio.sleep(0.05)
        seen[str(project_id)] = (first, registry.resolve_stage_model("Coder", "tier-default"))
    server.run_agent_workflow = stub
    try:
        p_admin = make_project(ADMIN, "admin-proj")
        client.put("/me/settings", headers=H(A), json={"routing_mode": "cloud_pinned", "pinned_model": "pinned-alice"})
        client.put("/me/settings", headers=H(ADMIN), json={"routing_mode": None})
        async def flip():
            await asyncio.sleep(0.02)
            registry.set_routing_mode("hybrid")
        await asyncio.gather(*(server.run_agent_workflow_wrapper(uuid.UUID(p), "x") for p in (pa, pb, p_admin)), flip())
        return seen, p_admin
    finally:
        server.run_agent_workflow = real
        registry.set_routing_mode("local")
seen, p_admin = asyncio.run(wrapper_runs())
check(seen[pa] == ("pinned-alice", "pinned-alice"), f"alice's run pinned throughout {seen[pa]}")
check(seen[pb] == ("tier-default", "tier-default"), f"bob's run cloud throughout {seen[pb]}")
check(seen[p_admin] == ("local_model", "local_model"), f"admin run keeps launch-time 'local' despite the flip {seen[p_admin]}")

print("10. hosted guard (the backend refuses to boot)")
env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "GADS_DEPLOYMENT": "hosted"}
for mode in ("single_user", "dev_header"):
    r = subprocess.run([sys.executable, "-c", "import gads.core.server"],
                       env={**env, "GADS_AUTH": mode}, capture_output=True, text=True, cwd=TMP, timeout=300)
    check(r.returncode != 0 and "GADS_DEPLOYMENT=hosted" in r.stderr, f"server import refuses GADS_AUTH={mode}")

shutil.rmtree(TMP, ignore_errors=True)
SQLModel.metadata.drop_all(engine)
print("\nall checks passed" if not FAILS else f"\n{len(FAILS)} check(s) FAILED")
sys.exit(1 if FAILS else 0)
