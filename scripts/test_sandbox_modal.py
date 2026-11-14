"""Regression guard for the Modal sandbox backend (HOSTED-3c). No LLM calls; costs about a cent.

    PYTHONPATH=src uv run python scripts/test_sandbox_modal.py

Needs Modal credentials (~/.modal.toml), GADS_MODAL_IMAGE_ID and GADS_SECRET_KEY (both from .env);
skips (exit 0) without them. Starts 2-3 sandboxes and terminates everything it started.
Covers: nothing but execute creates a sandbox · a create cancelled by its caller still completes,
and concurrent callers share one sandbox · probe_<pid> and <pid> share the sandbox · tunnel auth ·
the 3a sync rules over a real network (push/pull, read-only dataset copy, a forced remote rewrite
never written through the local link, disallowed and cross-workspace links not pushed,
idempotence) · token invisible to task code · a backend restart reattaches (same sandbox, kernel
state kept) · a sandbox that died is replaced and the workspace re-pushed.
"""
import asyncio
import hashlib
import os
import sys
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
TMP = Path(tempfile.mkdtemp(prefix="gads_modal_"))
WS, DS, OUT = TMP / "ws", TMP / "ds", TMP / "outside"
for d in (WS, DS, OUT):
    d.mkdir()
os.environ.update({"GADS_WORKSPACE_ROOT": str(WS), "GADS_DATASETS_ROOT": str(DS),
                   "GADS_SANDBOX_BACKEND": "modal", "GADS_OTEL_ENDPOINT": "",
                   "GADS_MODAL_IDLE_S": "300", "GADS_MODAL_MAX_LIFETIME_S": "1800"})

if not (os.environ.get("GADS_MODAL_IMAGE_ID") and os.environ.get("GADS_SECRET_KEY")
        and Path("~/.modal.toml").expanduser().exists()):
    print("(GADS_MODAL_IMAGE_ID, GADS_SECRET_KEY or Modal credentials missing; skipped)")
    sys.exit(0)

import httpx  # noqa: E402
import modal  # noqa: E402

from gads.core import config  # noqa: E402
from gads.tools import sandbox_modal as sm  # noqa: E402
from gads.tools.sandbox import make_sandbox  # noqa: E402

FAILS = []


def check(cond, label):
    print(f"  {'✓' if cond else '✗'} {label}", flush=True)
    if not cond:
        FAILS.append(label)


def tree_hash(root: Path) -> str:
    hsh = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            hsh.update(str(p.relative_to(root)).encode()); hsh.update(p.read_bytes())
    return hsh.hexdigest()


async def running_for(pid):
    app = await sm._app()
    return [sb.object_id async for sb in modal.Sandbox.list.aio(app_id=app.app_id, tags={"project": pid})]


async def run(code, key, session=None, workspace_id=None):
    sb = make_sandbox()
    try:
        return await sb.execute(code, project_id=uuid.UUID(key), session_id=session or key,
                                workspace_id=workspace_id)
    finally:
        await sb.close()


async def main():
    pid = str(uuid.uuid4())
    local = WS / pid
    local.mkdir()
    (local / "input.csv").write_text("a,b\n1,2\n3,4\n")
    started = [pid]
    try:
        print("1. only execute creates a sandbox")
        b = make_sandbox()
        await b.health()
        await b.reset_session(pid)
        check(await b.poll_logs(pid) is None, "poll_logs on a project without a sandbox is a no-op")
        await b.close()
        check(await running_for(pid) == [], "health / reset_session / poll_logs started nothing")

        print("2. a cancelled create still completes; concurrent callers share one sandbox")
        try:
            await asyncio.wait_for(run("print('early')", pid), timeout=2.0)
            check(False, "the first call was expected to time out during create")
        except asyncio.TimeoutError:
            pass
        r1, r2 = await asyncio.gather(run("x = 41; print('a')", pid), run("print('b')", pid))
        ids = await running_for(pid)
        check(len(ids) == 1, f"exactly one sandbox for the project ({ids})")
        check(r1.error is None and r2.error is None and "a" in r1.stdout and "b" in r2.stdout,
              f"both concurrent calls ran ({r1.error} {r2.error})")
        entry = sm._SANDBOXES[pid]

        print("3. probe_<pid> and <pid> share the sandbox")
        r = await run("import os; print('probe sees', os.path.exists('input.csv'))", pid,
                      session=f"probe_{pid}", workspace_id=pid)
        check(r.error is None and "probe sees True" in r.stdout and await running_for(pid) == ids,
              f"probe session ran in the same sandbox on the same workspace ({r.stdout.strip()} {r.error})")

        print("4. tunnel auth")
        url = entry["url"]
        async with httpx.AsyncClient(timeout=20) as c:
            check((await c.get(f"{url}/files/{pid}")).status_code == 401, "no token -> 401")
            check((await c.get(f"{url}/files/{pid}", headers={"Authorization": "Bearer x"})).status_code == 401,
                  "wrong token -> 401")
            check((await c.get(f"{url}/health")).json().get("auth") is True, "service reports auth on")

        print("5. sync rules over the tunnel")
        (DS / "data.csv").write_text("x\n10\n20\n")
        os.symlink(DS / "data.csv", local / "data.csv")
        (OUT / "secret.txt").write_text("host secret")
        os.symlink(OUT / "secret.txt", local / "host.txt")
        up = WS / str(uuid.uuid4()); up.mkdir(); (up / "model.json").write_text('{"up": 1}')
        (local / "upstream").mkdir(); os.symlink(f"../../{up.name}/model.json", local / "upstream" / "model.json")
        (other := WS / str(uuid.uuid4())).mkdir(); (other / "theirs.txt").write_text("someone else's")
        os.symlink(other / "theirs.txt", local / "stray.txt")
        ds_before = tree_hash(DS)
        code = """
import pandas as pd, os, json
df = pd.read_csv('input.csv'); d = pd.read_csv('data.csv')
print('rows', len(df), len(d), json.load(open('upstream/model.json'))['up'], 'x', x)
df.assign(c=df.a + df.b).to_csv('out.csv', index=False)
os.makedirs('sub', exist_ok=True); json.dump({'ok': True}, open('sub/plot.json', 'w'))
try:
    open('data.csv', 'w').write('x\\nTAMPERED\\n'); print('dataset write: ALLOWED')
except PermissionError:
    print('dataset write: PermissionError')
print('host.txt remote:', os.path.exists('host.txt'), 'stray remote:', os.path.exists('stray.txt'))
"""
        r = await run(code, pid)
        check(r.error is None and "rows 2 2 1 x 41" in r.stdout,
              f"code ran on pushed inputs in the live kernel ({r.stdout.strip()[:80]} {r.error})")
        check("host.txt remote: False stray remote: False" in r.stdout,
              "disallowed and cross-workspace links were not pushed")
        check((local / "out.csv").read_text().splitlines()[1] == "1,2,3" and (local / "sub" / "plot.json").exists(),
              "outputs pulled back (incl. a new subdirectory)")
        check("dataset write: PermissionError" in r.stdout, "the pushed dataset copy is read-only remotely")
        r = await run("import os; os.chmod('data.csv', 0o644); open('data.csv','w').write('T\\n'); print('forced')", pid)
        check("forced" in r.stdout and os.path.islink(local / "data.csv") and tree_hash(DS) == ds_before
              and r.sync_stats.get("pull_skipped_links") == 1,
              f"a forced remote rewrite never writes through the local link ({r.sync_stats})")
        r = await run("print('noop')", pid)
        check(r.sync_stats.get("push_files") == 0 and r.sync_stats.get("pull_files") == 0,
              f"nothing re-sent when nothing changed ({r.sync_stats})")

        print("6. the token is invisible to task code")
        tok = entry["headers"]["Authorization"].split()[1]
        probe = f"""
import os, glob
hits = []
for q in glob.glob('/proc/[0-9]*/environ') + glob.glob('/proc/[0-9]*/cmdline'):
    try:
        if {tok!r}.encode() in open(q, 'rb').read(): hits.append(q)
    except OSError:
        pass
print('hits', hits, os.environ.get('SANDBOX_TOKEN'), os.getuid())
"""
        r = await run(probe, pid)
        check("hits [] None 1000" in r.stdout, f"no readable process environ/cmdline holds the token ({r.stdout.strip()[:120]})")

        print("6b. live logs and session reset reach the live sandbox")
        poller = make_sandbox()
        job = asyncio.ensure_future(run("import time\nfor i in range(6):\n    print('tick', i, flush=True); time.sleep(1)", pid))
        seen = ""
        for _ in range(20):
            await asyncio.sleep(1)
            logs = await poller.poll_logs(pid)
            seen = (logs or {}).get("stdout", "") or seen
            if "tick 2" in seen and not job.done():
                break
        await job
        check("tick 2" in seen and not seen.endswith("tick 5\n"),
              f"poll_logs streamed stdout while the cell was still running ({seen.strip()[-40:]!r})")
        await run("y = 7", pid)
        await poller.reset_session(pid)
        await poller.close()
        r = await run("print('y gone', 'y' not in dir())", pid)
        check("y gone True" in r.stdout and await running_for(pid) == ids,
              f"reset_session cleared the kernel, same sandbox ({r.stdout.strip()} {r.error})")
        r = await run("x = 41", pid)  # restore the variable step 7 checks

        print("7. a backend restart reattaches")
        sm.reset_registry()
        r = await run("print('x still', x)", pid)
        check(r.error is None and "x still 41" in r.stdout and await running_for(pid) == ids,
              f"same sandbox found again, kernel state kept ({r.stdout.strip()} {r.error})")

        print("8. a dead sandbox is replaced")
        await sm._SANDBOXES[pid]["sb"].terminate.aio()
        r = await run("print('fresh', 'x' in dir(), open('input.csv').read().count(chr(10)))", pid)
        for _ in range(30):  # the listing lags a terminate by a few seconds
            new_ids = await running_for(pid)
            if ids[0] not in new_ids:
                break
            await asyncio.sleep(1)
        check(r.error is None and "fresh False 3" in r.stdout and len(new_ids) == 1 and new_ids != ids,
              f"new sandbox, empty kernel, workspace re-pushed ({r.stdout.strip()} {r.error} {r.sync_stats})")
    finally:
        for p in started:
            await sm.terminate(p)
        for _ in range(30):  # the listing lags a terminate by a few seconds
            if await running_for(pid) == []:
                break
            await asyncio.sleep(1)
        check(await running_for(pid) == [], "everything this test started is terminated")


asyncio.run(main())
import shutil  # noqa: E402

shutil.rmtree(TMP, ignore_errors=True)
print("\nall checks passed" if not FAILS else f"\n{len(FAILS)} check(s) FAILED")
sys.exit(1 if FAILS else 0)
