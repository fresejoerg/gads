"""Regression guard for the isolated (no shared disk) sandbox backend (HOSTED-3a). No LLM calls.

Needs the STAND-IN sandbox: the production sandbox image with no workspace or dataset mount,
started with a SANDBOX_TOKEN, e.g.

    docker run -d --name sandbox-isolated --user 1000:1000 -p 127.0.0.1:8010:8000 \\
        -e SANDBOX_TOKEN="$(cat token)" -e WORKSPACE_ROOT=/tmp/workspaces mylocalstack-sandbox:hosted3a
    GADS_TEST_SANDBOX_URL=http://localhost:8010 GADS_TEST_SANDBOX_TOKEN_FILE=token \\
        PYTHONPATH=src uv run python scripts/test_sandbox_sync.py

Skips (exit 0) when the stand-in is not reachable. Workspaces and datasets live in a temp dir.
Covers: auth (and a wrong token becoming the ConnectionError result) · push/pull round trip ·
a remote rewrite of a pushed dataset never writes through the local link (datasets root hash
unchanged) · links to outside roots not pushed · idempotence (nothing re-sent) · sandbox restart
resets the sync record · remote symlinks neither listed nor served · path/workspace validation ·
a cancelled transfer leaves no partial file and the next call completes · the token is invisible
to task code.
"""
import asyncio
import hashlib
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

URL = os.environ.get("GADS_TEST_SANDBOX_URL", "http://localhost:8010").rstrip("/")
TOKEN = Path(os.environ["GADS_TEST_SANDBOX_TOKEN_FILE"]).read_text().strip() \
    if os.environ.get("GADS_TEST_SANDBOX_TOKEN_FILE") else ""
CONTAINER = os.environ.get("GADS_TEST_SANDBOX_CONTAINER", "sandbox-isolated")
TMP = Path(tempfile.mkdtemp(prefix="gads_sync_"))
WS, DS, OUT = TMP / "ws", TMP / "ds", TMP / "outside"
for d in (WS, DS, OUT):
    d.mkdir()
os.environ.update({"GADS_WORKSPACE_ROOT": str(WS), "GADS_DATASETS_ROOT": str(DS),
                   "GADS_SANDBOX_URL": URL, "GADS_SANDBOX_TOKEN": TOKEN,
                   "GADS_SANDBOX_BACKEND": "isolated_http", "GADS_OTEL_ENDPOINT": ""})

import httpx  # noqa: E402

try:
    h = httpx.get(f"{URL}/health", timeout=3).json()
except Exception:
    print(f"(stand-in sandbox not reachable at {URL}; skipped)")
    sys.exit(0)

from gads.tools import sandbox_sync  # noqa: E402
from gads.tools.sandbox import make_sandbox  # noqa: E402

FAILS = []


def check(cond, label):
    print(f"  {'✓' if cond else '✗'} {label}")
    if not cond:
        FAILS.append(label)


def tree_hash(root: Path) -> str:
    hsh = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            hsh.update(str(p.relative_to(root)).encode()); hsh.update(p.read_bytes())
    return hsh.hexdigest()


auth = {"Authorization": f"Bearer {TOKEN}"}
pid = str(uuid.uuid4())
local = WS / pid
local.mkdir()


async def run(code, key=pid, sb=None):
    sb = sb or make_sandbox()
    try:
        return await sb.execute(code, project_id=uuid.UUID(key), session_id=key)
    finally:
        await sb.close()


print("1. auth")
check(h.get("auth") is True, "stand-in requires a token")
check(httpx.get(f"{URL}/files/{pid}", timeout=10).status_code == 401, "no token -> 401")
check(httpx.get(f"{URL}/files/{pid}", headers={"Authorization": "Bearer wrong"}, timeout=10).status_code == 401,
      "wrong token -> 401")
os.environ["GADS_SANDBOX_TOKEN"] = "wrong"
r = asyncio.run(run("print(1)"))
check((r.error or {}).get("ename") == "ConnectionError" and "push" in (r.error or {}).get("evalue", ""),
      "wrong token: execute returns the ConnectionError result, no code ran")
os.environ["GADS_SANDBOX_TOKEN"] = TOKEN

print("2. push / run / pull")
(local / "input.csv").write_text("a,b\n1,2\n3,4\n")
(DS / "data.csv").write_text("x\n10\n20\n")
os.symlink(DS / "data.csv", local / "data.csv")                 # mounted dataset (link)
(OUT / "secret.txt").write_text("host secret")
os.symlink(OUT / "secret.txt", local / "host.txt")               # link to a disallowed root
up = WS / str(uuid.uuid4()); up.mkdir(); (up / "model.json").write_text('{"up": 1}')
(local / "upstream").mkdir(); os.symlink(f"../../{up.name}/model.json", local / "upstream" / "model.json")
ds_before = tree_hash(DS)
code = """
import pandas as pd, os, json
df = pd.read_csv('input.csv'); d = pd.read_csv('data.csv')
print('rows', len(df), len(d), json.load(open('upstream/model.json'))['up'])
df.assign(c=df.a + df.b).to_csv('out.csv', index=False)
os.makedirs('sub', exist_ok=True); json.dump({'ok': True}, open('sub/plot.json', 'w'))
try:                                                   # task code tries to rewrite the dataset
    open('data.csv', 'w').write('x\\nTAMPERED\\n'); print('dataset write: ALLOWED')
except PermissionError:
    print('dataset write: PermissionError')
print('host.txt remote:', os.path.exists('host.txt'))
"""
r = asyncio.run(run(code))
check(r.error is None and "rows 2 2 1" in r.stdout, f"code ran on pushed inputs ({r.stdout.strip()[:80]} {r.error})")
check("host.txt remote: False" in r.stdout, "link to a disallowed root was not pushed")
check((local / "out.csv").read_text().splitlines()[1] == "1,2,3" and
      (local / "sub" / "plot.json").exists(), "outputs pulled back (incl. a new subdirectory)")
check("dataset write: PermissionError" in r.stdout,
      "the pushed dataset copy is read-only remotely (parity with the local :ro mount)")
check(os.path.islink(local / "data.csv") and tree_hash(DS) == ds_before, "datasets root unchanged, local link intact")
r2 = asyncio.run(run("import os, stat; os.chmod('data.csv', 0o644); open('data.csv','w').write('x\\nTAMPERED\\n'); print('forced')"))
check("forced" in r2.stdout and os.path.islink(local / "data.csv") and tree_hash(DS) == ds_before
      and r2.sync_stats.get("pull_skipped_links") == 1,
      f"even a forced remote rewrite never writes through the local link ({r2.sync_stats})")
(other := WS / str(uuid.uuid4())).mkdir(); (other / "theirs.txt").write_text("someone else's")
os.symlink(other / "theirs.txt", local / "stray.txt")          # cross-workspace link, NOT under upstream/
r3 = asyncio.run(run("import os; print('stray remote:', os.path.exists('stray.txt'), 'upstream:', os.path.exists('upstream/model.json'))"))
check("stray remote: False upstream: True" in r3.stdout, "cross-workspace links are pushed only from upstream/")
(local / "stray.txt").unlink()
check(not list(local.rglob(".sync_*")), "no temp files left locally")

print("3. idempotence + change detection")
r = asyncio.run(run("print('noop')"))
check(r.sync_stats.get("push_files") == 0 and r.sync_stats.get("pull_files") == 0,
      f"nothing re-sent when nothing changed ({r.sync_stats})")
(local / "input.csv").write_text("a,b\n5,6\n")
r = asyncio.run(run("print(open('input.csv').read().splitlines()[1])"))
check(r.sync_stats.get("push_files") == 1 and "5,6" in r.stdout, "a changed local file is pushed again")

print("4. remote symlinks and path validation")
r = asyncio.run(run("import os; os.symlink('/etc/passwd', 'pw.txt'); print('linked')"))
listing = httpx.get(f"{URL}/files/{pid}", headers=auth, timeout=10).json()
check("pw.txt" not in [f["path"] for f in listing["files"]] and not (local / "pw.txt").exists(),
      "a symlink created remotely is neither listed nor pulled")
check(httpx.get(f"{URL}/files/{pid}/pw.txt", headers=auth, timeout=10).status_code == 403, "...nor served")
# httpx normalises a literal "../" away client-side, so send it percent-encoded.
for raw in (f"/files/{pid}/%2E%2E/escape.txt", f"/files/{pid}/sub/%2E%2E/%2E%2E/escape.txt"):
    code_ = httpx.put(URL + raw, headers=auth, content=b"x", timeout=10).status_code
    check(code_ != 200, f"PUT {raw.replace(pid, '<ws>')} refused ({code_})")
leak = subprocess.run(["docker", "exec", CONTAINER, "sh", "-c", "ls /tmp/workspaces/escape.txt /tmp/escape.txt 2>/dev/null"],
                      capture_output=True, text=True).stdout.strip()
check(not leak, "nothing was written outside the workspace")
check(httpx.get(f"{URL}/files/bad%20id", headers=auth, timeout=10).status_code == 400, "invalid workspace id refused")

print("5. a cancelled transfer leaves nothing half-done")
big = local / "big.bin"
big.write_bytes(os.urandom(64 * 1024 * 1024))


async def cancelled_then_complete():
    try:
        await asyncio.wait_for(run("print('big')"), timeout=0.05)
        return "not cancelled"
    except asyncio.TimeoutError:
        pass
    await asyncio.sleep(0.5)
    mid = httpx.get(f"{URL}/files/{pid}", headers=auth, timeout=10).json()
    names = [f["path"] for f in mid["files"]]
    partial = [n for n in names if n == "big.bin" and
               next(f["size"] for f in mid["files"] if f["path"] == n) != big.stat().st_size]
    # The listing hides .sync_* temp files, so look inside the container for leftovers.
    leftovers = subprocess.run(["docker", "exec", CONTAINER, "find", "/tmp/workspaces", "-name", ".sync_*"],
                               capture_output=True, text=True).stdout.split()
    partial += leftovers
    r2 = await run("import os; print('size', os.path.getsize('big.bin'))")
    return partial, r2
res = asyncio.run(cancelled_then_complete())
if res == "not cancelled":
    check(False, "transfer was cancelled (call finished in under 50 ms?)")
else:
    partial, r2 = res
    check(not partial, "no partial big.bin on the remote after the cancel")
    check(f"size {big.stat().st_size}" in r2.stdout, "the next call completed the push")
big.unlink()

print("6. sandbox restart resets the sync record")
subprocess.run(["docker", "restart", CONTAINER], check=True, capture_output=True)
for _ in range(60):
    try:
        if httpx.get(f"{URL}/health", timeout=2).status_code == 200:
            break
    except Exception:
        pass
    asyncio.run(asyncio.sleep(1))
r = asyncio.run(run("print(open('input.csv').read().splitlines()[1])"))
check(r.error is None and "5,6" in r.stdout and r.sync_stats.get("push_files", 0) >= 2,
      f"after a restart the workspace is re-pushed ({r.sync_stats})")

print("7. the token is invisible to task code")
probe = f"""
import os, glob
tok = {TOKEN!r}
hits = [p for p in glob.glob('/proc/[0-9]*/environ') + glob.glob('/proc/[0-9]*/cmdline')
        if (lambda q: (lambda b: tok.encode() in b)(open(q, 'rb').read()) if os.access(q, os.R_OK) else False)(p)]
print('hits', hits, os.environ.get('SANDBOX_TOKEN'), os.environ.get('SANDBOX_TOKEN_FD'))
"""
r = asyncio.run(run(probe))
check("hits [] None None" in r.stdout, f"no process environ/cmdline holds the token ({r.stdout.strip()[:120]})")

asyncio.run(make_sandbox().reset_session(pid))
import shutil  # noqa: E402
shutil.rmtree(TMP, ignore_errors=True)
print("\nall checks passed" if not FAILS else f"\n{len(FAILS)} check(s) FAILED")
sys.exit(1 if FAILS else 0)
