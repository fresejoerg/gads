"""One Modal sandbox per project (HOSTED-3c, 032 phase 3). Only the sandbox moves; the backend stays.

`ModalBackend` is `IsolatedHttpBackend` (push -> run -> pull over HTTP, 3a) plus a per-project
sandbox started from the pinned image (3b) and reached through Modal's encrypted tunnel. The
3a seam does the routing: `endpoint(key)` returns that project's tunnel URL and bearer token.

Rules (spike findings and the call sites they protect, approach_docs/CURRENT_SPRINT.md HOSTED-3c):
  * ONLY `execute` CREATES A SANDBOX. `_cleanup_stale_sessions` calls `reset_session` for every
    idle project, and `poll_logs`/`health` run all the time: on a project with no sandbox they
    are no-ops (endpoint raises LookupError, which the inherited methods swallow).
  * CREATION SURVIVES A CANCELLED CALLER. Callers wrap execute in `wait_for` (5-30 s). Creation
    runs as one shielded task per project that every caller awaits, so a cancel cannot leave a
    billed sandbox nobody knows about, and two callers cannot start two.
  * A SANDBOX IS FOUND AGAIN, NOT DUPLICATED. It is named after its project and tagged; the first
    use in a process reconciles the registry from one `Sandbox.list`. Its token is never stored:
    it is HMAC(GADS_SECRET_KEY, nonce) with the nonce in a tag, so it is recoverable after a
    backend restart and useless to anyone without the key.
  * A DEAD SANDBOX IS NOT A TASK FAILURE. Liveness is checked before every execute; a sandbox that
    exited (idle timeout, max lifetime, OOM: a hard memory limit kills the whole sandbox) is
    replaced, with a loud log that its kernel state is lost. The sync record resets with it.
  * NO KEEP-ALIVE CONNECTIONS. Modal's idle timeout does not fire while a client socket is open,
    and an idle pooled socket stays open; without this a sandbox would never stop.
  * Modal is only called through `.aio`: a blocking call would stall the dispatcher and /ws.

The service runs as uid 1000 via setpriv: Modal ignores the image's USER and runs as root.
Not yet: cost capture per sandbox, egress policy, explicit release when a project finishes (the
idle timeout stops it), the sandbox's own LLM access (the local proxy is not reachable from Modal).
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import re
import secrets
import time
import uuid
from typing import Dict, Optional

import httpx

from gads.core import config
from gads.tools.sandbox import ExecutionResult
from gads.tools.sandbox_sync import IsolatedHttpBackend, _conn_error

SERVICE_PORT = 8000
SERVICE_CMD = ("setpriv", "--reuid=1000", "--regid=1000", "--init-groups", "--", "sh", "-c",
               "cd /app && HOME=/home/sandboxuser exec uvicorn app:app --host 0.0.0.0 --port 8000")
# What the local compose sets and the image does not.
SERVICE_ENV = {"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
               "WORKSPACE_ROOT": "/app/workspaces"}
TAG = "gads"
_PID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

# Module level: make_sandbox() returns a fresh backend instance per call.
_SANDBOXES: Dict[str, dict] = {}   # project id -> {"sb", "url", "headers", "started"}
_CREATING: Dict[str, asyncio.Task] = {}
_RECONCILED = False
_RECONCILE_LOCK: Optional[asyncio.Lock] = None
_APP = None


def project_of(key: Optional[str]) -> Optional[str]:
    """The project a session / workspace key belongs to: `<pid>` and `probe_<pid>` share one sandbox."""
    m = _PID.search(key or "")
    return m.group(0).lower() if m else None


def token_for(nonce: str) -> str:
    key = config.secret_key()
    if not key:
        raise RuntimeError("GADS_SECRET_KEY is not set; the modal backend derives sandbox tokens from it")
    return hmac.new(key.encode(), b"gads-modal-sandbox:" + nonce.encode(), hashlib.sha256).hexdigest()


def sandbox_name(pid: str) -> str:
    return f"gads-{pid}"


def reset_registry() -> None:
    """Forget known sandboxes (tests). Does not terminate anything."""
    global _RECONCILED
    _SANDBOXES.clear()
    _CREATING.clear()
    _RECONCILED = False


async def _app():
    global _APP
    if _APP is None:
        import modal
        _APP = await modal.App.lookup.aio(config.modal_app(), create_if_missing=True)
    return _APP


async def _entry_for(sb, pid: str) -> dict:
    tags = await sb.get_tags.aio()
    tunnels = await sb.tunnels.aio()
    return {"sb": sb, "url": tunnels[SERVICE_PORT].url.rstrip("/"),
            "headers": {"Authorization": f"Bearer {token_for(tags['nonce'])}"},
            "started": time.time()}


async def _reconcile() -> None:
    """Once per process: register every running GADS sandbox of the app (backend restart)."""
    global _RECONCILED, _RECONCILE_LOCK
    if _RECONCILED:
        return
    if _RECONCILE_LOCK is None:
        _RECONCILE_LOCK = asyncio.Lock()
    async with _RECONCILE_LOCK:
        if _RECONCILED:
            return
        import modal
        app = await _app()
        async for sb in modal.Sandbox.list.aio(app_id=app.app_id, tags={TAG: "1"}):
            tags = await sb.get_tags.aio()
            pid = tags.get("project")
            if not pid or pid in _SANDBOXES or "nonce" not in tags:
                continue
            try:
                _SANDBOXES[pid] = await _entry_for(sb, pid)
                print(f"    [Modal] reattached sandbox {sb.object_id} for {pid[:8]}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"    [Modal] cannot reattach {sb.object_id} for {pid[:8]} ({e}); terminating it", flush=True)
                await sb.terminate.aio()
        _RECONCILED = True


async def _wait_healthy(entry: dict, deadline_s: float = 120.0) -> None:
    async with httpx.AsyncClient(timeout=10.0, limits=httpx.Limits(max_keepalive_connections=0)) as c:
        t0 = time.monotonic()
        while True:
            try:
                r = await c.get(f"{entry['url']}/health")
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if time.monotonic() - t0 > deadline_s:
                raise TimeoutError(f"sandbox service not healthy after {deadline_s:.0f}s")
            await asyncio.sleep(1.0)


async def _create(pid: str) -> dict:
    import modal
    image_id = config.modal_image_id()
    if not image_id:
        raise RuntimeError("GADS_MODAL_IMAGE_ID is not set (build it with scripts/modal_sandbox_image.py)")
    app = await _app()
    nonce = secrets.token_hex(16)
    t0 = time.monotonic()
    try:
        sb = await modal.Sandbox.create.aio(
            *SERVICE_CMD, app=app, image=modal.Image.from_id(image_id), name=sandbox_name(pid),
            cpu=config.modal_cpu(), memory=(config.modal_memory_mb(), config.modal_memory_mb()),
            timeout=config.modal_max_lifetime_s(), idle_timeout=config.modal_idle_s(),
            encrypted_ports=[SERVICE_PORT], env=SERVICE_ENV, tags={TAG: "1", "project": pid, "nonce": nonce},
            secrets=[modal.Secret.from_dict({"SANDBOX_TOKEN": token_for(nonce)})])
    except modal.exception.AlreadyExistsError:
        # Another process (or a create this process lost track of) owns it: attach.
        sb = await modal.Sandbox.from_name.aio(config.modal_app(), sandbox_name(pid))
        entry = await _entry_for(sb, pid)
        print(f"    [Modal] attached to existing sandbox {sb.object_id} for {pid[:8]}", flush=True)
        return entry
    entry = await _entry_for(sb, pid)
    await _wait_healthy(entry)
    print(f"    [Modal] started sandbox {sb.object_id} for {pid[:8]} in {time.monotonic() - t0:.1f}s "
          f"({config.modal_cpu():g} CPU, {config.modal_memory_mb()} MB)", flush=True)
    return entry


async def _create_and_register(pid: str) -> dict:
    try:
        entry = await _create(pid)
        _SANDBOXES[pid] = entry
        return entry
    finally:
        _CREATING.pop(pid, None)


async def ensure(pid: str) -> dict:
    """The project's live sandbox, started if there is none. Safe to cancel: creation continues."""
    await _reconcile()
    entry = _SANDBOXES.get(pid)
    if entry is not None:
        rc = await entry["sb"].poll.aio()
        if rc is None:
            return entry
        _SANDBOXES.pop(pid, None)
        print(f"    [Modal] !!! sandbox {entry['sb'].object_id} for {pid[:8]} EXITED (rc={rc}: idle timeout, "
              f"max lifetime or OOM). Its KERNEL STATE IS LOST; starting a new one.", flush=True)
    task = _CREATING.get(pid)
    if task is None:
        task = asyncio.ensure_future(_create_and_register(pid))
        _CREATING[pid] = task
    return await asyncio.shield(task)


async def terminate(pid: str) -> bool:
    """Stop the project's sandbox, if any (explicit cleanup; kernel state is lost)."""
    await _reconcile()
    entry = _SANDBOXES.pop(pid, None)
    if entry is None:
        return False
    await entry["sb"].terminate.aio()
    return True


class ModalBackend(IsolatedHttpBackend):
    """IsolatedHttpBackend whose endpoint is the project's own Modal sandbox."""

    def __init__(self, base_url: Optional[str] = None):
        super().__init__(base_url=base_url)
        # No keep-alive: an idle pooled socket keeps a Modal sandbox from ever going idle.
        self.client = httpx.AsyncClient(timeout=720.0, limits=httpx.Limits(max_keepalive_connections=0))

    def endpoint(self, key: Optional[str] = None) -> tuple:
        pid = project_of(key)
        entry = _SANDBOXES.get(pid) if pid else None
        if entry is None:
            raise LookupError(f"no Modal sandbox for {key!r}")
        return entry["url"], entry["headers"]

    async def health(self) -> None:
        """Modal reachable and configured. Creates nothing (there is no project here)."""
        if not config.modal_image_id():
            raise RuntimeError("GADS_MODAL_IMAGE_ID is not set")
        token_for("health")
        await _app()

    async def reset_session(self, session_id: str):
        try:
            await _reconcile()  # after a restart, still reset the sandbox the old process left
        except Exception:  # noqa: BLE001
            pass
        await super().reset_session(session_id)

    async def execute(self, code: str, project_id: uuid.UUID, session_id: str = "default",
                      workspace_id: Optional[str] = None) -> ExecutionResult:
        pid = project_of(workspace_id or session_id) or project_of(str(project_id))
        if pid is None:
            return _conn_error(f"modal backend: no project in session {session_id!r}")
        for attempt in (1, 2):
            try:
                await ensure(pid)
            except Exception as e:  # noqa: BLE001
                return _conn_error(f"Modal sandbox unavailable: {type(e).__name__}: {e}")
            res = await super().execute(code, project_id, session_id=session_id, workspace_id=workspace_id)
            err = res.error or {}
            if err.get("ename") != "ConnectionError" or not await _drop_if_dead(pid):
                return res
            # `poll` can still say "running" just after a sandbox died, so the call hit a dead tunnel.
            if "sync (push)" in err.get("evalue", "") and attempt == 1:
                continue  # the code never ran: safe to run it once in a fresh sandbox
            err["evalue"] = (err.get("evalue", "") + " | The Modal sandbox died during this call; "
                             "its kernel state is lost (the next call starts a new one).")
            return res
        return res


async def _drop_if_dead(pid: str) -> bool:
    """After a failed call: is the project's sandbox actually gone? If so, forget it."""
    entry = _SANDBOXES.get(pid)
    if entry is None:
        return True
    rc = await entry["sb"].poll.aio()
    if rc is None:
        # poll() lags a death by seconds, so probe the service; three misses before killing a
        # sandbox Modal still reports as running (a slow tunnel must not cost the kernel).
        async with httpx.AsyncClient(timeout=10.0, limits=httpx.Limits(max_keepalive_connections=0)) as c:
            for i in range(3):
                try:
                    if (await c.get(f"{entry['url']}/health")).status_code == 200:
                        return False
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(2.0)
                rc = await entry["sb"].poll.aio()
                if rc is not None:
                    break
    _SANDBOXES.pop(pid, None)
    print(f"    [Modal] !!! sandbox {entry['sb'].object_id} for {pid[:8]} is GONE (rc={rc}; 137 = killed: OOM or terminate). "
          "Its KERNEL STATE IS LOST.", flush=True)
    try:
        await entry["sb"].terminate.aio()
    except Exception:  # noqa: BLE001
        pass
    return True
