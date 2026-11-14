"""A sandbox with NO shared disk: the workspace is synced over HTTP (HOSTED-3a, 032 §6.4).

The local stack bind-mounts each project workspace into the sandbox. A remote sandbox (Modal,
Docker Cloud Sandboxes) cannot, so `IsolatedHttpBackend.execute` wraps every call:

    push changed workspace files  ->  run the code  ->  pull new / changed outputs

The canonical workspace stays on the backend host: reporting, dashboard rebuilds, downloads and
the executor's "new files" scan all keep reading it there, unchanged.

Safety rules (each one has a test in scripts/test_sandbox_sync.py):
  * A PULL NEVER WRITES THROUGH A LOCAL SYMLINK. Datasets sit in a workspace as links into the
    read-only datasets root. Remotely they are plain, writable copies, so task code can rewrite
    them; writing that back through the link would modify the source dataset (the 2026-05
    incident, which the sandbox's read-only mount prevents locally but cannot prevent here).
    Such files are skipped and logged. Any symlinked directory on the way is refused too.
  * PUSH follows a link only into an allowed root and sends the CONTENT: this workspace; the
    datasets root (pushed READ-ONLY, so a task that overwrites a dataset fails remotely exactly
    as it does against the local read-only mount, instead of later nodes silently computing on
    altered data); another workspace ONLY from `upstream/` (artifacts_from mounts), and only if the
    project's owner owns that workspace (`upstream_check`, the rule download_file applies).
  * TRANSFERS ARE ATOMIC. Both sides write a temp file and rename it; the record of what was
    synced is updated only after a transfer completed, so a call cancelled by its caller's
    timeout leaves no partial file and the next call finishes the job.
  * SYNC STATE OUTLIVES BACKEND INSTANCES (they are created per call) and is keyed by the remote
    sandbox's instance_id: a restarted sandbox has an empty workspace, so a new id resets it.
  * A sync failure returns the same ConnectionError result an unreachable sandbox does today.

Not yet: deletions are not synced (a file removed on one side stays on the other).
"""
from __future__ import annotations

import asyncio
import os
import re
import tempfile
import time
import uuid
from typing import Dict, Iterator, Optional, Tuple

from gads.core import config
from gads.tools.sandbox import ExecutionResult, LocalDockerBackend

_CHUNK = 1024 * 1024
_SKIP_PREFIXES = (".sync_", ".upload_")
_UUID = re.compile(r"^[0-9a-fA-F-]{36}$")

# Shared across instances (make_sandbox() returns a fresh backend per call).
_STATE: Dict[str, dict] = {}
_LOCKS: Dict[str, asyncio.Lock] = {}


def reset_state() -> None:
    """Forget everything synced (tests)."""
    _STATE.clear()


def _conn_error(msg: str) -> ExecutionResult:
    return ExecutionResult(stdout="", stderr=msg,
                           error={"ename": "ConnectionError", "evalue": msg, "traceback": []},
                           execution_time_ms=0, kernel_state={})


def _inside(path: str, root: str) -> bool:
    try:
        p, r = os.path.realpath(path), os.path.realpath(root)
    except OSError:
        return False
    return p == r or p.startswith(r + os.sep)


def _workspace_of(path: str, workspaces: str) -> str:
    """The workspace (project id) a path under the workspaces root resolves into."""
    return os.path.relpath(os.path.realpath(path), os.path.realpath(workspaces)).split(os.sep)[0]


def upstream_check(project_key: str):
    """Predicate: may this project's sandbox receive files from upstream workspace `w`?

    Only if the project's OWNER owns `w` (the admin owns everything). The remote sandbox gets a
    copy, so this is the one place a cross-tenant link would turn into a cross-tenant read;
    download_file applies the same rule. Looked up once per workspace per push.
    """
    from gads.core import tenancy  # lazy: needs the DB, and only remote backends get here
    owner = tenancy.owner_of_project(uuid.UUID(project_key))
    cache: Dict[str, bool] = {}

    def ok(w: str) -> bool:
        if w == project_key:
            return True
        if w not in cache:
            cache[w] = tenancy.owns_project(owner, w)
        return cache[w]
    return ok


def _local_files(local: str, upstream_ok=lambda w: False) -> Iterator[Tuple[str, str, bool]]:
    """(relative path, real path to read, readonly) for every file to push.

    Regular files, plus symlinked files whose target is inside an allowed root (pushed as
    content; dataset content read-only). Symlinked directories are not descended into.
    """
    datasets, workspaces = config.datasets_root(), config.workspace_root()
    for dirpath, dirnames, filenames in os.walk(local, followlinks=False):
        dirnames[:] = [d for d in dirnames if not os.path.islink(os.path.join(dirpath, d))]
        for name in filenames:
            if name.startswith(_SKIP_PREFIXES):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, local)
            if os.path.islink(full):
                if not os.path.isfile(full):
                    continue
                if _inside(full, local):
                    yield rel, os.path.realpath(full), False
                elif _inside(full, datasets):
                    yield rel, os.path.realpath(full), True
                elif rel.split(os.sep)[0] == "upstream" and _inside(full, workspaces):
                    if upstream_ok(_workspace_of(full, workspaces)):
                        yield rel, os.path.realpath(full), False
                    else:
                        print(f"    [Sync] not pushing link {rel}: upstream workspace has another owner", flush=True)
                else:
                    print(f"    [Sync] not pushing link {rel}: target outside the allowed roots", flush=True)
            elif os.path.isfile(full):
                yield rel, full, False


def _crosses_link(local: str, rel: str) -> bool:
    cur = local
    for part in os.path.normpath(rel).split(os.sep):
        cur = os.path.join(cur, part)
        if os.path.islink(cur):
            return True
    return False


def _sig(path: str) -> Tuple[int, int]:
    st = os.stat(path)
    return st.st_size, st.st_mtime_ns


class IsolatedHttpBackend(LocalDockerBackend):
    """LocalDockerBackend's HTTP API + an explicit workspace sync around every execute."""

    async def _listing(self, key: str) -> dict:
        url, headers = self.endpoint(key)
        r = await self.client.get(f"{url}/files/{key}", headers=headers, timeout=60.0)
        r.raise_for_status()
        return r.json()

    def _state_for(self, key: str, instance_id: str) -> dict:
        st = _STATE.get(key)
        if st is None or st["instance"] != instance_id:
            if st is not None:
                print(f"    [Sync] sandbox for {key[:8]} restarted (instance changed); full re-push", flush=True)
            st = {"instance": instance_id, "pushed": {}, "pulled": {}}
            _STATE[key] = st
        return st

    async def _push(self, key: str, local: str, stats: dict) -> None:
        listing = await self._listing(key)
        st = self._state_for(key, listing["instance_id"])
        remote = {f["path"] for f in listing["files"]}
        url, headers = self.endpoint(key)
        for rel, src, readonly in _local_files(local, upstream_check(key)):
            sig = _sig(src)
            if st["pushed"].get(rel) == sig and rel in remote:
                continue

            async def body(path=src):
                with open(path, "rb") as f:
                    while True:
                        chunk = f.read(_CHUNK)
                        if not chunk:
                            break
                        yield chunk
            r = await self.client.put(f"{url}/files/{key}/{rel}", content=body(), headers=headers,
                                      params={"readonly": "1"} if readonly else None, timeout=600.0)
            r.raise_for_status()
            resp = r.json()
            if resp.get("instance_id") != st["instance"]:
                raise RuntimeError("sandbox restarted during push")
            # Record only after the transfer completed (atomicity across cancellation).
            st["pushed"][rel] = sig
            st["pulled"][rel] = (resp["size"], resp["mtime_ns"])  # don't pull our own push back
            stats["push_files"] += 1
            stats["push_bytes"] += sig[0]

    async def _pull(self, key: str, local: str, stats: dict) -> None:
        listing = await self._listing(key)
        st = _STATE.get(key)
        if st is None or st["instance"] != listing["instance_id"]:
            raise RuntimeError("sandbox restarted during execution; outputs lost")
        url, headers = self.endpoint(key)
        for f in listing["files"]:
            rel, rsig = f["path"], (f["size"], f["mtime_ns"])
            if st["pulled"].get(rel) == rsig:
                continue
            if os.path.isabs(rel) or os.path.normpath(rel).startswith(".."):
                print(f"    [Sync] refusing remote path {rel!r}", flush=True)
                continue
            if _crosses_link(local, rel):
                # Never write through a link (e.g. task code rewrote a pushed dataset copy).
                print(f"    [Sync] not pulling {rel}: the local path is a symlink (source data stays untouched)",
                      flush=True)
                st["pulled"][rel] = rsig
                stats["pull_skipped_links"] += 1
                continue
            dest = os.path.join(local, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if _crosses_link(local, rel):  # makedirs must not have raced a link into the path
                continue
            fd, tmp = tempfile.mkstemp(prefix=".sync_", dir=os.path.dirname(dest))
            try:
                with os.fdopen(fd, "wb") as out:
                    async with self.client.stream("GET", f"{url}/files/{key}/{rel}", headers=headers,
                                                  timeout=600.0) as r:
                        r.raise_for_status()
                        async for chunk in r.aiter_bytes(_CHUNK):
                            out.write(chunk)
                if os.path.islink(dest):
                    continue
                os.replace(tmp, dest)
            finally:
                if os.path.exists(tmp):
                    os.remove(tmp)
            st["pulled"][rel] = rsig
            st["pushed"][rel] = _sig(dest)  # don't push it straight back
            stats["pull_files"] += 1
            stats["pull_bytes"] += rsig[0]

    async def execute(self, code: str, project_id: uuid.UUID, session_id: str = "default",
                      workspace_id: Optional[str] = None) -> ExecutionResult:
        key = workspace_id or session_id
        local = config.workspace_dir(key)
        if not _UUID.match(key or "") or not os.path.isdir(local):
            # Not a project workspace (e.g. a scratch session): nothing to sync.
            return await super().execute(code, project_id, session_id=session_id, workspace_id=workspace_id)
        stats = {"push_files": 0, "push_bytes": 0, "pull_files": 0, "pull_bytes": 0,
                 "pull_skipped_links": 0}
        lock = _LOCKS.setdefault(key, asyncio.Lock())
        async with lock:
            t0 = time.monotonic()
            try:
                await self._push(key, local, stats)
            except Exception as e:
                return _conn_error(f"Sandbox file sync (push) failed: {type(e).__name__}: {e}")
            stats["push_s"] = round(time.monotonic() - t0, 3)
            res = await super().execute(code, project_id, session_id=session_id, workspace_id=workspace_id)
            t1 = time.monotonic()
            last = None
            for _ in range(2):  # one retry: a pull failure after a successful run loses outputs
                try:
                    await self._pull(key, local, stats)
                    last = None
                    break
                except Exception as e:
                    last = e
            stats["pull_s"] = round(time.monotonic() - t1, 3)
            if last is not None:
                return _conn_error(f"Sandbox file sync (pull) failed: {type(last).__name__}: {last}")
        res.sync_stats = stats
        if stats["push_files"] or stats["pull_files"] or stats["pull_skipped_links"]:
            print(f"    [Sync] {key[:8]}: pushed {stats['push_files']} file(s) "
                  f"{stats['push_bytes'] / 1e6:.2f} MB in {stats['push_s']}s, pulled {stats['pull_files']} "
                  f"{stats['pull_bytes'] / 1e6:.2f} MB in {stats['pull_s']}s"
                  + (f", {stats['pull_skipped_links']} link(s) not written through" if stats['pull_skipped_links'] else ""),
                  flush=True)
        return res
