"""Who is calling, and what they may touch (approach_docs/032 §4, HOSTED-1).

Three pieces:

1. Identity. `current_user(conn)` resolves the caller per `GADS_AUTH` (core/config.py):
   `single_user` makes every request the admin (the local research setup, unchanged);
   `dev_header` trusts `X-GADS-User: <email>` for local multi-user tests. Hosted deployments
   refuse both (config.auth_mode); Cloudflare Access JWT verification is phase 4.

2. Route policy, DEFAULT DENY. Every route is listed in ROUTE_POLICY. `enforce_route_policy`
   runs as an app-wide dependency: a route missing from the table is refused (403), and
   `scripts/test_tenancy.py` fails on it, so a new endpoint cannot silently ship open.
     public — no identity needed (health)
     read   — any caller; the handler serves shared knowledge, or filters by owner itself
     admin  — admin only
     owner  — the path's {project_id} / {task_id} must belong to the caller (admin: any);
              a non-owner gets 404, so project ids do not leak by status code
     self   — the caller's own record (/me)

3. Ownership helpers for handlers that filter lists or take ids in a body, and for /ws.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Set

from fastapi import HTTPException
from sqlmodel import Session, select
from starlette.requests import HTTPConnection

from gads.core import config
from gads.core.database import engine
from gads.core.models import Project, Task, User


@dataclass(frozen=True)
class Caller:
    """A detached snapshot of the calling user (safe to pass across sessions and tasks)."""
    id: uuid.UUID
    email: str
    role: str
    settings: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


def _snapshot(u: User) -> Caller:
    return Caller(id=u.id, email=u.email, role=u.role, settings=dict(u.settings_json or {}))


def get_or_create_user(email: str) -> Caller:
    """The User for this email, created on first sight (the admin email gets role admin)."""
    email = (email or "").strip().lower()
    with Session(engine) as session:
        u = session.exec(select(User).where(User.email == email)).first()
        if u is None:
            u = User(email=email, role="admin" if email == config.admin_email() else "tester")
            session.add(u)
            session.commit()
            session.refresh(u)
        return _snapshot(u)


def admin_user() -> Caller:
    return get_or_create_user(config.admin_email())


_ADMIN_CACHE: Dict[str, Any] = {}


def _cached_admin() -> Caller:
    """single_user mode resolves every request to the admin: refresh from the DB at most
    every few seconds instead of on each request (the UI polls streams once a second)."""
    import time
    hit = _ADMIN_CACHE.get("caller")
    if hit is not None and time.monotonic() - _ADMIN_CACHE.get("at", 0) < 5.0:
        return hit
    caller = admin_user()
    _ADMIN_CACHE.update(caller=caller, at=time.monotonic())
    return caller


def current_user(conn: HTTPConnection) -> Caller:
    """Resolve the caller of an HTTP request or websocket. Raises 401 when unidentified."""
    cached = getattr(conn.state, "gads_caller", None)
    if cached is not None:
        return cached
    mode = config.auth_mode()
    if mode == "single_user":
        caller = _cached_admin()
    elif mode == "dev_header":
        email = (conn.headers.get("x-gads-user") or "").strip().lower()
        if not email or "@" not in email:
            raise HTTPException(status_code=401, detail="X-GADS-User header required (GADS_AUTH=dev_header)")
        caller = get_or_create_user(email)
    else:  # pragma: no cover - config.auth_mode validates
        raise HTTPException(status_code=500, detail=f"unsupported GADS_AUTH {mode!r}")
    conn.state.gads_caller = caller
    return caller


# ——— ownership ————————————————————————————————————————————————————————————————————————

def owns_project(caller: Caller, project_id: Any, session: Optional[Session] = None) -> bool:
    """True if the caller may see this project. Unknown ids are False (→ 404 either way)."""
    try:
        pid = project_id if isinstance(project_id, uuid.UUID) else uuid.UUID(str(project_id))
    except (ValueError, TypeError):
        return False
    own = session or Session(engine)
    try:
        p = own.get(Project, pid)
        if p is None:
            # Archive recovery: a workspace with no DB row is still the admin's to open.
            return caller.is_admin
        # NULL owner = admin-only (rows from before tenancy that escaped the backfill).
        return caller.is_admin or (p.owner_id is not None and p.owner_id == caller.id)
    finally:
        if session is None:
            own.close()


def require_project(caller: Caller, project_id: Any) -> None:
    if caller.is_admin:  # admin may open any project (and archive workspaces): no DB lookup
        return
    if not owns_project(caller, project_id):
        raise HTTPException(status_code=404, detail="Project not found")


def project_of_task(task_id: Any, session: Optional[Session] = None) -> Optional[uuid.UUID]:
    try:
        tid = task_id if isinstance(task_id, uuid.UUID) else uuid.UUID(str(task_id))
    except (ValueError, TypeError):
        return None
    own = session or Session(engine)
    try:
        t = own.get(Task, tid)
        return t.project_id if t else None
    finally:
        if session is None:
            own.close()


def require_task(caller: Caller, task_id: Any) -> None:
    if caller.is_admin:  # admin sees everything, including stream keys with no Task row
        return
    pid = project_of_task(task_id)
    if pid is None or not owns_project(caller, pid):
        raise HTTPException(status_code=404, detail="Task not found")


def owned_project_ids(caller: Caller, session: Session) -> Optional[Set[uuid.UUID]]:
    """The caller's project ids, or None for the admin (meaning: all)."""
    if caller.is_admin:
        return None
    return set(session.exec(select(Project.id).where(Project.owner_id == caller.id)).all())


def project_owner_id(project_id: uuid.UUID) -> Optional[uuid.UUID]:
    with Session(engine) as session:
        p = session.get(Project, project_id)
        return p.owner_id if p else None


def owner_of_project(project_id: uuid.UUID) -> Caller:
    """The project's owner as a Caller (the admin for an ownerless project)."""
    with Session(engine) as session:
        p = session.get(Project, project_id)
        if p is not None and p.owner_id is not None:
            u = session.get(User, p.owner_id)
            if u is not None:
                return _snapshot(u)
    return admin_user()


def event_project_id(payload: Dict[str, Any], session: Session) -> Optional[uuid.UUID]:
    """The project an outbox event belongs to (payload project_id, else via its task_id)."""
    raw = (payload or {}).get("project_id")
    if raw:
        try:
            return uuid.UUID(str(raw))
        except ValueError:
            return None
    tid = (payload or {}).get("task_id")
    return project_of_task(tid, session) if tid else None


def event_visible(caller: Caller, payload: Dict[str, Any], session: Session) -> bool:
    """Whether a /ws event may be delivered to this caller. Unattributable events: admin only."""
    if caller.is_admin:
        return True
    pid = event_project_id(payload, session)
    return pid is not None and owns_project(caller, pid, session)


# ——— route policy (default deny) ————————————————————————————————————————————————————————

PUBLIC, READ, ADMIN, OWNER, SELF = "public", "read", "admin", "owner", "self"

ROUTE_POLICY: Dict[tuple, str] = {
    ("GET", "/health"): PUBLIC,
    # Caller identity + own settings.
    ("GET", "/me"): SELF,
    ("GET", "/me/settings"): SELF,
    ("PUT", "/me/settings"): SELF,
    # Shared knowledge: read for everyone, writes admin (032 §9).
    ("GET", "/recipes"): READ,
    ("GET", "/recipes/{filename}"): READ,
    ("POST", "/recipes/{filename}"): ADMIN,
    ("POST", "/knowledge/validate"): ADMIN,   # deep validation of natives runs code checks
    ("GET", "/knowledge/items"): READ,
    ("POST", "/knowledge/{item_type}/{filename}/reset"): ADMIN,
    ("GET", "/native"): READ,
    ("GET", "/native/{filename}"): READ,
    ("POST", "/native/{filename}"): ADMIN,
    ("GET", "/knowledge/graph"): READ,
    ("GET", "/knowledge/coverage"): READ,
    ("GET", "/knowledge/{item_type}/{item_id}/history"): READ,
    ("GET", "/knowledge/{item_type}/{item_id}/diff"): READ,
    ("GET", "/knowledge/{item_type}/{item_id}/impact"): ADMIN,    # spec library + run counts
    ("GET", "/knowledge/{item_type}/{item_id}/evidence"): ADMIN,  # the run ledger
    ("GET", "/knowledge/{item_type}/{item_id}"): READ,
    ("GET", "/taxonomy"): READ,
    ("POST", "/taxonomy/validate"): READ,     # stateless check of a tag block
    ("GET", "/taxonomy/coverage"): ADMIN,     # over the spec library
    ("GET", "/taxonomy/specs"): ADMIN,
    ("GET", "/taxonomy/recipes"): READ,
    ("GET", "/taxonomy/runs"): READ,          # handler filters to the caller's projects
    ("GET", "/prompts"): READ,
    ("GET", "/hierarchy"): READ,
    ("POST", "/prompts/{agent_name}"): ADMIN,
    ("DELETE", "/prompts/{agent_name}"): ADMIN,
    ("GET", "/skills"): READ,
    ("GET", "/skills/match"): READ,
    ("GET", "/skills/{filename}"): READ,
    ("POST", "/skills/{filename}"): ADMIN,
    # Global settings layer: anyone reads the defaults, only the admin changes them.
    ("GET", "/config"): READ,
    ("POST", "/config"): ADMIN,
    ("POST", "/maintenance/cleanup"): ADMIN,
    # Spec library + spec launches (they can mount another run's workspace): admin.
    ("GET", "/specs"): ADMIN,
    ("GET", "/specs/{filename}"): ADMIN,
    ("POST", "/specs/{filename}"): ADMIN,
    ("POST", "/projects/from-spec"): ADMIN,
    # Projects.
    ("GET", "/projects"): READ,               # handler filters to the caller's projects
    ("POST", "/projects"): READ,              # handler checks existing_project_id ownership
    ("GET", "/projects/{project_id}"): OWNER,
    ("DELETE", "/projects/{project_id}"): OWNER,
    ("GET", "/projects/{project_id}/kernel"): OWNER,
    ("POST", "/projects/{project_id}/rehydrate"): OWNER,
    ("POST", "/projects/{project_id}/followup"): OWNER,
    ("GET", "/projects/{project_id}/notes"): OWNER,
    ("POST", "/projects/{project_id}/notes"): OWNER,
    ("POST", "/projects/{project_id}/finish"): OWNER,
    ("POST", "/projects/{project_id}/unpin"): OWNER,
    ("POST", "/projects/{project_id}/cancel"): OWNER,
    ("GET", "/projects/{project_id}/files/{file_path:path}"): OWNER,
    ("GET", "/projects/{project_id}/files"): OWNER,
    ("POST", "/projects/{project_id}/datasets"): OWNER,
    ("POST", "/projects/{project_id}/register-external"): ADMIN,  # links host paths
    ("GET", "/tasks/{task_id}/stream"): OWNER,
    ("WS", "/ws"): READ,                       # handler filters events per caller
}


def route_key(route) -> Iterable[tuple]:
    """(method, path) keys for a Starlette/FastAPI route; websocket routes use 'WS'."""
    methods = getattr(route, "methods", None)
    if methods is None:  # WebSocketRoute / APIWebSocketRoute
        return [("WS", route.path)]
    return [(m, route.path) for m in methods if m not in ("HEAD", "OPTIONS")]


def check_policy(caller: Optional[Caller], policy: Optional[str], path_params: Dict[str, Any]) -> None:
    """Raise if the caller may not use a route with this policy. `None` policy = unlisted = deny."""
    if policy is None:
        raise HTTPException(status_code=403, detail="Route has no access policy (default deny)")
    if policy == PUBLIC:
        return
    assert caller is not None
    if policy in (READ, SELF):
        return
    if policy == ADMIN:
        if not caller.is_admin:
            raise HTTPException(status_code=403, detail="Admin only")
        return
    if policy == OWNER:
        if "project_id" in path_params:
            require_project(caller, path_params["project_id"])
        elif "task_id" in path_params:
            require_task(caller, path_params["task_id"])
        else:
            raise HTTPException(status_code=403, detail="Owner route without a project or task id")
        return
    raise HTTPException(status_code=403, detail=f"Unknown policy {policy!r}")


def enforce_route_policy(conn: HTTPConnection) -> None:
    """App-wide dependency: resolve the caller and apply the route's policy (default deny)."""
    route = conn.scope.get("route")
    keys = list(route_key(route)) if route is not None else []
    method = "WS" if conn.scope.get("type") == "websocket" else conn.scope.get("method", "GET")
    policy = next((ROUTE_POLICY.get(k) for k in keys if k[0] == method), None)
    caller = None if policy == PUBLIC else current_user(conn)
    check_policy(caller, policy, dict(conn.path_params or {}))
