"""Deployment configuration: every machine-specific path and URL, in one place (032 phase 0).

The defaults are the local research stack's values, so a machine without these variables set
behaves exactly as before. A hosted deployment overrides them from its environment.

Stdlib + dotenv only, and no import of gads.core.database: the Streamlit UI and the offline
evaluators (gads.eval) import this module and must not need GADS_DATABASE_URL.

Values are read at import time, like the module constants they replace. Code that must honour
an override set after import (tests) calls the functions, which re-read the environment.
"""
import os
import uuid
from typing import Union

from dotenv import load_dotenv

# Imported by modules that load before server.py's own load_dotenv() runs.
load_dotenv()

DEPLOYMENTS = ("local", "hosted")

# Local research-stack defaults. The only place /home/joergf may appear in src/.
_LOCAL_WORKSPACE_ROOT = "/home/joergf/projects/MyLocalStack/data/workspaces"
_LOCAL_DATASETS_ROOT = "/home/joergf/datasets"
_LOCAL_HEADER_IMAGE = "/home/joergf/Generated Image May 05, 2026 - 10_53PM.png"
_LOCAL_SANDBOX_URL = "http://localhost:8000"


def deployment() -> str:
    value = (os.getenv("GADS_DEPLOYMENT") or "local").strip().lower()
    if value not in DEPLOYMENTS:
        raise ValueError(f"GADS_DEPLOYMENT must be one of {DEPLOYMENTS}, got {value!r}")
    return value


def workspace_root() -> str:
    """Directory holding one workspace per project (shared with the sandbox container)."""
    return os.getenv("GADS_WORKSPACE_ROOT") or _LOCAL_WORKSPACE_ROOT


def workspace_dir(project_id: Union[uuid.UUID, str]) -> str:
    return f"{workspace_root()}/{project_id}"


def datasets_root() -> str:
    """Source-dataset root, bind-mounted read-only into the sandbox (see server._mount_external_dataset)."""
    return os.getenv("GADS_DATASETS_ROOT") or _LOCAL_DATASETS_ROOT


def sandbox_url() -> str:
    return (os.getenv("GADS_SANDBOX_URL") or _LOCAL_SANDBOX_URL).rstrip("/")


def sandbox_backend() -> str:
    return (os.getenv("GADS_SANDBOX_BACKEND") or "local_docker").strip().lower()


def header_image() -> str:
    return os.getenv("GADS_HEADER_IMAGE") or _LOCAL_HEADER_IMAGE


def otel_endpoint() -> str:
    """OTLP traces endpoint. Empty means tracing is off (spans are created but not exported)."""
    return (os.getenv("GADS_OTEL_ENDPOINT") or "").strip()


AUTH_MODES = ("single_user", "dev_header")
# Modes that do NOT verify who is calling. Never acceptable on a hosted deployment; Cloudflare
# Access JWT verification (032 §3) arrives as its own mode in phase 4.
_UNVERIFIED_AUTH_MODES = ("single_user", "dev_header")


def auth_mode() -> str:
    """How the backend learns who is calling (core/tenancy.py).

    single_user — every request is the admin (the local research setup; nothing changes).
    dev_header  — trusts an `X-GADS-User: <email>` header. For local multi-user tests only.
    """
    value = (os.getenv("GADS_AUTH") or "single_user").strip().lower()
    if value not in AUTH_MODES:
        raise ValueError(f"GADS_AUTH must be one of {AUTH_MODES}, got {value!r}")
    if deployment() == "hosted" and value in _UNVERIFIED_AUTH_MODES:
        raise ValueError(f"GADS_AUTH={value!r} does not verify identity and is refused when "
                         "GADS_DEPLOYMENT=hosted")
    return value


def admin_email() -> str:
    """The admin's login. A generic placeholder by default: the local single user."""
    return (os.getenv("GADS_ADMIN_EMAIL") or "admin@localhost").strip().lower()


def secret_key() -> str:
    """Fernet key that encrypts stored provider keys (core/credentials.py). Empty = storing refused."""
    return (os.getenv("GADS_SECRET_KEY") or "").strip()


# Import-time snapshots, for the modules that used constants.
DEPLOYMENT = deployment()
WORKSPACE_ROOT = workspace_root()
DATASETS_ROOT = datasets_root()
SANDBOX_URL = sandbox_url()
