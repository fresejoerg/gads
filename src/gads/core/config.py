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


# Import-time snapshots, for the modules that used constants.
DEPLOYMENT = deployment()
WORKSPACE_ROOT = workspace_root()
DATASETS_ROOT = datasets_root()
SANDBOX_URL = sandbox_url()
