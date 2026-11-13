"""Bring-your-own-key LLM credentials (HOSTED-2a, approach_docs/032 §7 fallback path).

LiteLLM's team-scoped deployments are enterprise-only, but the OSS proxy honours a provider
`api_key` sent in the request BODY (verified for Gemini, Anthropic and OpenAI, streaming too;
the key does not appear in proxy logs, its DB or the MLflow traces). So GADS stores each user's
provider keys itself, Fernet-encrypted with GADS_SECRET_KEY, and `llm.py` sends the right one
per call. The proxy's own virtual key (`kwargs["api_key"]`) still authenticates GADS to the
proxy and is unchanged.

A run's credentials are a frozen snapshot in a ContextVar, next to the settings snapshot:
  GATEWAY                 → the gateway's keys (runs of admin-owned projects; nothing changes)
  RunCredentials(keys=…)  → a tester's run: every model call MUST use one of these keys.
  unset                   → gateway only with GADS_AUTH=single_user (the local research setup);
                            in a multi-user mode a model call with no context fails closed, so a
                            code path that forgot to set credentials can never bill the gateway.
FAIL CLOSED: in a tester's run, a model whose provider has no stored key, or a model that is
not BYOK-eligible at all, raises MissingProviderKey. It never falls back to the gateway key.

Which provider a model belongs to comes from the proxy's /model/info (`litellm_provider`),
never from the model's name. A deployment with a custom `api_base` (Kimi via Moonshot,
local_model) is NOT BYOK-eligible: a tester's OpenAI key must never be sent to Moonshot.
"""
from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import httpx
from sqlmodel import Session, select

from gads.core import config

BYOK_PROVIDERS = ("anthropic", "gemini", "openai")

# One cheap model per provider for validating a key before it is stored. Must be on the
# proxy and BYOK-eligible; checked against the catalog at validation time.
VALIDATION_MODELS = {
    "gemini": "gemini-3.5-flash-lite",
    "anthropic": "claude-haiku-5.5",
    "openai": "gpt-6-luna",
}


class MissingProviderKey(RuntimeError):
    """A tester's run tried to call a model it holds no key for (fail closed)."""


class SecretKeyMissing(RuntimeError):
    pass


# ——— encryption ———————————————————————————————————————————————————————————————————————

def _fernet():
    from cryptography.fernet import Fernet
    key = config.secret_key()
    if not key:
        raise SecretKeyMissing("GADS_SECRET_KEY is not set; refusing to store or read provider keys")
    return Fernet(key.encode())


def encrypt(raw: str) -> str:
    return _fernet().encrypt(raw.encode()).decode()


def decrypt(token: str) -> str:
    return _fernet().decrypt(token.encode()).decode()


# ——— model → provider catalog (from the proxy) ———————————————————————————————————————————

_CATALOG: Dict[str, object] = {}
_CATALOG_TTL_S = 300.0


def _proxy_base() -> str:
    return (os.getenv("LITELLM_BASE_URL") or "http://localhost:4000/v1").rstrip("/").removesuffix("/v1")


def _proxy_key() -> str:
    return os.getenv("LITELLM_MASTER_KEY", "sk-1234")


def parse_catalog(model_info: List[dict]) -> Dict[str, Optional[str]]:
    """{model_name: provider} for BYOK-eligible deployments, None for the rest.

    Eligible = litellm_provider in BYOK_PROVIDERS and no custom api_base. A model with several
    deployments is eligible only if every deployment is (else a call could land on one that is not).
    """
    out: Dict[str, Optional[str]] = {}
    for m in model_info or []:
        name = m.get("model_name")
        if not name:
            continue
        mi, lp = m.get("model_info") or {}, m.get("litellm_params") or {}
        prov = mi.get("litellm_provider")
        eligible = prov in BYOK_PROVIDERS and not lp.get("api_base")
        this = prov if eligible else None
        if name in out and out[name] != this:
            this = None
        out[name] = this
    return out


def model_providers(force: bool = False) -> Dict[str, Optional[str]]:
    """The proxy's model → BYOK provider map, cached for a few minutes."""
    now = time.monotonic()
    if not force and _CATALOG.get("map") is not None and now - float(_CATALOG.get("at", 0)) < _CATALOG_TTL_S:
        return _CATALOG["map"]  # type: ignore[return-value]
    resp = httpx.get(f"{_proxy_base()}/model/info", headers={"Authorization": f"Bearer {_proxy_key()}"},
                     timeout=15.0)
    resp.raise_for_status()
    mapping = parse_catalog(resp.json().get("data", []))
    _CATALOG.update(map=mapping, at=now)
    return mapping


# ——— per-run credentials ————————————————————————————————————————————————————————————————

@dataclass(frozen=True)
class RunCredentials:
    keys: Dict[str, str] = field(repr=False)               # provider -> raw key (never in a repr)
    providers: Dict[str, Optional[str]] = field(default_factory=dict)  # model -> provider

    def allows(self, model: str) -> bool:
        prov = self.providers.get(model)
        return prov is not None and prov in self.keys

    def key_for(self, model: str) -> str:
        prov = self.providers.get(model)
        if prov is None:
            raise MissingProviderKey(f"Model '{model}' cannot be used with your own keys "
                                     "(not a BYOK model: unknown, local, or a custom endpoint)")
        if prov not in self.keys:
            raise MissingProviderKey(f"Model '{model}' needs a {prov} key; add one under your model keys")
        return self.keys[prov]


class _Gateway:
    def __repr__(self):
        return "GATEWAY"


GATEWAY = _Gateway()  # explicit marker: this run uses the gateway's own provider keys

_run_credentials: ContextVar[object] = ContextVar("gads_run_credentials", default=None)


@contextmanager
def credentials_scope(creds):
    """Set a run's credentials: GATEWAY or a RunCredentials."""
    token = _run_credentials.set(creds)
    try:
        yield
    finally:
        _run_credentials.reset(token)


def current():
    """GATEWAY, a RunCredentials, or None (unset)."""
    return _run_credentials.get()


def provider_key_for(model: str) -> Optional[str]:
    """The provider key to send for this call: None = gateway. Raises fail-closed."""
    creds = _run_credentials.get()
    if creds is GATEWAY:
        return None
    if creds is None:
        if config.auth_mode() == "single_user":
            return None
        raise MissingProviderKey("No run credentials are set for this model call (multi-user mode "
                                 "refuses to fall back to the gateway key)")
    return creds.key_for(model)


def filter_models(models: List[str]) -> List[str]:
    """Inside a BYOK run, keep only models this run holds a key for; else unchanged."""
    creds = _run_credentials.get()
    if not isinstance(creds, RunCredentials):
        return list(models)
    return [m for m in models if creds.allows(m)]


# ——— storage ————————————————————————————————————————————————————————————————————————————

def _session():
    from gads.core.database import engine
    return Session(engine)


def list_keys(user_id: uuid.UUID) -> List[dict]:
    from gads.core.models import ProviderKey
    with _session() as s:
        rows = s.exec(select(ProviderKey).where(ProviderKey.user_id == user_id)).all()
        return [{"provider": r.provider, "last4": r.last4,
                 "created_at": r.created_at.isoformat() if r.created_at else None}
                for r in sorted(rows, key=lambda r: r.provider)]


def load_keys(user_id: uuid.UUID) -> Dict[str, str]:
    """Decrypted keys for a run snapshot. A key that no longer decrypts is skipped (logged)."""
    from gads.core.models import ProviderKey
    out: Dict[str, str] = {}
    with _session() as s:
        for r in s.exec(select(ProviderKey).where(ProviderKey.user_id == user_id)).all():
            try:
                out[r.provider] = decrypt(r.ciphertext)
            except SecretKeyMissing:
                raise
            except Exception:
                print(f"  [Credentials] stored {r.provider} key for {user_id} does not decrypt "
                      "(GADS_SECRET_KEY changed?); skipped", flush=True)
    return out


def store_key(user_id: uuid.UUID, provider: str, raw: str) -> dict:
    from gads.core.models import ProviderKey
    if provider not in BYOK_PROVIDERS:
        raise ValueError(f"provider must be one of {BYOK_PROVIDERS}")
    raw = (raw or "").strip()
    if len(raw) < 8:
        raise ValueError("key looks too short")
    token = encrypt(raw)
    with _session() as s:
        row = s.exec(select(ProviderKey).where(ProviderKey.user_id == user_id,
                                               ProviderKey.provider == provider)).first()
        if row is None:
            row = ProviderKey(user_id=user_id, provider=provider, ciphertext=token, last4=raw[-4:])
        else:
            row.ciphertext, row.last4 = token, raw[-4:]
        s.add(row)
        s.commit()
    return {"provider": provider, "last4": raw[-4:]}


def delete_key(user_id: uuid.UUID, provider: str) -> bool:
    from gads.core.models import ProviderKey
    with _session() as s:
        row = s.exec(select(ProviderKey).where(ProviderKey.user_id == user_id,
                                               ProviderKey.provider == provider)).first()
        if row is None:
            return False
        s.delete(row)
        s.commit()
        return True


async def validate_key(provider: str, raw: str) -> None:
    """One tiny call through the proxy with the candidate key in the body. Raises ValueError."""
    model = VALIDATION_MODELS.get(provider)
    providers = model_providers()
    if not model or providers.get(model) != provider:
        raise ValueError(f"No validation model available for {provider}")
    body = {"model": model, "messages": [{"role": "user", "content": "Reply with OK"}],
            "max_tokens": 5, "api_key": raw}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{_proxy_base()}/v1/chat/completions", json=body,
                                 headers={"Authorization": f"Bearer {_proxy_key()}"})
    if resp.status_code != 200:
        # Never echo the provider's message: some echo (masked) key material.
        raise ValueError(f"The {provider} key was rejected (HTTP {resp.status_code})")


def run_credentials_for_owner(owner):
    """The credentials snapshot for a run owned by `owner` (a tenancy.Caller).

    Admin → GATEWAY (gateway keys, unchanged). Tester → their decrypted keys + the model
    catalog; a tester with no keys gets an empty RunCredentials, so every call fails closed.
    """
    if getattr(owner, "is_admin", False):
        return GATEWAY
    return RunCredentials(keys=load_keys(owner.id), providers=model_providers())
