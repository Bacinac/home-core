"""Language models as DIDA and BABA both reach them: the provider's live model
list, a liveness ping, one-shot JSON calls, the sentence a failure is shown as,
how each Claude model is asked to think or not to, and the settings behind it
all with the routes that edit them.

Raw HTTP on httpx, no vendor SDK: a provider here needs its model list and one
completion endpoint. What a product decides itself it passes in: the roles its
models play, how the key is encrypted, who may change the settings (the
dependencies it mounts the router with). The table is the product's, created by
its own migration in the shape `AiSettings` reads:

    ai_settings(provider text PRIMARY KEY, api_key_encrypted bytea NOT NULL,
                enabled boolean, models jsonb, last_used_at timestamptz,
                created_at timestamptz, updated_at timestamptz)
"""

from __future__ import annotations

import base64
import copy
import json
import logging
import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import asyncpg
import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODELS_URL = "https://api.openai.com/v1/models"

PROVIDER_NAMES = {"anthropic": "Anthropic", "openai": "OpenAI"}

# How a Claude model is asked not to think. The Models API reports only which
# thinking types turn it ON, so this cannot be read live. Opus 5.5 and Fable
# cannot turn it off at all: the least they do is think at the lowest effort. A
# model missing here gets neither field, which every model accepts.
_THINKING_OFF: dict[str, dict[str, Any]] = {
    "claude-fable-5-1": {"output_config": {"effort": "low"}},
    "claude-fable-5": {"output_config": {"effort": "low"}},
    "claude-opus-5-5": {"output_config": {"effort": "low"}},
    "claude-sonnet-5-5": {"thinking": {"type": "between_tools"}},
    "claude-haiku-5-5": {"thinking": {"type": "disabled"}},
    "claude-opus-5": {"thinking": {"type": "disabled"}},
    "claude-sonnet-5": {"thinking": {"type": "disabled"}},
    "claude-opus-4-8": {"thinking": {"type": "disabled"}},
    "claude-opus-4-7": {"thinking": {"type": "disabled"}},
}

# What a product's role must ask of a model it sends `claude_options` to.
THINKS = frozenset({"thinking.types.adaptive", "effort.low"})


@dataclass(slots=True, frozen=True)
class Role:
    """A job a product gives a model. `needs` are capability paths from the
    provider's model list ("image_input", "structured_outputs",
    "thinking.types.adaptive"); a model offered for the role supports them all."""

    key: str
    needs: frozenset[str] = frozenset()


@dataclass(slots=True, frozen=True)
class ProviderModel:
    id: str
    label: str
    # None: the provider publishes no capabilities (OpenAI), so nothing is ruled out
    caps: frozenset[str] | None

    def serves(self, role: Role) -> bool:
        return self.caps is None or role.needs <= self.caps


@dataclass(slots=True)
class CallResult:
    ok: bool
    text: str | None
    model: str | None
    latency_ms: int
    error: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None


class ProviderModelsError(Exception):
    """The provider's model list could not be read (bad key, outage, changed shape)."""


class AiNotConfigured(LookupError):
    """No usable provider for the call: none enabled, no model for the role, or
    a stored key that will not decrypt. The message says where to fix it."""


def failure_text(provider: str, status: int | None, detail: str) -> str:
    """The one sentence a failed provider call is shown as, wherever it failed."""
    name = PROVIDER_NAMES.get(provider, provider)
    if status is None:
        return f"{name} cannot be reached: {detail}"
    if status == 401:
        return f"{name} rejected the API key — replace it in Settings → AI"
    if status == 403:
        return f"{name} does not let this key do that: {detail}"
    if status == 404:
        return f"{name} has no such model — choose another in Settings → AI ({detail})"
    if status == 429:
        return f"{name} rate limit reached — try again shortly"
    if status >= 500:
        return f"{name} is unavailable right now ({status}: {detail})"
    return f"{name} refused the request ({status}: {detail})"


def claude_options(
    model: str, *, think: bool, effort: str = "low", schema: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The `thinking` and `output_config` arguments of a Messages call, for a
    model of a role that `needs` THINKS. `think` asks for adaptive thinking at
    `effort`; without it the model thinks as little as it can. `schema` asks for
    a structured answer."""
    if think:
        opts: dict[str, Any] = {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}}
    else:
        opts = copy.deepcopy(_THINKING_OFF.get(model, {}))
    if schema is not None:
        opts.setdefault("output_config", {})["format"] = {"type": "json_schema", "schema": schema}
    return opts


def mask_key(plaintext: str | None) -> str:
    """UI-safe preview: first 4 and last 4 characters; stars for a short key."""
    if not plaintext:
        return ""
    if len(plaintext) <= 12:
        return "•" * len(plaintext)
    return f"{plaintext[:4]}…{plaintext[-4:]}"


def _client(timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout)


def _error_detail(r: httpx.Response) -> str:
    # Both providers answer errors as {"error": {"message": ...}}, but upstream
    # gateways (Cloudflare 502, nginx 504) return raw HTML, which is noise in the UI.
    try:
        err = r.json().get("error", {})
        return err.get("message") or json.dumps(err)[:300]
    except (ValueError, AttributeError):
        text = (r.text or "").lstrip()
        if text.startswith("<") or "html" in text[:80].lower():
            return "upstream gateway error"
        return text[:300]


def _network(e: httpx.HTTPError) -> str:
    return f"{e.__class__.__name__}: {e}"[:400]


def _supported(tree: dict[str, Any], prefix: str = "") -> Iterator[str]:
    for name, node in tree.items():
        if not isinstance(node, dict):
            continue
        path = f"{prefix}{name}"
        if node.get("supported"):
            yield path
        yield from _supported(node, f"{path}.")


async def _anthropic_models(api_key: str) -> list[ProviderModel]:
    headers = {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION}
    found: list[ProviderModel] = []
    params: dict[str, str | int] = {"limit": 1000}
    try:
        async with _client(15) as client:
            while True:
                r = await client.get(ANTHROPIC_MODELS_URL, headers=headers, params=params)
                if r.status_code != 200:
                    raise ProviderModelsError(failure_text("anthropic", r.status_code, _error_detail(r)))
                body = r.json()
                for m in body["data"]:
                    caps = frozenset(_supported(m["capabilities"]))
                    found.append(ProviderModel(m["id"], m.get("display_name") or m["id"], caps))
                if not body.get("has_more"):
                    return found
                params["after_id"] = body["last_id"]
    except httpx.HTTPError as e:
        raise ProviderModelsError(failure_text("anthropic", None, _network(e))) from e
    except (ValueError, KeyError, TypeError) as e:
        raise ProviderModelsError(f"unexpected model list from Anthropic: {e!r}"[:400]) from e


# OpenAI's model list carries no capability data, so the id is all there is to
# tell a chat model from audio, image, embedding and Responses-only ones.
# Dated snapshots duplicate their alias.
_OPENAI_CHAT = re.compile(r"^(gpt-([4-9]|\d{2})|o\d)")
_OPENAI_NOT_CHAT = re.compile(
    r"audio|realtime|tts|transcribe|search|image|embedding|instruct|preview"
    r"|codex|deep-research|-pro\b"
)
_OPENAI_SNAPSHOT = re.compile(r"-\d{4}(-\d{2}-\d{2})?$")


def openai_chat_models(data: list[dict[str, Any]]) -> list[ProviderModel]:
    chosen = [
        m
        for m in data
        if _OPENAI_CHAT.match(m["id"])
        and not _OPENAI_NOT_CHAT.search(m["id"])
        and not _OPENAI_SNAPSHOT.search(m["id"])
    ]
    chosen.sort(key=lambda m: m.get("created") or 0, reverse=True)
    return [ProviderModel(m["id"], m["id"], None) for m in chosen]


async def _openai_models(api_key: str) -> list[ProviderModel]:
    headers = {"authorization": f"Bearer {api_key}"}
    try:
        async with _client(15) as client:
            r = await client.get(OPENAI_MODELS_URL, headers=headers)
        if r.status_code != 200:
            raise ProviderModelsError(failure_text("openai", r.status_code, _error_detail(r)))
        return openai_chat_models(r.json()["data"])
    except httpx.HTTPError as e:
        raise ProviderModelsError(failure_text("openai", None, _network(e))) from e
    except (ValueError, KeyError, TypeError) as e:
        raise ProviderModelsError(f"unexpected model list from OpenAI: {e!r}"[:400]) from e


async def models(provider: str, api_key: str) -> list[ProviderModel]:
    """What the key can call, newest first, as the provider lists it."""
    if provider == "anthropic":
        return await _anthropic_models(api_key)
    if provider == "openai":
        return await _openai_models(api_key)
    raise ProviderModelsError(f"unsupported provider: {provider!r}")


_PING = "Reply with the single word OK."


async def ping(provider: str, api_key: str, model: str) -> CallResult:
    """The cheapest call that proves the key and the model together."""
    if provider == "anthropic":
        # Current Claude models think before answering and the thinking counts
        # against max_tokens; with 8 the ping stops at max_tokens on a valid key.
        return await _anthropic_call(api_key, {
            "model": model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": _PING}],
            **claude_options(model, think=False),
        })
    if provider == "openai":
        # The budget covers internal reasoning too: with 8, a reasoning model burns
        # it all before any visible text and fails with finish_reason=length.
        return await _openai_call(api_key, {
            "model": model,
            "max_completion_tokens": 1024,
            "messages": [{"role": "user", "content": _PING}],
        })
    raise ValueError(f"unsupported provider: {provider!r}")


async def text_json(
    provider: str, api_key: str, model: str, *, system: str, user_text: str, max_tokens: int = 8000
) -> CallResult:
    """One question about text that the model answers in JSON. Skipping an image
    where the task is numbers keeps the model on the numbers."""
    if provider == "anthropic":
        return await _anthropic_call(api_key, {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user_text}]}],
        }, http_timeout_s=60)
    if provider == "openai":
        return await _openai_call(api_key, {
            "model": model,
            "max_completion_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
        }, http_timeout_s=60)
    raise ValueError(f"unsupported provider: {provider!r}")


async def vision_json(
    provider: str,
    api_key: str,
    model: str,
    *,
    system: str,
    user_text: str,
    image_jpeg: bytes,
    max_tokens: int = 8000,
) -> CallResult:
    """One picture and a question the model answers in JSON; the caller's prompt
    asks for the JSON and the caller parses it. 8000 tokens leave a reasoning
    model room to think before the visible answer."""
    b64 = base64.standard_b64encode(image_jpeg).decode("ascii")
    if provider == "anthropic":
        return await _anthropic_call(api_key, {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": user_text},
            ]}],
        }, http_timeout_s=90)
    if provider == "openai":
        return await _openai_call(api_key, {
            "model": model,
            "max_completion_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": user_text},
                ]},
            ],
        }, http_timeout_s=180)
    raise ValueError(f"unsupported provider: {provider!r}")


def _failed(latency_ms: int, error: str) -> CallResult:
    return CallResult(ok=False, text=None, model=None, latency_ms=latency_ms, error=error)


async def _post(provider: str, url: str, headers: dict[str, str], payload: dict[str, Any],
                http_timeout_s: float) -> tuple[dict[str, Any] | CallResult, int]:
    start = time.monotonic()
    try:
        async with _client(http_timeout_s) as client:
            r = await client.post(url, headers=headers, json=payload)
    except httpx.HTTPError as e:
        return _failed(int((time.monotonic() - start) * 1000), failure_text(provider, None, _network(e))), 0
    latency_ms = int((time.monotonic() - start) * 1000)
    if r.status_code != 200:
        return _failed(latency_ms, failure_text(provider, r.status_code, _error_detail(r))), latency_ms
    try:
        return r.json(), latency_ms
    except ValueError:
        return _failed(latency_ms, f"non-JSON response from {PROVIDER_NAMES[provider]}"), latency_ms


async def _anthropic_call(api_key: str, payload: dict[str, Any], *, http_timeout_s: float = 30) -> CallResult:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    body, latency_ms = await _post("anthropic", ANTHROPIC_URL, headers, payload, http_timeout_s)
    if isinstance(body, CallResult):
        return body
    blocks = body.get("content") or []
    text = "".join(
        b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"
    ).strip()
    usage = body.get("usage") or {}
    stop_reason = body.get("stop_reason")
    error = None
    if stop_reason in ("max_tokens", "refusal"):
        category = (body.get("stop_details") or {}).get("category")
        error = (
            f"stop_reason={stop_reason}"
            + (f" ({category})" if category else "")
            + f", output_tokens={usage.get('output_tokens')}"
        )
    return CallResult(
        ok=error is None,
        text=text,
        model=body.get("model"),
        latency_ms=latency_ms,
        error=error,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
    )


async def _openai_call(api_key: str, payload: dict[str, Any], *, http_timeout_s: float = 30) -> CallResult:
    headers = {"authorization": f"Bearer {api_key}", "content-type": "application/json"}
    body, latency_ms = await _post("openai", OPENAI_URL, headers, payload, http_timeout_s)
    if isinstance(body, CallResult):
        return body
    choices = body.get("choices") or []
    text = ""
    finish_reason = None
    if choices and isinstance(choices[0], dict):
        finish_reason = choices[0].get("finish_reason")
        content = (choices[0].get("message") or {}).get("content")
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text = "".join(
                p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
            ).strip()
    usage = body.get("usage") or {}
    if not text and finish_reason == "length":
        # A reasoning model spent the whole budget thinking and said nothing.
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        return CallResult(
            ok=False,
            text="",
            model=body.get("model"),
            latency_ms=latency_ms,
            error=(
                f"empty content (finish_reason=length, reasoning_tokens={reasoning}, "
                f"completion_tokens={usage.get('completion_tokens')})"
            ),
        )
    if not text:
        log.info("openai empty content: finish_reason=%s usage=%s", finish_reason, usage)
    return CallResult(
        ok=True,
        text=text,
        model=body.get("model"),
        latency_ms=latency_ms,
        error=None,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
    )


# --- settings -----------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Setting:
    provider: str
    # None: the stored key will not decrypt (rotated secret, corrupt row)
    api_key: str | None
    enabled: bool
    models: dict[str, str]
    last_used_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(slots=True, frozen=True)
class Pick:
    provider: str
    api_key: str
    model: str


_COLUMNS = (
    "provider, api_key_encrypted, enabled, models::text AS models, "
    "last_used_at, created_at, updated_at"
)


class AiSettings:
    """The product's `ai_settings` table, read and written with the key sealed
    by the product's own `encrypt` and opened by its `decrypt` (None when it
    will not open). `providers` in the order a role falls back through them."""

    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        providers: tuple[str, ...],
        roles: tuple[Role, ...],
        encrypt: Callable[[str], bytes],
        decrypt: Callable[[bytes], str | None],
    ) -> None:
        self.pool = pool
        self.providers = providers
        self.roles = roles
        self._encrypt = encrypt
        self._decrypt = decrypt

    def _setting(self, row: asyncpg.Record) -> Setting:
        return Setting(
            provider=row["provider"],
            api_key=self._decrypt(bytes(row["api_key_encrypted"])),
            enabled=row["enabled"],
            models=json.loads(row["models"]),
            last_used_at=row["last_used_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    async def all(self) -> list[Setting]:
        rows = await self.pool.fetch(f"SELECT {_COLUMNS} FROM ai_settings ORDER BY provider")  # noqa: S608
        return [self._setting(r) for r in rows]

    async def get(self, provider: str) -> Setting | None:
        row = await self.pool.fetchrow(
            f"SELECT {_COLUMNS} FROM ai_settings WHERE provider = $1", provider  # noqa: S608
        )
        return self._setting(row) if row else None

    async def put(self, provider: str, *, api_key: str | None, enabled: bool,
                  models: dict[str, str]) -> Setting:
        """`api_key` None keeps the stored key; a new row must bring one."""
        # The stored key is looked up in the VALUES, not left to the conflict
        # branch: NOT NULL is checked before ON CONFLICT is.
        row = await self.pool.fetchrow(
            f"""
            INSERT INTO ai_settings (provider, api_key_encrypted, enabled, models)
            VALUES ($1, COALESCE($2::bytea, (SELECT api_key_encrypted FROM ai_settings WHERE provider = $1)),
                    $3, $4::text::jsonb)
            ON CONFLICT (provider) DO UPDATE SET
                api_key_encrypted = EXCLUDED.api_key_encrypted,
                enabled = EXCLUDED.enabled,
                models = EXCLUDED.models,
                updated_at = now()
            RETURNING {_COLUMNS}
            """,  # noqa: S608
            provider, self._encrypt(api_key) if api_key else None, enabled, json.dumps(models),
        )
        return self._setting(row)

    async def delete(self, provider: str) -> bool:
        return not (await self.pool.execute(
            "DELETE FROM ai_settings WHERE provider = $1", provider
        )).endswith(" 0")

    async def touch(self, provider: str) -> None:
        await self.pool.execute("UPDATE ai_settings SET last_used_at = now() WHERE provider = $1", provider)

    async def pick(self, role: str, provider: str | None = None) -> Pick:
        """The provider, key and model a call in `role` goes to: `provider` if
        named, else the first enabled one in the product's order."""
        for p in (provider,) if provider else self.providers:
            s = await self.get(p)
            if s is None or not s.enabled:
                continue
            name = PROVIDER_NAMES.get(p, p)
            if s.api_key is None:
                raise AiNotConfigured(f"the stored {name} key will not decrypt — enter it again in Settings → AI")
            model = s.models.get(role)
            if not model:
                raise AiNotConfigured(f"no {name} model is chosen for {role} — choose one in Settings → AI")
            return Pick(p, s.api_key, model)
        if provider:
            raise AiNotConfigured(f"{PROVIDER_NAMES.get(provider, provider)} is not configured — open Settings → AI")
        raise AiNotConfigured("no AI provider is configured — open Settings → AI")


# --- routes -------------------------------------------------------------------


class SettingOut(BaseModel):
    provider: str
    api_key_masked: str
    enabled: bool
    models: dict[str, str]
    last_used_at: datetime | None
    created_at: datetime
    updated_at: datetime


class SettingsOut(BaseModel):
    providers: list[str]
    roles: list[str]
    settings: list[SettingOut]


class SettingIn(BaseModel):
    """`api_key` omitted keeps the stored key, so a model can be changed without
    pasting the key again; removing a key is DELETE, never an empty string."""

    provider: str = Field(..., min_length=1, max_length=32)
    api_key: str | None = Field(None, min_length=10, max_length=512)
    enabled: bool = True
    models: dict[str, str]


class ModelsIn(BaseModel):
    """`api_key` lists with a key not yet saved; omitted, the stored key is used."""

    provider: str = Field(..., min_length=1, max_length=32)
    api_key: str | None = Field(None, min_length=10, max_length=512)


class ModelOut(BaseModel):
    id: str
    label: str
    roles: list[str]


class ModelsOut(BaseModel):
    models: list[ModelOut]


class TestIn(BaseModel):
    provider: str = Field(..., min_length=1, max_length=32)
    api_key: str | None = Field(None, min_length=10, max_length=512)
    models: list[str] = Field(default_factory=list, max_length=8)


class TestResult(BaseModel):
    model: str
    ok: bool
    latency_ms: int
    error: str | None


class TestOut(BaseModel):
    ok: bool
    results: list[TestResult]


def _out(s: Setting) -> SettingOut:
    return SettingOut(
        provider=s.provider,
        api_key_masked=mask_key(s.api_key),
        enabled=s.enabled,
        models=s.models,
        last_used_at=s.last_used_at,
        created_at=s.created_at,
        updated_at=s.updated_at,
    )


def _known(settings: AiSettings, provider: str) -> None:
    if provider not in settings.providers:
        raise HTTPException(400, f"unsupported provider: {provider!r}")


async def _key_for(settings: AiSettings, provider: str, typed: str | None) -> str:
    _known(settings, provider)
    if typed:
        return typed.strip()
    s = await settings.get(provider)
    if s is None:
        raise HTTPException(400, "no stored key; pass api_key")
    if s.api_key is None:
        raise HTTPException(400, f"the stored {PROVIDER_NAMES.get(provider, provider)} key "
                                 "will not decrypt — enter it again")
    return s.api_key


async def _put(settings: AiSettings, body: SettingIn) -> SettingOut:
    _known(settings, body.provider)
    roles = {r.key for r in settings.roles}
    chosen = {k: v.strip() for k, v in body.models.items() if v.strip()}
    if missing := sorted(roles - set(chosen)):
        raise HTTPException(400, f"choose a model for every role (missing: {', '.join(missing)})")
    if unknown := sorted(set(chosen) - roles):
        raise HTTPException(400, f"unknown roles: {', '.join(unknown)}")
    key = body.api_key.strip() if body.api_key else None
    if not key and await settings.get(body.provider) is None:
        raise HTTPException(400, "an API key is required for a provider not yet configured")
    return _out(await settings.put(body.provider, api_key=key, enabled=body.enabled, models=chosen))


async def _models_for_roles(settings: AiSettings, body: ModelsIn) -> ModelsOut:
    try:
        found = await models(body.provider, await _key_for(settings, body.provider, body.api_key))
    except ProviderModelsError as e:
        raise HTTPException(502, str(e)) from e
    out = []
    for m in found:
        if serves := [r.key for r in settings.roles if m.serves(r)]:
            out.append(ModelOut(id=m.id, label=m.label, roles=serves))
    return ModelsOut(models=out)


async def _test(settings: AiSettings, body: TestIn) -> TestOut:
    key = await _key_for(settings, body.provider, body.api_key)
    wanted = list(dict.fromkeys(m for m in body.models if m))
    if not wanted and (s := await settings.get(body.provider)):
        wanted = list(dict.fromkeys(s.models.values()))
    if not wanted:
        raise HTTPException(400, "no model to test; pass models")
    results = []
    for model in wanted:
        r = await ping(body.provider, key, model)
        results.append(TestResult(model=r.model or model, ok=r.ok, latency_ms=r.latency_ms, error=r.error))
    ok = all(r.ok for r in results)
    if ok and not body.api_key:
        await settings.touch(body.provider)
    return TestOut(ok=ok, results=results)


def ai_router(settings_of: Callable[[Request], AiSettings]) -> APIRouter:
    """GET/PUT /ai/settings, DELETE /ai/settings/{provider}, POST /ai/models and
    POST /ai/test — the contract the kit's AI cards talk to. `settings_of` finds
    the product's AiSettings for a request (it is built once the pool exists)."""
    router = APIRouter(tags=["ai"])

    @router.get("/ai/settings", response_model=SettingsOut)
    async def list_settings(request: Request) -> SettingsOut:
        settings = settings_of(request)
        return SettingsOut(
            providers=list(settings.providers),
            roles=[r.key for r in settings.roles],
            settings=[_out(s) for s in await settings.all() if s.provider in settings.providers],
        )

    @router.put("/ai/settings", response_model=SettingOut)
    async def put_setting(body: SettingIn, request: Request) -> SettingOut:
        return await _put(settings_of(request), body)

    @router.delete("/ai/settings/{provider}", status_code=204)
    async def delete_setting(provider: str, request: Request) -> Response:
        settings = settings_of(request)
        _known(settings, provider)
        if not await settings.delete(provider):
            raise HTTPException(404, "provider not configured")
        return Response(status_code=204)

    @router.post("/ai/models", response_model=ModelsOut)
    async def list_models(body: ModelsIn, request: Request) -> ModelsOut:
        return await _models_for_roles(settings_of(request), body)

    @router.post("/ai/test", response_model=TestOut)
    async def test(body: TestIn, request: Request) -> TestOut:
        return await _test(settings_of(request), body)

    return router
