"""Passwords, the server's secret, the signed session token and the cookie that
carries it. Who a user is, what a role may do and how a session is revoked are
each product's own; these are the parts that must not differ.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import asyncpg
import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import HTTPException, Request, Response

log = logging.getLogger(__name__)

ALGO = "HS256"

_hasher = PasswordHasher()

# Verified against when the username does not exist: without it that answer
# returns before any argon2 work and is told apart from a wrong password by its
# time alone, which enumerates the accounts.
_NOBODY = _hasher.hash("home-core-nobody")


def _verify(plain: str, hashed: str) -> bool:
    try:
        _hasher.verify(hashed, plain)
        return True
    except VerifyMismatchError:
        return False
    except Exception:
        log.exception("argon2 verify raised unexpected error")
        return False


async def hash_password(plain: str) -> str:
    return await asyncio.to_thread(_hasher.hash, plain)


async def verify_password(plain: str, hashed: str | None) -> bool:
    """Whether `plain` matches; `hashed` None — no such user — costs the same
    and is False. argon2 is deliberately slow, so it runs off the event loop."""
    ok = await asyncio.to_thread(_verify, plain, _NOBODY if hashed is None else hashed)
    return ok and hashed is not None


def load_or_create_secret(state_dir: Path, env: str) -> str:
    """The key in `env`, else one generated once and kept in the state dir, so
    it survives restarts and differs per install."""
    value = os.environ.get(env, "").strip()
    if value:
        return value
    state_dir.mkdir(parents=True, exist_ok=True)
    key_file = state_dir / "secret-key"
    if key_file.exists():
        return key_file.read_text().strip()
    key = secrets.token_hex(32)
    key_file.write_text(key)
    with contextlib.suppress(Exception):
        key_file.chmod(0o600)
    log.info("Generated new secret key at %s (chmod 600)", key_file)
    return key


def encode_session_token(sub: object, secret: str, *, token_version: int = 0, ttl: timedelta) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": str(sub),
        "iat": int(now.timestamp()),
        "exp": int((now + ttl).timestamp()),
        "tv": int(token_version),
        # Two sign-ins in the same second would otherwise mint the same token.
        "jti": secrets.token_urlsafe(12),
    }
    return jwt.encode(payload, secret, algorithm=ALGO)


def decode_session_token[T](token: str, secret: str, parse: Callable[[str], T]) -> tuple[T, int]:
    """(subject, token version) of a token this server signed, or a 401."""
    try:
        payload = jwt.decode(token, secret, algorithms=[ALGO])
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "session expired") from None
    except jwt.InvalidTokenError:
        raise HTTPException(401, "invalid session") from None
    try:
        sub = parse(payload["sub"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(401, "malformed session token") from None
    return sub, int(payload.get("tv", 0))


def request_is_https(request: Request | None) -> bool:
    if request is None:
        return False
    proto = request.headers.get("x-forwarded-proto", "")
    if proto:
        return proto.split(",", 1)[0].strip().lower() == "https"
    return request.url.scheme == "https"


@dataclass(frozen=True, slots=True)
class SessionCookie:
    """HttpOnly, so a script cannot read the token; SameSite=Lax, so a
    cross-site form does not carry it; Secure whenever the request arrived over
    HTTPS, so the token is never sent back in the clear on that path while plain
    HTTP on the LAN still works. `always_secure` forces Secure on."""

    name: str
    ttl: timedelta
    always_secure: bool = False

    def set(self, response: Response, token: str, request: Request | None = None) -> None:
        response.set_cookie(
            key=self.name,
            value=token,
            max_age=int(self.ttl.total_seconds()),
            httponly=True,
            samesite="lax",
            secure=self.always_secure or request_is_https(request),
            path="/",
        )

    def clear(self, response: Response) -> None:
        response.delete_cookie(self.name, path="/")


def _write_bootstrap_secret(state_dir: Path, username: str, password: str) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    secret_file = state_dir / "bootstrap_admin_password"
    tmp = secret_file.with_suffix(".tmp")
    tmp.write_text(
        f"{password}\n"
        f"# Bootstrap admin password for user {username!r}.\n"
        f"# Log in, change the password from /account, then delete this file.\n"
    )
    # A network share may refuse chmod; the file is still better than a log line.
    with contextlib.suppress(OSError):
        tmp.chmod(0o600)
    tmp.replace(secret_file)
    return secret_file


async def bootstrap_admin_if_empty(pool: asyncpg.Pool, state_dir: Path, *, env_prefix: str) -> None:
    """With no users at all, create the admin named by `<prefix>_ADMIN_USERNAME`
    with `<prefix>_ADMIN_PASSWORD`. A generated password is never logged: it is
    written to `bootstrap_admin_password` in the state dir (mode 0600), to be
    read once, used to sign in and changed."""
    if await pool.fetchval("SELECT count(*) FROM users"):
        return
    username = os.environ.get(f"{env_prefix}_ADMIN_USERNAME", "admin").strip() or "admin"
    password = os.environ.get(f"{env_prefix}_ADMIN_PASSWORD", "").strip()
    generated = not password
    if generated:
        password = secrets.token_urlsafe(18)
    await pool.execute(
        "INSERT INTO users (username, password_hash, role) VALUES ($1, $2, 'admin')",
        username,
        await hash_password(password),
    )
    if not generated:
        log.info("Bootstrap admin created from env. username=%s", username)
        return
    secret_file = await asyncio.to_thread(_write_bootstrap_secret, state_dir, username, password)
    log.warning(
        "Bootstrap admin %r created with a generated password. Read it from %s "
        "(mode 0600), log in, change it from /account, then delete that file. "
        "Set %s_ADMIN_PASSWORD to skip this on next reset.",
        username,
        secret_file,
        env_prefix,
    )
