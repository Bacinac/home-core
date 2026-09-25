"""Passwords and the session token: what a forged, expired or foreign token
meets, and what a guess at an unknown user costs."""

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import jwt
import pytest
from fastapi import HTTPException

from home_core.auth import (
    ALGO,
    decode_session_token,
    encode_session_token,
    hash_password,
    verify_password,
)

_SECRET = "test-secret-key-0123456789abcdef"
_TTL = timedelta(days=7)
now = datetime.now(UTC)


def run(coro):
    return asyncio.run(coro)


def raises_401(*a):
    with pytest.raises(HTTPException) as exc_info:
        decode_session_token(*a)
    assert exc_info.value.status_code == 401


def test_argon2_password_hashing():
    h = run(hash_password("s3cret-passphrase"))
    assert h.startswith("$argon2")
    assert run(verify_password("s3cret-passphrase", h)) is True
    assert run(verify_password("wrong", h)) is False
    assert run(verify_password("", h)) is False
    assert run(hash_password("x")) != run(hash_password("x"))


def test_no_such_user_is_never_a_match():
    assert run(verify_password("home-core-nobody", None)) is False


def test_round_trip_keeps_the_subject_type_the_product_parses():
    assert decode_session_token(encode_session_token(42, _SECRET, token_version=7, ttl=_TTL), _SECRET, int) == (42, 7)
    uid = uuid4()
    assert decode_session_token(encode_session_token(uid, _SECRET, ttl=_TTL), _SECRET, UUID) == (uid, 0)


def test_two_sign_ins_in_one_second_are_two_tokens():
    assert encode_session_token(1, _SECRET, ttl=_TTL) != encode_session_token(1, _SECRET, ttl=_TTL)


def test_wrong_secret_garbage_and_expiry_are_refused():
    tok = encode_session_token(42, _SECRET, ttl=_TTL)
    raises_401(tok, "a-different-secret-of-fully-sufficient-length", int)
    raises_401("not.a.jwt", _SECRET, int)
    expired = jwt.encode(
        {"sub": "1", "iat": int((now - timedelta(hours=2)).timestamp()),
         "exp": int((now - timedelta(hours=1)).timestamp()), "tv": 0},
        _SECRET, algorithm=ALGO,
    )
    raises_401(expired, _SECRET, int)


def test_an_unsigned_token_is_refused():
    none_tok = jwt.encode({"sub": "1", "exp": int((now + timedelta(hours=1)).timestamp()), "tv": 0},
                          key="", algorithm="none")
    raises_401(none_tok, _SECRET, int)


def test_a_missing_or_foreign_subject_is_refused():
    no_sub = jwt.encode({"exp": int((now + timedelta(hours=1)).timestamp()), "tv": 0}, _SECRET, algorithm=ALGO)
    raises_401(no_sub, _SECRET, int)
    raises_401(encode_session_token(uuid4(), _SECRET, ttl=_TTL), _SECRET, int)
