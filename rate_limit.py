"""Token buckets, the client a request is keyed on, and the login limits.

Single process, no store of its own: each bucket table is an LRU capped at
`max_keys`, so a caller cycling synthetic addresses cannot grow it without
bound.

A forwarding header counts only when the peer that delivered it is trusted:
honoured from anyone, it would let any host mint a fresh bucket per request.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import time
from collections import OrderedDict
from dataclasses import dataclass

from fastapi import HTTPException, Request

log = logging.getLogger(__name__)


@dataclass(slots=True)
class _Bucket:
    tokens: float
    last_refill: float


class TokenBucketLimiter:
    """capacity = the burst, refill_per_s = the sustained rate: "5 a minute" is
    capacity=5, refill_per_s=5/60 — five at once, then one every 12 seconds."""

    def __init__(self, *, capacity: int, refill_per_s: float, max_keys: int = 10_000) -> None:
        self._capacity = float(capacity)
        self._refill = refill_per_s
        self._max_keys = max_keys
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()

    def take(self, key: str, cost: float = 1.0) -> bool:
        now = time.monotonic()
        b = self._buckets.get(key)
        if b is None:
            b = _Bucket(tokens=self._capacity, last_refill=now)
            self._buckets[key] = b
            while len(self._buckets) > self._max_keys:
                self._buckets.popitem(last=False)
        else:
            self._buckets.move_to_end(key)
            elapsed = now - b.last_refill
            if elapsed > 0:
                b.tokens = min(self._capacity, b.tokens + elapsed * self._refill)
                b.last_refill = now
        if b.tokens >= cost:
            b.tokens -= cost
            return True
        return False

    def clear(self) -> None:
        self._buckets.clear()


class TrustedPeers:
    """The peers allowed to name the client in X-Forwarded-For.

    Entries are IPs, CIDRs, hostnames, or `private` for every private and
    loopback address. A hostname is normally the compose service of the proxy
    in front and is re-resolved every `ttl_s`, because a recreated container
    comes back on a new address. An empty spec trusts nobody."""

    def __init__(self, spec: str, ttl_s: float = 30.0) -> None:
        self._nets: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        self._names: list[str] = []
        self._private = False
        for entry in (e.strip() for e in spec.split(",")):
            if not entry:
                continue
            if entry == "private":
                self._private = True
                continue
            try:
                self._nets.append(ipaddress.ip_network(entry, strict=False))
            except ValueError:
                self._names.append(entry)
        self._ttl_s = ttl_s
        self._resolved: frozenset[str] = frozenset()
        self._resolved_at = float("-inf")

    def __contains__(self, peer: str) -> bool:
        try:
            addr = ipaddress.ip_address(peer)
        except ValueError:
            return False
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
            addr = addr.ipv4_mapped
        if self._private and (addr.is_private or addr.is_loopback):
            return True
        return any(addr in net for net in self._nets) or str(addr) in self._resolved_names()

    def _resolved_names(self) -> frozenset[str]:
        now = time.monotonic()
        if self._names and now - self._resolved_at >= self._ttl_s:
            ips: set[str] = set()
            for name in self._names:
                try:
                    ips.update(str(info[4][0]) for info in socket.getaddrinfo(name, None))
                except OSError as exc:
                    log.warning("trusted proxy %r does not resolve: %s", name, exc)
            self._resolved = frozenset(ips)
            self._resolved_at = now
        return self._resolved


def forwarded_client(peer: str | None, xff: str | None, trusted: TrustedPeers) -> str:
    """The peer itself, unless it is trusted — then the rightmost X-Forwarded-For
    hop that is not itself a trusted proxy. Each trusted proxy appends the
    address it received from; everything left of the first untrusted hop was
    written by the client and is whatever it chose."""
    if peer is None:
        return "?"
    if peer not in trusted:
        return peer
    hops = [h.strip() for h in (xff or "").split(",") if h.strip()]
    for hop in reversed(hops):
        if hop not in trusted:
            return hop
    return hops[0] if hops else peer


def client_ip(request: Request, trusted: TrustedPeers) -> str:
    peer = request.client.host if request.client is not None else None
    return forwarded_client(peer, request.headers.get("x-forwarded-for"), trusted)


_RETRY_AFTER = {"Retry-After": "15"}


class LoginLimits:
    """Five guesses a minute per client, ten per account.

    The account cap is what holds when the client address can be rotated —
    a distributed guess at one user, or a trusted network whose hosts may
    write their own forwarding header — and is case-folded so mixed-case
    retries do not multiply it. Both sit far below what a guessing attack
    needs against argon2 and far above a person mistyping across two
    devices."""

    def __init__(self, trusted: TrustedPeers) -> None:
        self.trusted = trusted
        self.per_client = TokenBucketLimiter(capacity=5, refill_per_s=5 / 60.0)
        self.per_account = TokenBucketLimiter(capacity=10, refill_per_s=10 / 60.0)

    def enforce_client(self, request: Request) -> None:
        """FastAPI dependency of the login endpoint."""
        ip = client_ip(request, self.trusted)
        if not self.per_client.take(ip):
            log.warning("rate limit: login attempts exhausted for %s", ip)
            raise HTTPException(429, "too many login attempts; try again in a few seconds", headers=_RETRY_AFTER)

    def enforce_account(self, username: str) -> None:
        """Called once the body is parsed, before the password is verified."""
        key = username.strip().lower()
        if key and not self.per_account.take(key):
            log.warning("rate limit: login attempts exhausted for user %r", key)
            raise HTTPException(429, "too many login attempts; try again in a few seconds", headers=_RETRY_AFTER)

    def clear(self) -> None:
        self.per_client.clear()
        self.per_account.clear()
