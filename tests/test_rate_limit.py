"""The limiter's arithmetic, and a client address the caller cannot write."""

from types import SimpleNamespace

import pytest

from home_core import rate_limit
from home_core.rate_limit import LoginLimits, TokenBucketLimiter, TrustedPeers, forwarded_client


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(rate_limit, "time", SimpleNamespace(monotonic=c.monotonic))
    return c


def test_a_burst_of_five_then_one_every_twelve_seconds(clock):
    lim = TokenBucketLimiter(capacity=5, refill_per_s=5 / 60)
    assert all(lim.take("a") for _ in range(5))
    assert not lim.take("a")
    clock.now += 11.9
    assert not lim.take("a")
    clock.now += 0.2
    assert lim.take("a")
    assert not lim.take("a")


def test_an_idle_client_refills_only_to_the_burst(clock):
    lim = TokenBucketLimiter(capacity=5, refill_per_s=5 / 60)
    for _ in range(5):
        lim.take("a")
    clock.now += 3600
    assert sum(lim.take("a") for _ in range(10)) == 5


def test_one_client_exhausting_its_bucket_leaves_the_next_untouched(clock):
    lim = TokenBucketLimiter(capacity=5, refill_per_s=5 / 60)
    for _ in range(6):
        lim.take("attacker")
    assert lim.take("neighbour")


def test_the_key_table_is_bounded(clock):
    lim = TokenBucketLimiter(capacity=1, refill_per_s=0.0, max_keys=3)
    for k in "abcd":
        lim.take(k)
    assert len(lim._buckets) == 3
    assert "a" not in lim._buckets


WEB = TrustedPeers("172.20.0.7")


def test_a_direct_caller_is_its_own_client_whatever_it_forwards():
    assert forwarded_client("192.168.10.50", "1.2.3.4", WEB) == "192.168.10.50"


def test_the_trusted_proxy_names_the_client():
    assert forwarded_client("172.20.0.7", "203.0.113.9", WEB) == "203.0.113.9"


def test_only_the_hop_the_proxy_appended_counts():
    assert forwarded_client("172.20.0.7", "6.6.6.6, 203.0.113.9", WEB) == "203.0.113.9"


def test_a_chain_of_trusted_proxies_is_walked_to_the_first_untrusted_hop():
    both = TrustedPeers("172.20.0.7, 172.20.0.8")
    assert forwarded_client("172.20.0.7", "6.6.6.6, 203.0.113.9, 172.20.0.8", both) == "203.0.113.9"


def test_a_trusted_proxy_without_the_header_is_the_client():
    assert forwarded_client("172.20.0.7", None, WEB) == "172.20.0.7"
    assert forwarded_client("172.20.0.7", " , ", WEB) == "172.20.0.7"


def test_nothing_is_trusted_by_default():
    assert forwarded_client("172.20.0.7", "203.0.113.9", TrustedPeers("")) == "172.20.0.7"


def test_cidr_and_ipv4_mapped_peers():
    lan = TrustedPeers("10.0.0.0/8")
    assert forwarded_client("10.1.2.3", "203.0.113.9", lan) == "203.0.113.9"
    assert forwarded_client("::ffff:10.1.2.3", "203.0.113.9", lan) == "203.0.113.9"
    assert forwarded_client("11.1.2.3", "203.0.113.9", lan) == "11.1.2.3"


def test_private_trusts_every_internal_hop_and_no_public_one():
    internal = TrustedPeers("private")
    # the tunnel: the client, then the host the tunnel reached the proxy from
    assert forwarded_client("172.18.0.5", "8.8.8.8, 192.168.1.100", internal) == "8.8.8.8"
    assert forwarded_client("127.0.0.1", "8.8.8.8", internal) == "8.8.8.8"
    assert forwarded_client("8.8.8.8", "10.0.0.1", internal) == "8.8.8.8"


def test_a_hostname_is_resolved_to_the_peer_it_names():
    assert forwarded_client("127.0.0.1", "203.0.113.9", TrustedPeers("localhost")) == "203.0.113.9"


def test_an_unresolvable_name_trusts_nobody():
    ghost = TrustedPeers("no-such-proxy.invalid")
    assert forwarded_client("172.20.0.7", "203.0.113.9", ghost) == "172.20.0.7"


def test_no_peer_at_all():
    assert forwarded_client(None, "203.0.113.9", WEB) == "?"


def test_one_account_is_guarded_across_rotating_clients(clock):
    limits = LoginLimits(TrustedPeers(""))
    for _ in range(10):
        limits.enforce_account("Ivo")
    with pytest.raises(Exception) as refused:
        limits.enforce_account("IVO ")
    assert refused.value.status_code == 429
    limits.enforce_account("someone-else")
