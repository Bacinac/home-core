# home-core

What DIDA and BABA do the same way on the server.

## Why this exists

The two grew from one codebase and each copied the other's plumbing. The copies
then drifted: one product's background tasks logged their failures and the
other's vanished; one verified passwords off the event loop and the other
stalled on them; three plugin registries differed only in their names. A fix
reached one product and not the other. So what must not differ lives here once.

What each product decides for itself — who a user is, what a role may do, how a
session is revoked, which proxies it trusts, what its plugins are — stays with
the product and is passed in.

## What is in it

- `tasks` — `spawn()`: fire-and-forget work held strongly until it ends, its
  failure logged; `cancel_all_tasks()` for an orderly shutdown
- `health` — `HealthMarker(product, service)`: the file a compose healthcheck
  stats, `/tmp/<product>_healthy_<service>`
- `plugins` — `Plugins(group, kind, accepts)`: plugins found by entry point,
  each gated on `is_available()`, `get`, `names`, `select`
- `rate_limit` — `TokenBucketLimiter`; `TrustedPeers` (IPs, CIDRs, compose
  service names, or `private`) and `client_ip()`, which honours
  X-Forwarded-For only from a trusted peer and walks it to the first hop that
  is not a trusted proxy; `LoginLimits`: five guesses a minute per client, ten
  per account
- `auth` — argon2 `hash_password` / `verify_password` (off the event loop; no
  such user costs the same), `load_or_create_secret`, the signed session token
  (`encode_session_token` / `decode_session_token`), `SessionCookie` (HttpOnly,
  Lax, Secure over HTTPS), `bootstrap_admin_if_empty`

`rate_limit` and `auth` are for a product's api and need what it already
carries: FastAPI, argon2-cffi, PyJWT and asyncpg. The rest is the standard
library.

## How a product consumes it

A git submodule at `core/src/home_core`, packaged with the product's core:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/dida_core", "src/home_core"]
```

There is no standalone clone: change it inside one product's submodule, run
the tests there, commit and push, then bump the pointer in the other product.
The tests are `tests/` here, run by each product's gate.

## Licence

[PolyForm Noncommercial 1.0.0](LICENSE.md) — free for personal and
noncommercial use; a commercial licence on request. Outside contributions
(pull requests) are not taken. Required Notice: Copyright (c) 2026 Ivo Bošković.
