"""Fire-and-forget work that can neither vanish nor fail unseen.

A bare `asyncio.create_task` holds the task only weakly, so the loop may
collect it mid-flight, and its exception surfaces only as a delayed "Task
exception was never retrieved" — a failed publish simply disappears. `spawn`
keeps a strong reference until the task ends and logs its failure loudly.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

_log = logging.getLogger(__name__)

_TASKS: set[asyncio.Task[Any]] = set()


def spawn(
    coro: Coroutine[Any, Any, Any],
    *,
    name: str | None = None,
    log: logging.Logger | None = None,
) -> asyncio.Task[Any]:
    """Run `coro` as a supervised task; a failure is logged as an ERROR through
    `log`, naming the work by `name` (else the coroutine's). Cancellation is
    silent: that is an orderly shutdown."""
    task = asyncio.get_running_loop().create_task(coro, name=name)
    _TASKS.add(task)
    label = name or getattr(coro, "__qualname__", str(coro))
    lg = log or _log

    def _done(t: asyncio.Task[Any]) -> None:
        _TASKS.discard(t)
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            lg.error("background task %r failed: %s", label, exc, exc_info=exc)

    task.add_done_callback(_done)
    return task


async def cancel_all_tasks() -> None:
    """Cancel and await every spawned task still running, so a shutdown tears
    down the loops before the pools and sessions they use."""
    tasks = list(_TASKS)
    for t in tasks:
        t.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
