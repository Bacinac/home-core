"""A spawned task that fails is logged; one that is cancelled is not."""

import asyncio
import logging

from home_core.tasks import _TASKS, cancel_all_tasks, spawn


def test_a_failure_is_logged_and_the_task_released(caplog):
    async def boom():
        raise RuntimeError("publish failed")

    async def main():
        t = spawn(boom(), name="publish")
        await asyncio.gather(t, return_exceptions=True)
        await asyncio.sleep(0)
        return t

    with caplog.at_level(logging.ERROR):
        t = asyncio.run(main())
    assert t not in _TASKS
    assert any("'publish' failed: publish failed" in r.getMessage() for r in caplog.records)


def test_cancel_all_tasks_tears_down_silently(caplog):
    async def main():
        spawn(asyncio.sleep(3600), name="loop")
        await asyncio.sleep(0)
        await cancel_all_tasks()
        return len(_TASKS)

    with caplog.at_level(logging.ERROR):
        assert asyncio.run(main()) == 0
    assert not caplog.records
