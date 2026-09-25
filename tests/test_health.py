"""The liveness file a compose healthcheck stats: a service touches it in its
main loop, and a stopped loop lets the mtime drift past the threshold. These
write to tmp_path, never a running service's own file, and age the mtime
directly."""

import asyncio
import os
import time

import pytest

from home_core.health import HealthMarker

# The compose healthchecks' threshold: a file older than this is "stale".
STALE_THRESHOLD_S = 30.0


def test_health_marker_default_path_is_tmp_service():
    # No file is written by __init__ — only the path is derived from the service name.
    marker = HealthMarker("dida", "engine")
    assert str(marker.path) == "/tmp/dida_healthy_engine", "default path is /tmp/<product>_healthy_<svc>"


def test_health_marker_touch_writes_marker_file(tmp_path):
    path = tmp_path / "beat_engine"
    marker = HealthMarker("dida", "engine", path=path)
    marker.touch()
    assert path.exists(), "touch() creates the liveness file"
    content = path.read_text()
    assert f"dida_healthy service=engine pid={os.getpid()}" in content, \
        "marker records the service name and the writing process pid"
    assert content.endswith("\n"), "marker line is newline-terminated"


def test_health_marker_touch_creates_missing_parent_dir(tmp_path):
    path = tmp_path / "nested" / "sub" / "beat"
    marker = HealthMarker("dida", "adapter", path=path)
    marker.touch()
    assert path.parent.is_dir(), "_ensure() mkdir -p's the parent directory"
    assert path.exists(), "the marker file lands inside the freshly-created dir"


def test_health_marker_touch_refreshes_a_fresh_mtime(tmp_path):
    path = tmp_path / "beat"
    marker = HealthMarker("dida", "svc", path=path)
    # Simulate a hung service: create the file, then age its mtime past the threshold.
    marker.touch()
    stale = time.time() - 3600
    os.utime(path, (stale, stale))
    age_before = time.time() - path.stat().st_mtime
    assert age_before > STALE_THRESHOLD_S, "artificially-aged file reads as stale"
    # The main loop resumes touching -> mtime jumps back to ~now -> healthy again.
    marker.touch()
    age_after = time.time() - path.stat().st_mtime
    assert age_after < STALE_THRESHOLD_S, "a fresh touch() brings the file back inside the window"


def test_health_marker_ensure_writes_body_only_once(tmp_path):
    path = tmp_path / "beat"
    marker = HealthMarker("dida", "svc", path=path)
    marker.touch()  # first touch writes the body and latches _ensured
    # Overwrite the body out-of-band; a second touch must only bump mtime, not rewrite.
    path.write_text("MUTATED-EXTERNALLY")
    marker.touch()
    assert path.read_text() == "MUTATED-EXTERNALLY", \
        "_ensure() is idempotent — touch() after the first does not rewrite the body"


def test_health_marker_run_loop_touches_then_cancels(tmp_path):
    path = tmp_path / "beat"
    marker = HealthMarker("dida", "svc", path=path)

    async def main():
        task = asyncio.create_task(marker.run_loop(interval=0.01))
        await asyncio.sleep(0.03)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(main())
    assert path.exists(), "run_loop touches the file on its first iteration"
    assert time.time() - path.stat().st_mtime < STALE_THRESHOLD_S, \
        "the mtime is fresh after the loop ran"


def test_an_unwritable_place_does_not_stop_the_service(tmp_path):
    (tmp_path / "ro").mkdir(mode=0o500)
    HealthMarker("dida", "x", path=tmp_path / "ro" / "sub" / "h").touch()
