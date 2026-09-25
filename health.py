"""File-based liveness for services without HTTP.

The service touches a file in its main loop and the compose healthcheck stats
its mtime. One that hangs in a native call or stops consuming its bus stops
refreshing the file, the mtime drifts past the threshold and docker restarts
the container. The file lives in the container's own /tmp, off the mounted
volumes.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

log = logging.getLogger(__name__)


class HealthMarker:
    """`/tmp/<product>_healthy_<service>` unless `path` says otherwise."""

    def __init__(self, product: str, service: str, *, path: str | os.PathLike | None = None) -> None:
        self.product = product
        self.service = service
        self.path = Path(path if path is not None else f"/tmp/{product}_healthy_{service}")
        self._ensured = False

    def _ensure(self) -> None:
        if self._ensured:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(f"{self.product}_healthy service={self.service} pid={os.getpid()}\n")
            self._ensured = True
        except OSError as e:
            log.warning("health: cannot create %s: %s", self.path, e)

    def touch(self) -> None:
        self._ensure()
        try:
            now = time.time()
            os.utime(self.path, (now, now))
        except OSError as e:
            log.debug("health: utime(%s) failed: %s", self.path, e)

    async def run_loop(self, interval: float = 5.0) -> None:
        while True:
            self.touch()
            await asyncio.sleep(interval)
