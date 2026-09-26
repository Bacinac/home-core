"""Plugins found by entry point.

A plugin package names its class under a group in its pyproject:

    [project.entry-points."baba.backends"]
    onnxruntime = "baba_backend_onnxruntime:ONNXRuntimeBackend"

Every installed plugin is loaded and asked `is_available()`, so a missing
dependency disables that one plugin instead of stopping discovery.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import entry_points
from threading import RLock

log = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class _Entry[T]:
    name: str
    cls: type[T]
    available: bool


class Plugins[T]:
    """The plugins of one entry-point group. `kind` names them in logs and
    errors; `accepts` decides whether a loaded object is one."""

    def __init__(self, group: str, kind: str, accepts: Callable[[object], bool]) -> None:
        self.group = group
        self.kind = kind
        self._accepts = accepts
        self._entries: dict[str, _Entry[T]] = {}
        self._lock = RLock()
        self._discovered = False

    def discover(self) -> None:
        """Scan the installed packages once."""
        with self._lock:
            if self._discovered:
                return
            for ep in entry_points(group=self.group):
                try:
                    cls = ep.load()
                except Exception as exc:
                    log.warning("Failed to load %s %r: %s", self.kind, ep.name, exc, exc_info=True)
                    continue
                if not (isinstance(cls, type) and self._accepts(cls)):
                    log.warning("Entry point %r is not a %s", ep.name, self.kind)
                    continue
                self._entries[ep.name] = _Entry(ep.name, cls, self._available(ep.name, cls))
                log.info("Discovered %s %r (available=%s)", self.kind, ep.name, self._entries[ep.name].available)
            self._discovered = True

    def register(self, name: str, cls: type[T]) -> None:
        """Explicit registration, for tests and in-tree plugins."""
        with self._lock:
            self._entries[name] = _Entry(name, cls, self._available(name, cls))

    def _available(self, name: str, cls: type[T]) -> bool:
        try:
            return bool(cls.is_available())  # type: ignore[attr-defined]
        except Exception as exc:
            log.warning(
                "%s %r is_available() raised: %s — marking unavailable", self.kind, name, exc, exc_info=True
            )
            return False

    def names(self, only_available: bool = False) -> list[str]:
        self.discover()
        return [e.name for e in self._entries.values() if not only_available or e.available]

    def get(self, name: str) -> type[T]:
        self.discover()
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(f"{self.kind} {name!r} not registered. Known: {self.names()}")
        return entry.cls

    def select(self, preference: list[str]) -> type[T]:
        """The first available plugin of `preference`."""
        self.discover()
        for name in preference:
            entry = self._entries.get(name)
            if entry is not None and entry.available:
                return entry.cls
        raise RuntimeError(
            f"No available {self.kind} in preference list {preference}. "
            f"Available: {self.names(only_available=True)}"
        )
