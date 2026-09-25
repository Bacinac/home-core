"""Entry-point discovery: every way a plugin can be broken leaves the others
found, and asking twice scans once."""

import pytest

from home_core import plugins
from home_core.plugins import Plugins


class Good:
    @classmethod
    def is_available(cls) -> bool:
        return True


class Flaky:
    @classmethod
    def is_available(cls) -> bool:
        raise RuntimeError("driver missing")


class Missing:
    @classmethod
    def is_available(cls) -> bool:
        return False


class _EP:
    def __init__(self, name, target):
        self.name, self._target = name, target

    def load(self):
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


def _accepts(cls) -> bool:
    return hasattr(cls, "is_available")


@pytest.fixture
def found(monkeypatch):
    scans = []

    def entry_points(group):
        scans.append(group)
        return [
            _EP("good", Good), _EP("flaky", Flaky), _EP("missing", Missing),
            _EP("broken", ImportError("no module")), _EP("value", 42), _EP("other", object),
        ]

    monkeypatch.setattr(plugins, "entry_points", entry_points)
    return scans


def test_discovery_keeps_what_loads_and_gates_it(found):
    reg = Plugins("x.plugins", "plugin", _accepts)
    assert reg.names() == ["good", "flaky", "missing"]
    assert reg.names(only_available=True) == ["good"]
    assert reg.get("good") is Good
    reg.names()
    assert found == ["x.plugins"]


def test_an_unknown_name_says_what_is_known(found):
    with pytest.raises(KeyError, match="good"):
        Plugins("x.plugins", "plugin", _accepts).get("nope")


def test_select_takes_the_first_available_preference(found):
    reg = Plugins("x.plugins", "decoder", _accepts)
    assert reg.select(["missing", "flaky", "good"]) is Good
    with pytest.raises(RuntimeError, match="No available decoder"):
        reg.select(["missing", "flaky"])


def test_register_gates_on_availability(found):
    reg = Plugins("x.plugins", "plugin", _accepts)
    reg.register("late", Flaky)
    assert "late" in reg.names() and "late" not in reg.names(only_available=True)
