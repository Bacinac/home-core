"""The liveness file the compose healthcheck stats."""

import os

from home_core.health import HealthMarker


def test_the_file_is_named_for_its_product_and_refreshed(tmp_path):
    assert str(HealthMarker("dida", "engine").path) == "/tmp/dida_healthy_engine"
    m = HealthMarker("baba", "detector", path=tmp_path / "h")
    m.touch()
    assert (tmp_path / "h").read_text().startswith("baba_healthy service=detector")
    os.utime(tmp_path / "h", (0, 0))
    m.touch()
    assert (tmp_path / "h").stat().st_mtime > 0


def test_an_unwritable_place_does_not_stop_the_service(tmp_path):
    (tmp_path / "ro").mkdir(mode=0o500)
    HealthMarker("dida", "x", path=tmp_path / "ro" / "sub" / "h").touch()
