from rth.config import load_settings
from rth.stances import load_stances


def test_settings_load():
    s = load_settings()
    assert s.scoring.tolerance_deg > 0
    assert s.nebius.base_url.startswith("https://")


def test_stances_load():
    lib = load_stances()
    assert "ready" in lib
    assert len(lib.ids()) >= 6
    assert all(s.description for s in lib.stances)
