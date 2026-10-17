import json

import numpy as np
import pytest

from hand.compare import compare
from hand.signs import Session, load_signs


def entry(id, status, target=(0.0,) * 8, movement=None):
    return {"id": id, "gloss": id.upper(), "source_url": "https://www.british-sign.co.uk/x",
            "status": status, "instruction": "do it", "target": list(target),
            "hold_seconds": 3.0, "movement": movement, "notes": ""}


def test_real_file_loads_only_shown_statuses():
    signs = load_signs()
    ids = [s.id for s in signs]
    assert len(signs) == 6
    assert "four" not in ids and "nine" not in ids
    assert {s.status for s in signs} <= {"exact", "approximated"}
    for s in signs:
        assert s.target.shape == (8,) and np.all(s.target >= -1) and np.all(s.target <= 1)


def test_filters_unverified_and_not_representable(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps([entry("a", "exact"), entry("b", "unverified"),
                             entry("c", "not_representable"), entry("d", "approximated")]))
    assert [s.id for s in load_signs(p)] == ["a", "d"]


def test_bad_target_raises_and_out_of_range_is_clamped(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps([entry("a", "exact", target=[0, 0, 0])]))
    with pytest.raises(ValueError):
        load_signs(p)
    p.write_text(json.dumps([entry("a", "exact", target=[5, 5, -5, -5, 0, 0, 0, 0])]))
    assert list(load_signs(p)[0].target[:4]) == [1.0, 1.0, 0.0, -1.0]


def test_movement_sign_is_labelled_handshape_only():
    by_id = {s.id: s for s in load_signs()}
    assert "handshape only" in by_id["sorry"].status_label
    assert "handshape only" not in by_id["one"].status_label
    assert by_id["one"].source_host == "www.british-sign.co.uk"


def run_frames(session, vec, start, seconds, dt=0.1):
    t, done = start, False
    while t < start + seconds and not done:
        _, done = session.update(compare(vec, session.sign.target), t)
        t += dt
    return t, done


def test_session_pass_advances_and_scorecard():
    signs = load_signs()
    s = Session(signs)
    first = s.sign
    t, done = run_frames(s, first.target, 0.0, 4.0)
    assert done and s.result[first.id] == "passed" and s.sign is not first
    # wrong hand for the second sign: finger counted off, no pass
    bad = first.target.copy()
    second = s.sign
    bad = np.clip(second.target + np.array([0, 0, 0.9, 0, 0, 0, 0, 0]), -1, 1)
    run_frames(s, bad, t, 2.0)
    assert s.result[second.id] is None and s.off[second.id]["index"] > 0
    s.skip()
    assert s.result[second.id] == "skipped"
    card = "\n".join(s.scorecard())
    assert "passed" in card and "skipped" in card and "most often off: index" in card


def test_no_hand_never_passes_and_resets_hold():
    s = Session(load_signs())
    for k in range(60):
        _, done = s.update(None, k * 0.1)
        assert not done
    t, _ = run_frames(s, s.sign.target, 10.0, 1.0)
    assert 0 < s.hold.update(True, t)[0] < 1
    s.update(None, t + 1.0)  # lost the hand longer than the flicker allowance
    assert s.hold.update(False, t + 1.1)[0] == 0.0


def test_finish_back_repeat_restart():
    signs = load_signs()
    s = Session(signs)
    for _ in signs:
        s.skip()
    assert s.finished and all(r == "skipped" for r in s.result.values())
    s.back()
    assert not s.finished and s.i == len(signs) - 1
    s.repeat()
    assert s.i == len(signs) - 1
    s.restart()
    assert s.i == 0 and all(r is None for r in s.result.values())
    s.back()
    assert s.i == 0
