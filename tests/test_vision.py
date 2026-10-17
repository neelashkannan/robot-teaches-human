import hashlib

import numpy as np
import pytest

from hand.vision import HandTracker, ModelIntegrityError, verify_model


def test_hash_check_accepts_matching_file(tmp_path):
    f = tmp_path / "m.task"
    f.write_bytes(b"model bytes")
    h = tmp_path / "m.sha256"
    h.write_text(hashlib.sha256(b"model bytes").hexdigest() + "  m.task\n")
    assert verify_model(f, h) == b"model bytes"


def test_hash_check_refuses_modified_file(tmp_path):
    f = tmp_path / "m.task"
    f.write_bytes(b"model bytes")
    h = tmp_path / "m.sha256"
    h.write_text(hashlib.sha256(b"model bytes").hexdigest() + "  m.task\n")
    f.write_bytes(b"model bytes!")
    with pytest.raises(ModelIntegrityError):
        verify_model(f, h)


def test_tracker_refuses_modified_model(tmp_path):
    f = tmp_path / "m.task"
    f.write_bytes(b"not the real model")
    with pytest.raises(ModelIntegrityError):
        HandTracker(f)  # default pinned hash does not match


def test_blank_frame_gives_no_hand():
    t = HandTracker()
    try:
        blank = np.zeros((480, 640, 3), np.uint8)
        assert t.process(blank, 0) == (None, None)
        assert t.world is None
        assert t.process(blank, 0) == (None, None)  # repeated timestamp must not crash
    finally:
        t.close()
