import base64
from unittest import mock

import cv2
import numpy as np

from coach_fakes import FakeOpenAI, fake_client, make_settings, rep, stances, timeout_error
from rth.coach import voice as voice_mod
from rth.coach.vision_tip import (
    NebiusVisionCoach,
    clean_tip,
    encode_jpeg_data_url,
    make_vision_coach,
)
from rth.config import VoiceSettings
from rth.contracts import Limb


def test_encode_downscales_to_512():
    url = encode_jpeg_data_url(np.zeros((720, 1280, 3), np.uint8), 512)
    assert url.startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) == 512


def test_clean_tip():
    assert clean_tip('"Make tight fists with both hands." Also smile.') == (
        "Make tight fists with both hands."
    )
    assert len(clean_tip(" ".join(["word"] * 30)).split()) <= 15
    assert clean_tip("") is None
    assert clean_tip("This helps your back pain.") is None


def test_vision_tip_success_sends_image(tmp_path):
    s = make_settings(tmp_path, enable_vision_tip=True)
    fake = FakeOpenAI(chat=["Turn to face the robot squarely."])
    coach = NebiusVisionCoach(fake_client(s, fake), s, stances())
    tip = coach.tip(np.zeros((480, 640, 3), np.uint8), rep(0, "guard", False, worst=Limb.LEFT_FOREARM))
    assert tip is not None and tip.tip == "Turn to face the robot squarely."
    content = fake.chat_calls[0]["messages"][0]["content"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "Fighting guard" in content[0]["text"]


def test_vision_tip_failure_returns_none(tmp_path):
    s = make_settings(tmp_path, enable_vision_tip=True)
    for reply in [timeout_error(), "", "   "]:
        coach = NebiusVisionCoach(fake_client(s, FakeOpenAI(chat=[reply])), s, stances())
        assert coach.tip(np.zeros((100, 100, 3), np.uint8), rep(0, "guard", True)) is None


def test_make_vision_coach_off_or_no_key(tmp_path):
    assert make_vision_coach(make_settings(tmp_path)) is None  # disabled by default
    assert make_vision_coach(make_settings(tmp_path, enable_vision_tip=True)) is None  # no key


# --------------------------------------------------------------------------- voice


def test_mac_voice_new_sentence_stops_previous():
    procs = [mock.MagicMock(), mock.MagicMock()]
    for p in procs:
        p.poll.return_value = None  # still speaking
    with mock.patch.object(voice_mod.subprocess, "Popen", side_effect=procs) as popen:
        v = voice_mod.MacVoice(VoiceSettings(voice="Daniel", rate=200))
        v.say("Bend your knees.")
        v.say("-Now hold it.")
    assert popen.call_count == 2
    procs[0].terminate.assert_called_once()
    args = popen.call_args_list[1].args[0]
    assert args == ["say", "-r", "200", "-v", "Daniel", "Now hold it."]
    assert v.last_text == "-Now hold it."


def test_mac_voice_survives_missing_binary():
    with mock.patch.object(voice_mod.subprocess, "Popen", side_effect=FileNotFoundError):
        voice_mod.MacVoice(VoiceSettings()).say("hello")  # must not raise


def test_make_voice(tmp_path):
    s = make_settings(tmp_path)
    with mock.patch.object(voice_mod.shutil, "which", return_value=None):
        assert isinstance(voice_mod.make_voice(s), voice_mod.NullVoice)
    with mock.patch.object(voice_mod.shutil, "which", return_value="/usr/bin/say"):
        assert isinstance(voice_mod.make_voice(s), voice_mod.MacVoice)
    s.voice.enabled = False
    assert isinstance(voice_mod.make_voice(s), voice_mod.NullVoice)
