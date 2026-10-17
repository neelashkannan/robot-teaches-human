import json

import pytest

from coach_fakes import (
    FakeOpenAI,
    fake_client,
    make_settings,
    rep,
    stances,
    status_error,
    timeout_error,
)
from rth.coach.sensei import (
    NebiusSensei,
    RuleSensei,
    clean_say,
    extract_json,
    is_one_sided,
    make_sensei,
)
from rth.contracts import Limb, RepResult


def _good(**over):
    d = {"next_stance_id": "bow", "mirrored": False, "transition_s": 2.5, "hold_s": 3.0,
         "say": "Lovely work, now sink into a bow stance."}
    d.update(over)
    return json.dumps(d)


# --------------------------------------------------------------------------- rules


def test_rule_first_decision_is_start_stance(tmp_path):
    d = RuleSensei(make_settings(tmp_path), stances()).decide([], reps_left=5)
    assert d.next_stance_id == "ready" and not d.finished and d.source == "offline"


def test_rule_miss_repeats_slower_shorter_and_names_limb(tmp_path):
    s = make_settings(tmp_path)
    r = rep(0, "horse", False, err=30, worst=Limb.LEFT_THIGH, hold_s=3.0)
    d = RuleSensei(s, stances()).decide([r], reps_left=3)
    assert d.next_stance_id == "horse"
    assert d.transition_s > s.session.default_transition_s
    assert d.hold_s < 3.0
    assert "left thigh" in d.say


def test_rule_two_misses_gets_easier(tmp_path):
    lib = stances()
    hist = [rep(0, "crane", False, worst=Limb.RIGHT_SHIN), rep(1, "crane", False)]
    d = RuleSensei(make_settings(tmp_path), lib).decide(hist, reps_left=3)
    assert d.next_stance_id != "crane"
    assert lib.get(d.next_stance_id).difficulty <= lib.get("crane").difficulty


def test_rule_held_progresses_to_new_stance(tmp_path):
    lib = stances()
    hist = [rep(0, "ready", True, err=8)]
    d = RuleSensei(make_settings(tmp_path), lib).decide(hist, reps_left=3)
    assert d.next_stance_id != "ready" and not d.mirrored
    assert lib.get(d.next_stance_id).difficulty == 1  # easy before hard


def test_rule_session_ends_with_finished(tmp_path):
    s = make_settings(tmp_path)
    sensei = RuleSensei(s, stances())
    history: list[RepResult] = []
    total = 6
    d = sensei.decide(history, reps_left=total)
    for i in range(total):
        held = i % 3 != 1
        history.append(rep(i, d.next_stance_id, held, err=8 if held else 30,
                           worst=None if held else Limb.RIGHT_FOREARM, mirrored=d.mirrored))
        d = sensei.decide(history, reps_left=total - i - 1)
        assert d.next_stance_id in stances()
        assert 1.0 <= d.transition_s <= 5.0 and 1.5 <= d.hold_s <= 6.0
        assert d.say and len(d.say.split()) <= 20
        assert d.finished == (i == total - 1)
    assert d.finished and "of 6" in d.say


def test_rule_phrasing_varies(tmp_path):
    sensei = RuleSensei(make_settings(tmp_path), stances())
    says = {sensei.decide([rep(i, "horse", False, worst=Limb.LEFT_SHIN)], reps_left=3).say
            for i in range(3)}
    assert len(says) >= 2


def test_rule_mirrors_sometimes(tmp_path):
    lib = stances()
    mirrored = 0
    for seed in range(20):
        d = RuleSensei(make_settings(tmp_path), lib, seed=seed).decide(
            [rep(0, "bow", True, err=5)], reps_left=3)
        mirrored += d.mirrored and d.next_stance_id == "bow"
    assert 0 < mirrored < 20


def test_one_sided_detection():
    lib = stances()
    assert not is_one_sided(lib.get("horse")) and not is_one_sided(lib.get("ready"))
    assert all(is_one_sided(lib.get(s)) for s in ["bow", "guard", "crane", "white_crane", "archer"])


# --------------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "text",
    [
        _good(),
        "```json\n" + _good() + "\n```",
        "Sure! Here is my decision:\n" + _good() + "\nGood luck!",
        "<think>the visitor did well</think>" + _good(),
    ],
)
def test_extract_json_variants(text):
    assert extract_json(text)["next_stance_id"] == "bow"


def test_extract_json_garbage():
    with pytest.raises(ValueError):
        extract_json("I think the bow stance would be nice.")


def test_clean_say_caps_words():
    long = " ".join(["great"] * 40) + ". And more."
    assert len(clean_say(long).split()) <= 20
    assert clean_say('"Keep your back straight." Then relax.') == "Keep your back straight."
    # A short cheer keeps the cue that follows it.
    assert clean_say("Great hold! Now the crane: knee high. Breathe.") == (
        "Great hold! Now the crane: knee high."
    )


# --------------------------------------------------------------------------- Nebius


def _sensei(tmp_path, replies, **nebius):
    s = make_settings(tmp_path, **nebius)
    fake = FakeOpenAI(chat=replies)
    return NebiusSensei(fake_client(s, fake), s, stances()), fake


HIST = [rep(0, "horse", True, err=9)]


def test_nebius_success_uses_json_schema(tmp_path):
    sensei, fake = _sensei(tmp_path, [_good()])
    d = sensei.decide(HIST, reps_left=3)
    assert d.source == "nebius" and d.next_stance_id == "bow" and d.transition_s == 2.5
    call = fake.chat_calls[0]
    assert call["response_format"]["type"] == "json_schema"
    assert call["max_tokens"] <= 200
    assert "horse" in call["messages"][1]["content"]
    assert all(sid in call["messages"][0]["content"] for sid in stances().ids())


def test_nebius_clamps_numbers(tmp_path):
    sensei, _ = _sensei(tmp_path, [_good(transition_s=0.1, hold_s=60)])
    d = sensei.decide(HIST, reps_left=3)
    assert d.source == "nebius" and d.transition_s == 1.0 and d.hold_s == 6.0


def test_nebius_mirrored_ignored_for_symmetric_stance(tmp_path):
    sensei, _ = _sensei(tmp_path, [_good(next_stance_id="horse", mirrored=True)])
    assert sensei.decide(HIST, reps_left=3).mirrored is False


@pytest.mark.parametrize(
    "reply",
    [
        _good(next_stance_id="flying_dragon"),
        "not json at all",
        '{"next_stance_id": "bow", "say": ',
        json.dumps({"next_stance_id": "bow"}),  # missing say
        _good(transition_s="fast"),
        timeout_error(),
    ],
)
def test_nebius_bad_reply_falls_back_to_rules(tmp_path, reply):
    sensei, _ = _sensei(tmp_path, [reply])
    d = sensei.decide(HIST, reps_left=3)
    expected = RuleSensei(sensei.settings, stances()).decide(HIST, reps_left=3)
    assert d.source == "offline"
    assert d.next_stance_id == expected.next_stance_id


def test_nebius_long_or_medical_say_is_cleaned(tmp_path):
    long_say = "You " + "really " * 30 + "did well."
    sensei, _ = _sensei(tmp_path, [_good(say=long_say), _good(say="This will heal your back.")])
    assert len(sensei.decide(HIST, reps_left=3).say.split()) <= 20
    d = sensei.decide(HIST, reps_left=3)
    assert d.source == "nebius" and "heal" not in d.say


def test_nebius_finished_forced_by_reps_left(tmp_path):
    sensei, _ = _sensei(tmp_path, [_good(say="What a session, well done!")])
    d = sensei.decide(HIST, reps_left=0)
    assert d.finished and d.say.startswith("What a session")


def test_nebius_response_format_refused_downgrades(tmp_path):
    sensei, fake = _sensei(tmp_path, [status_error(400, "response_format not supported"),
                                      "Here you go: " + _good(), _good()])
    assert sensei.decide(HIST, reps_left=3).source == "nebius"
    assert "response_format" not in fake.chat_calls[1]
    sensei.decide(HIST, reps_left=3)
    assert "response_format" not in fake.chat_calls[2]


@pytest.mark.parametrize("mode", ["json_object", "none"])
def test_nebius_other_output_modes(tmp_path, mode):
    sensei, fake = _sensei(tmp_path, [_good()], structured_output=mode)
    assert sensei.decide(HIST, reps_left=3).source == "nebius"
    rf = fake.chat_calls[0].get("response_format")
    assert (rf is None) if mode == "none" else rf == {"type": "json_object"}


def test_make_sensei_offline_without_key(tmp_path):
    assert isinstance(make_sensei(make_settings(tmp_path), stances()), RuleSensei)


def test_nebius_finished_accepts_any_stance_id(tmp_path):
    sensei, _ = _sensei(tmp_path, [_good(next_stance_id="end", say="Bow out, great work!")])
    d = sensei.decide(HIST, reps_left=0)
    assert d.finished and d.source == "nebius" and d.next_stance_id in stances()


def test_nebius_prompt_carries_rule_plan(tmp_path):
    sensei, fake = _sensei(tmp_path, [_good()])
    sensei.decide(HIST, reps_left=3)
    assert "Default plan" in fake.chat_calls[0]["messages"][1]["content"]
