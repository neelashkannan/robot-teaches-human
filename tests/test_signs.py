import json

import numpy as np
import pytest

from hand.compare import compare
from hand.signs import REPO, Session, load_signs


def entry(id, status, target=(0.0,) * 8, movement=None):
    return {"id": id, "gloss": id.upper(), "source_url": "https://www.british-sign.co.uk/x",
            "status": status, "instruction": "do it", "target": list(target),
            "hold_seconds": 3.0, "movement": movement, "notes": ""}


def test_real_file_loads_only_shown_statuses():
    raw = json.loads((REPO / "data" / "signs.json").read_text())
    shown = [e["id"] for e in raw if e["status"] in ("exact", "approximated")]
    hidden = [e["id"] for e in raw if e["status"] not in ("exact", "approximated")]
    signs = load_signs()
    ids = [s.id for s in signs]
    assert shown and ids == shown
    assert not set(ids) & set(hidden)
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


def test_movement_sign_is_labelled_with_its_movement_check():
    by_id = {s.id: s for s in load_signs()}
    assert "wrist circle" in by_id["sorry"].status_label
    assert "wrist circle" not in by_id["one"].status_label
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


def test_force_pass_clicks_through_every_sign():
    from hand.signs import REPO, Session, load_signs
    s = Session(load_signs())
    n = len(s.signs)
    for _ in range(n):
        s.force_pass()
    assert s.finished
    assert all(r == "passed" for r in s.result.values())


def make_set(tmp_path, n):
    p = tmp_path / "s.json"
    p.write_text(json.dumps([entry(f"s{k}", "exact") for k in range(n)]))
    return load_signs(p)


@pytest.mark.parametrize("n", [1, 2, 6, 12])
def test_browse_next_prev_jump_wrap(tmp_path, n):
    s = Session(make_set(tmp_path, n), browse=True)
    seen = [s.i]
    for _ in range(n):
        s.next()
        seen.append(s.i)
    assert seen == [k % n for k in range(n + 1)]  # visits every sign, wraps back to the first
    s.prev()
    assert s.i == n - 1  # previous from the first wraps to the last
    for pos in range(1, n + 1):
        assert s.jump(pos) and s.i == pos - 1
    assert not s.jump(0) and not s.jump(n + 1) and s.i == n - 1  # bad position: no move
    assert not s.finished


def test_browse_force_pass_wraps_and_never_finishes(tmp_path):
    s = Session(make_set(tmp_path, 3), browse=True)
    for _ in range(7):
        s.force_pass()
    assert not s.finished and s.i == 7 % 3
    assert all(r == "passed" for r in s.result.values())


def test_browse_pass_is_recorded_but_does_not_move(tmp_path):
    signs = load_signs()
    s = Session(signs, browse=True)
    first = s.sign
    t, done = run_frames(s, first.target, 0.0, 4.0)
    assert done and s.result[first.id] == "passed" and s.sign is first  # stays put
    t, done = run_frames(s, first.target, t, 4.0)
    assert not done  # holding again does not re-announce a pass


def test_toggle_mode_and_practice_still_advances(tmp_path):
    signs = load_signs()
    s = Session(signs, browse=True)
    s.next()
    s.toggle_mode()
    assert not s.browse and s.i == 1
    cur = s.sign
    run_frames(s, cur.target, 0.0, 4.0)
    assert s.result[cur.id] == "passed" and s.sign is not cur  # practice advances
    s.toggle_mode()
    assert s.browse


def ok_result():
    return compare(np.zeros(8), np.zeros(8))


def test_real_statement_loads_me_then_sorry():
    from hand.signs import load_statement

    st = load_statement()
    assert st.english == "I'm sorry."
    assert [s.id for s in st.signs] == ["me", "sorry"]
    assert all(s.status in ("exact", "approximated") and s.source_url.startswith("https://") for s in st.signs)


def test_statement_with_unknown_sign_raises(tmp_path):
    from hand.signs import load_statement

    p = tmp_path / "st.json"
    p.write_text(json.dumps({"statements": [{"id": "x", "english": "X.", "gloss": "X", "signs": ["nope"]}],
                             "signs": []}))
    with pytest.raises(ValueError):
        load_statement("x", path=p)


def test_statement_lesson_learn_then_watch_then_sign_it_all():
    from hand.signs import DEMO_STEP_S, TOGETHER_HOLD_S, StatementLesson, statement_from_text

    # ME GOOD: neither sign has a movement check, so a steady matching hand passes on the hold alone.
    lesson = StatementLesson(statement_from_text("I'm good"))
    t = 0.0

    def hold_until_passed(seconds):
        nonlocal t
        start_id = lesson.sign.id
        while lesson.sign.id == start_id and not lesson.finished and t < 1000:
            lesson.update(ok_result(), t)
            t += 0.1
        assert t - seconds >= 0

    # Round 1: each sign on its own, normal hold.
    assert lesson.heading.startswith('"I\'m good"  Learn sign 1 of 2: ME')
    hold_until_passed(3.0)
    assert lesson.sign.id == "good" and lesson.demo_sign(t) is None
    hold_until_passed(3.0)

    # Round 2 opens with the robot signing the whole statement; marking is paused meanwhile.
    lesson.update(ok_result(), t)
    assert lesson.demo_sign(t).id == "me"
    assert lesson.demo_sign(t + DEMO_STEP_S).id == "good"
    assert lesson.update(ok_result(), t + 0.5) == (0.0, False)
    t += 2 * DEMO_STEP_S + 0.1
    assert lesson.demo_sign(t) is None
    assert "Now sign it all: [ME] GOOD" in lesson.heading
    assert lesson.sign.hold_seconds == TOGETHER_HOLD_S
    hold_until_passed(TOGETHER_HOLD_S)
    assert "ME [GOOD]" in lesson.heading
    hold_until_passed(TOGETHER_HOLD_S)

    assert lesson.finished
    card = lesson.scorecard()
    assert "You signed the whole statement." in card
    assert any(line.startswith("ME (I)") and "passed" in line for line in card)

    lesson.restart()
    assert not lesson.finished and lesson.sign.id == "me" and lesson.sign.hold_seconds == 3.0


def test_statement_lesson_skip_and_back_cross_rounds():
    from hand.signs import StatementLesson, load_statement

    lesson = StatementLesson(load_statement())
    lesson.skip()
    lesson.skip()  # both learn signs skipped -> on to signing it all
    assert lesson.round is lesson.together and lesson.sign.id == "me"
    lesson.back()  # back from the first sign of round 2 -> last sign of round 1
    assert lesson.round is lesson.learn and lesson.sign.id == "sorry"
    assert "Whole statement: not finished." in lesson.scorecard()


@pytest.mark.parametrize("text, gloss, unknown", [
    ("I'm good, thank you!", "ME GOOD THANK-YOU", ()),
    ("Hello, I want two please", "HELLO ME WANT TWO THANK-YOU", ()),
    ("Sorry, my mistake", "SORRY MY", ("mistake",)),
    ("You are great", "YOU GOOD", ()),
    ("I AM 7", "ME SEVEN", ()),
    ("Thanks a lot, goodbye", "THANK-YOU HELLO", ("lot",)),
])
def test_typed_statement_uses_phrasebook_signs_in_order(text, gloss, unknown):
    from hand.signs import statement_from_text

    st = statement_from_text(text)
    assert st.gloss == gloss
    assert st.unknown == unknown
    assert st.english == text


def test_typed_statement_with_no_known_word_is_none():
    from hand.signs import statement_from_text

    assert statement_from_text("What time is it?") is None
    assert statement_from_text("   ") is None


def test_typed_statement_is_capped_and_teachable():
    from hand.signs import MAX_TYPED_SIGNS, StatementLesson, statement_from_text

    st = statement_from_text("one two three five seven one two three five seven")
    assert len(st.signs) == MAX_TYPED_SIGNS
    lesson = StatementLesson(st)
    assert lesson.sign.id == "one" and not lesson.finished


def test_greetings_and_skipped_words_stay_visible():
    from hand.signs import StatementLesson, statement_from_text

    assert statement_from_text("good morning").gloss == "GOOD MORNING"
    assert statement_from_text("Good afternoon, my name is").gloss == "GOOD AFTERNOON MY NAME"
    st = statement_from_text("Good night, welcome")
    assert st.gloss == "GOOD"
    assert st.unknown == ("night (needs two hands)", "welcome (needs two hands)")
    lesson = StatementLesson(st)
    assert "night (needs two hands)" in lesson.note
    lesson.skip()  # into round two: the note stays
    assert "welcome" in lesson.note
    assert "No sign yet for: night (needs two hands)" in lesson.scorecard()[1]
