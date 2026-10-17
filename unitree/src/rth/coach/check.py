"""`rth check-nebius`: is the key there, do the model IDs exist, does each AI role work?

The first thing to run when the key arrives. Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import difflib
from collections.abc import Callable

import numpy as np

from rth.coach.client import NebiusClient
from rth.coach.picker import EmbeddingMovePicker, stance_text
from rth.coach.sensei import NebiusSensei
from rth.coach.vision_tip import NebiusVisionCoach
from rth.config import Settings, load_settings
from rth.contracts import Limb, RepResult
from rth.stances import StanceLibrary, load_stances

Printer = Callable[[str], None]


def closest_ids(model_id: str, available: list[str], n: int = 3) -> list[str]:
    """Likely intended IDs: fuzzy matches on the full ID and on the name after the org."""
    name = model_id.split("/")[-1].lower()
    by_name = {m.split("/")[-1].lower(): m for m in available}
    hits = [by_name[k] for k in difflib.get_close_matches(name, list(by_name), n=n, cutoff=0.5)]
    hits += difflib.get_close_matches(model_id, available, n=n, cutoff=0.5)
    stem = name.split("-")[0]
    hits += [m for m in available if stem and stem in m.lower()]
    return list(dict.fromkeys(hits))[:n]


def sample_rep() -> RepResult:
    """A plausible missed rep, used to smoke-test the sensei."""
    errs = {limb: 12.0 for limb in Limb}
    errs[Limb.LEFT_THIGH] = 38.0
    return RepResult(rep_index=0, stance_id="horse", mirrored=False, held=False, hold_s=3.0,
                     attempt_s=8.5, limb_errors_deg=errs, worst_limb=Limb.LEFT_THIGH)


def test_image() -> np.ndarray:
    """A simple stick figure on grey, so the vision call has something to look at."""
    import cv2

    img = np.full((720, 1280, 3), 200, np.uint8)
    c = (40, 40, 40)
    cv2.circle(img, (640, 160), 50, c, -1)
    cv2.line(img, (640, 210), (640, 420), c, 14)
    cv2.line(img, (640, 260), (500, 200), c, 12)
    cv2.line(img, (640, 260), (780, 200), c, 12)
    cv2.line(img, (640, 420), (520, 560), c, 12)
    cv2.line(img, (640, 420), (760, 560), c, 12)
    return img


def _usage(client: NebiusClient) -> str:
    rec = client.last_call
    if rec is None:
        return ""
    tok = f", {rec.prompt_tokens or 0}+{rec.completion_tokens or 0} tokens"
    slow = "  (SLOWER THAN TIMEOUT BUDGET)" if rec.latency_ms > client.settings.timeout_s * 1000 \
        else ""
    return f"{rec.latency_ms} ms{tok}{slow}"


def run(
    settings: Settings,
    stances: StanceLibrary,
    client: NebiusClient,
    *,
    list_all: bool = False,
    vision: bool = False,
    out: Printer = print,
) -> int:
    """All checks against an existing client; returns the exit code."""
    n = settings.nebius
    failures = 0

    out(f"Nebius Token Factory at {n.base_url}")
    try:
        models = client.list_models()
    except Exception as e:  # noqa: BLE001 - report every failure
        out(f"  FAIL  listing models: {e}")
        out("        Is the key valid? Check NEBIUS_API_KEY in .env.")
        return 1
    out(f"  ok    key works, {len(models)} models available ({_usage(client)})")
    if list_all:
        for m in models:
            out(f"        {m}")

    use_vision = vision or n.enable_vision_tip
    roles = [
        ("embedding", n.embedding_model, n.enable_picker),
        ("sensei", n.sensei_model, n.enable_sensei),
        ("vision", n.vision_model, use_vision),
    ]
    out("Model IDs")
    for role, model_id, needed in roles:
        if model_id in models:
            out(f"  ok    {role:<9} {model_id}")
            continue
        hint = ", ".join(closest_ids(model_id, models)) or "none close; run with --list"
        tag = "FAIL" if needed else "warn"
        failures += needed
        out(f"  {tag}  {role:<9} {model_id} not found. Closest: {hint}")

    out("Smoke tests (one call each)")
    if n.enable_picker:
        try:
            a, b = stance_text(stances.stances[0]), "something for balance"
            va, vb = client.embed("check", [a, b], model=n.embedding_model)
            cos = float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
            out(f"  ok    embedding  dim {len(va)}, cosine {cos:.3f}  [{_usage(client)}]")
            picker = EmbeddingMovePicker(client, settings, stances)
            pick = picker.pick_remote("something for balance")
            alts = ", ".join(f"{sid} {sim:.2f}" for sid, sim in pick.alternatives)
            out(f"  ok    picker     'something for balance' -> {pick.stance_id} "
                f"{pick.similarity:.2f} (then {alts})  [{_usage(client)}]")
        except Exception as e:  # noqa: BLE001 - report every failure
            failures += 1
            out(f"  FAIL  embedding  {e}")

    if n.enable_sensei:
        sensei = NebiusSensei(client, settings, stances)
        try:
            d = sensei.decide_remote([sample_rep()], reps_left=3)
            mode = sensei._format
            out(f"  ok    sensei     -> {d.next_stance_id}{' (mirrored)' if d.mirrored else ''}, "
                f"move {d.transition_s}s, hold {d.hold_s}s, output mode {mode}  "
                f"[{_usage(client)}]")
            out(f'        says: "{d.say}"')
        except Exception as e:  # noqa: BLE001 - report every failure
            failures += 1
            out(f"  FAIL  sensei     {e}")

    if use_vision:
        coach = NebiusVisionCoach(client, settings, stances)
        try:
            tip = coach.tip_remote(test_image(), sample_rep())
            out(f'  ok    vision     "{tip.tip}"  [{_usage(client)}]')
        except Exception as e:  # noqa: BLE001 - report every failure
            failures += 1
            out(f"  FAIL  vision     {e}")
    else:
        out("  skip  vision     (off; add --vision to test it)")

    out(f"Usage log: {n.usage_log}")
    out("All good." if not failures else f"{failures} problem(s). Fix the settings in "
        "config/settings.yaml (nebius section) and run again.")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rth check-nebius", description=__doc__)
    parser.add_argument("--list", action="store_true", help="print every available model ID")
    parser.add_argument("--vision", action="store_true", help="also test the vision tip")
    args = parser.parse_args(argv or [])

    settings = load_settings()  # also loads .env
    n = settings.nebius
    client = NebiusClient(n)
    if not client.available:
        print(f"No Nebius API key: {n.api_key_env} is not set.")
        print("  1. cp .env.example .env")
        print(f"  2. put your key after {n.api_key_env}= in .env (it is git-ignored)")
        print("  3. uv run rth check-nebius")
        print("Until then the demo runs fully offline (keyword picker, rule-based sensei).")
        return 2
    return run(settings, load_stances(settings.stances_path), client,
               list_all=args.list, vision=args.vision)
