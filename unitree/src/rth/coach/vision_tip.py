"""Optional vision tip: one webcam snapshot + the scores -> one tip the numbers can't catch.

Limb angles miss things like open hands instead of fists or facing the wrong way.
Off by default (settings.nebius.enable_vision_tip); every failure returns None.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Iterable

import cv2
import numpy as np

from rth.coach.client import NebiusClient
from rth.coach.sensei import clean_say, is_safe_say, limb_words, mean_error
from rth.config import Settings
from rth.contracts import RepResult, VisionCoach, VisionTip
from rth.stances import Stance, StanceLibrary, load_stances

MAX_TIP_WORDS = 15


def encode_jpeg_data_url(frame_bgr: np.ndarray, max_px: int = 512, quality: int = 80) -> str:
    """Downscale so the longest side is at most ``max_px`` and return a base64 JPEG data URL."""
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_px / float(max(h, w)))
    img = frame_bgr
    if scale < 1.0:
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        img = cv2.resize(frame_bgr, size, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def clean_tip(text: str, max_words: int = MAX_TIP_WORDS) -> str | None:
    """One sentence of at most ``max_words`` words, or None if nothing usable."""
    t = re.sub(r"<think>.*?(</think>|$)", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    t = clean_say(t, max_words)
    if not t:
        return None
    return t if is_safe_say(t) else None


class NebiusVisionCoach:
    """Vision-language model on Nebius; ``tip`` returns None on any failure."""

    def __init__(
        self,
        client: NebiusClient,
        settings: Settings,
        stances: StanceLibrary | Iterable[Stance] | None = None,
    ) -> None:
        self.client = client
        self.settings = settings
        if stances is None:
            try:
                stances = load_stances(settings.stances_path)
            except Exception:  # noqa: BLE001 - prompt just lacks names
                stances = []
        items = stances.stances if isinstance(stances, StanceLibrary) else list(stances)
        self._by_id = {s.id: s for s in items}

    def prompt(self, rep: RepResult) -> str:
        s = self._by_id.get(rep.stance_id)
        what = f"{s.name}: {s.description}" if s else rep.stance_id
        err = mean_error(rep)
        worst = limb_words(rep.worst_limb)
        scores = (
            f"Limb scoring already found: mean angle error {err:.0f} degrees"
            + (f", worst limb {worst}." if worst else ".")
            if err is not None else "Limb scoring saw too little of the body."
        )
        return (
            f"A visitor is copying this martial arts stance. {what}\n{scores}\n"
            "Look at the photo for one thing the limb angles cannot show: open hands instead "
            "of fists, facing the wrong way, looking down, hunched back, feet placement. "
            "Name one concrete change, or if it all looks right, one specific thing they do "
            "well; never just generic praise. Do not say left or right. Reply with one "
            f"encouraging sentence of at most {MAX_TIP_WORDS} words, no health advice."
        )

    def tip_remote(self, frame_bgr: np.ndarray, rep: RepResult) -> VisionTip:
        """Ask the model; raises on any failure (``check-nebius`` wants to see why)."""
        n = self.settings.nebius
        url = encode_jpeg_data_url(frame_bgr, n.vision_max_px)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": self.prompt(rep)},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }]
        reply = self.client.chat("vision", messages, model=n.vision_model,
                                 max_tokens=n.vision_max_tokens, temperature=0.3)
        tip = clean_tip(reply.text)
        if tip is None:
            raise ValueError(f"unusable tip: {reply.text[:80]!r}")
        return VisionTip(tip, source="nebius")

    def tip(self, frame_bgr: np.ndarray, rep: RepResult) -> VisionTip | None:
        try:
            return self.tip_remote(frame_bgr, rep)
        except Exception:  # noqa: BLE001 - any failure means offline
            return None


def make_vision_coach(
    settings: Settings,
    stances: StanceLibrary | Iterable[Stance] | None = None,
) -> VisionCoach | None:
    """A vision coach when enabled and a key is present, else None."""
    if not settings.nebius.enable_vision_tip:
        return None
    client = NebiusClient(settings.nebius)
    return NebiusVisionCoach(client, settings, stances) if client.available else None
