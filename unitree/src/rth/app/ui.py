"""Draws the whole booth screen into one BGR canvas with OpenCV.

Layout (scaled from a 1600x900 design): a top bar with the phase and stance name, the
visitor's mirrored webcam on the left, the robot on the right, and a bottom bar with the
one correction that matters most and the sensei's sentence. Text is rendered with a real
TrueType font through Pillow when available (OpenCV's Hershey fonts are the fallback).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from rth.app.session import Phase, Scorecard, SessionView, accuracy_pct
from rth.contracts import Joint, Limb, PersonPose, Scores

BGR = tuple[int, int, int]

# One calm dark palette and a single warm accent; green/red are reserved for limb status.
BG: BGR = (24, 19, 15)
PANEL: BGR = (36, 30, 25)
CARD: BGR = (46, 39, 33)
LINE: BGR = (66, 58, 51)
TEXT: BGR = (246, 243, 240)
MUTED: BGR = (176, 166, 157)
DIM: BGR = (122, 112, 104)
ACCENT: BGR = (64, 178, 255)
GOOD: BGR = (110, 205, 95)
BAD: BGR = (92, 92, 236)
UNSEEN: BGR = (130, 124, 118)

DESIGN_W, DESIGN_H = 1600, 900

# Weights map to faces inside each font collection: (path, index) candidates in order.
_SYSTEM = Path("/System/Library/Fonts")
_FONT_FACES: dict[str, list[tuple[str, int]]] = {
    "regular": [(str(_SYSTEM / "Avenir Next.ttc"), 7), (str(_SYSTEM / "HelveticaNeue.ttc"), 0)],
    "medium": [(str(_SYSTEM / "Avenir Next.ttc"), 5), (str(_SYSTEM / "HelveticaNeue.ttc"), 10)],
    "bold": [(str(_SYSTEM / "Avenir Next.ttc"), 2), (str(_SYSTEM / "HelveticaNeue.ttc"), 1)],
}


def _dejavu(weight: str) -> list[tuple[str, int]]:
    try:
        import matplotlib

        base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    except Exception:  # noqa: BLE001 - matplotlib is optional here
        return []
    name = "DejaVuSans-Bold.ttf" if weight == "bold" else "DejaVuSans.ttf"
    return [(str(base / name), 0)]


class Rect(NamedTuple):
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2


# --------------------------------------------------------------------------- text


class TextRenderer:
    """Anti-aliased text as cached alpha masks, blended into the canvas in any colour."""

    def __init__(self, use_truetype: bool = True, cache_size: int = 600) -> None:
        self._fonts: dict[tuple[int, str], object] = {}
        self._cache: OrderedDict[tuple, tuple[np.ndarray, int]] = OrderedDict()
        self._cache_size = cache_size
        self._pil = None
        if use_truetype:
            try:
                from PIL import Image, ImageDraw, ImageFont

                self._pil = (Image, ImageDraw, ImageFont)
            except ImportError:
                self._pil = None

    def _font(self, size: int, weight: str):
        key = (size, weight)
        if key not in self._fonts:
            font = None
            if self._pil is not None:
                for path, index in _FONT_FACES.get(weight, []) + _dejavu(weight):
                    try:
                        font = self._pil[2].truetype(path, size, index=index)
                        break
                    except OSError:
                        continue
            self._fonts[key] = font
        return self._fonts[key]

    def _hershey(self, size: int, weight: str) -> tuple[float, int]:
        return size / 30.0, 2 if weight == "bold" else 1

    def line_height(self, size: int, weight: str = "regular") -> int:
        font = self._font(size, weight)
        if font is None:
            return int(size * 1.25)
        ascent, descent = font.getmetrics()
        return ascent + descent

    def cap_center(self, size: int, weight: str = "regular") -> int:
        """Distance from the line top to the middle of capital letters, for centring."""
        font = self._font(size, weight)
        if font is None:
            return int(size * 0.6)
        _, top, _, bottom = font.getbbox("H", anchor="la")
        return (top + bottom) // 2

    def width(self, text: str, size: int, weight: str = "regular", tracking: float = 0) -> int:
        font = self._font(size, weight)
        if font is None:
            scale, thick = self._hershey(size, weight)
            return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]
        if tracking:
            return int(sum(font.getlength(c) + tracking for c in text) - tracking)
        return math.ceil(font.getlength(text))

    def mask(self, text: str, size: int, weight: str = "regular",
             tracking: float = 0) -> np.ndarray:
        key = (text, size, weight, tracking)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit[0]
        h = self.line_height(size, weight)
        w = max(1, self.width(text, size, weight, tracking) + 4)
        font = self._font(size, weight)
        if font is None:
            m = np.zeros((h, w), np.uint8)
            scale, thick = self._hershey(size, weight)
            cv2.putText(m, text, (0, int(size * 0.95)), cv2.FONT_HERSHEY_SIMPLEX, scale, 255,
                        thick, cv2.LINE_AA)
        else:
            image_mod, draw_mod, _ = self._pil
            img = image_mod.new("L", (w, h), 0)
            draw = draw_mod.Draw(img)
            if tracking:
                x = 0.0
                for c in text:
                    draw.text((x, 0), c, font=font, fill=255, anchor="la")
                    x += font.getlength(c) + tracking
            else:
                draw.text((0, 0), text, font=font, fill=255, anchor="la")
            m = np.asarray(img, np.uint8)
        self._cache[key] = (m, w)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return m

    def draw(self, canvas: np.ndarray, text: str, x: int, y: int, size: int, color: BGR,
             weight: str = "regular", anchor: str = "lt", tracking: float = 0,
             alpha: float = 1.0) -> int:
        """Draw one line. ``anchor`` is horizontal l/m/r + vertical t/m (cap centre)/b.

        Returns the drawn width.
        """
        if not text:
            return 0
        m = self.mask(text, size, weight, tracking)
        w = self.width(text, size, weight, tracking)
        if anchor[0] == "m":
            x -= w // 2
        elif anchor[0] == "r":
            x -= w
        if anchor[1] == "m":
            y -= self.cap_center(size, weight)
        elif anchor[1] == "b":
            y -= m.shape[0]
        blend_mask(canvas, m, x, y, color, alpha)
        return w

    def fit(self, text: str, size: int, weight: str, max_w: int) -> str:
        """Ellipsise ``text`` to fit ``max_w`` pixels."""
        if self.width(text, size, weight) <= max_w:
            return text
        words = text.split()
        while words and self.width(" ".join(words) + "…", size, weight) > max_w:
            words.pop()
        return (" ".join(words) + "…") if words else "…"

    def wrap(self, text: str, size: int, weight: str, max_w: int, max_lines: int = 2) -> list[str]:
        lines: list[str] = []
        current = ""
        for word in text.split():
            trial = f"{current} {word}".strip()
            if self.width(trial, size, weight) <= max_w or not current:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        if len(lines) > max_lines:
            lines = lines[: max_lines - 1] + [self.fit(" ".join(lines[max_lines - 1:]), size,
                                                       weight, max_w)]
        return lines


# --------------------------------------------------------------------------- primitives


def blend_mask(canvas: np.ndarray, mask: np.ndarray, x: int, y: int, color: BGR,
               alpha: float = 1.0) -> None:
    """Blend a solid colour through an 8-bit mask at (x, y), clipped to the canvas."""
    h, w = mask.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x0 >= x1 or y0 >= y1:
        return
    m = mask[y0 - y: y1 - y, x0 - x: x1 - x].astype(np.uint16)
    if alpha < 1.0:
        m = (m * round(alpha * 256)) >> 8
    m = m[..., None]
    roi = canvas[y0:y1, x0:x1]
    # Integer maths: about twice as fast as float and indistinguishable on screen.
    col = np.asarray(color, np.uint16)
    roi[:] = ((roi.astype(np.uint16) * (255 - m) + col * m + 127) // 255).astype(np.uint8)


def blend_solid(canvas: np.ndarray, r: Rect, color: BGR, alpha: float) -> None:
    """Blend a solid colour over a whole rectangle (no mask)."""
    x0, y0 = max(0, r.x), max(0, r.y)
    x1, y1 = min(canvas.shape[1], r.x2), min(canvas.shape[0], r.y2)
    if x0 >= x1 or y0 >= y1:
        return
    roi = canvas[y0:y1, x0:x1]
    if alpha >= 1.0:
        roi[:] = color
        return
    solid = np.empty_like(roi)
    solid[:] = color
    cv2.addWeighted(roi, 1.0 - alpha, solid, alpha, 0.0, dst=roi)


_MASKS: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()


def round_mask(w: int, h: int, r: int) -> np.ndarray:
    key = (w, h, r)
    m = _MASKS.get(key)
    if m is None:
        r = max(0, min(r, w // 2, h // 2))
        m = np.zeros((h, w), np.uint8)
        if r == 0:
            m[:] = 255
        else:
            cv2.rectangle(m, (r, 0), (w - r - 1, h - 1), 255, -1)
            cv2.rectangle(m, (0, r), (w - 1, h - r - 1), 255, -1)
            for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
                cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
        _MASKS[key] = m
        if len(_MASKS) > 64:
            _MASKS.popitem(last=False)
    return m


def fill_round(canvas: np.ndarray, r: Rect, radius: int, color: BGR, alpha: float = 1.0) -> None:
    if r.w <= 0 or r.h <= 0:
        return
    rad = max(0, min(radius, r.w // 2, r.h // 2))
    m = round_mask(r.w, r.h, radius)
    # Only the corner bands need the mask; the middle is a plain blend.
    blend_solid(canvas, Rect(r.x, r.y + rad, r.w, r.h - 2 * rad), color, alpha)
    if rad:
        blend_mask(canvas, m[:rad], r.x, r.y, color, alpha)
        blend_mask(canvas, m[r.h - rad:], r.x, r.y2 - rad, color, alpha)


_RINGS: OrderedDict[tuple[int, int, int, int], np.ndarray] = OrderedDict()


def outline_round(canvas: np.ndarray, r: Rect, radius: int, color: BGR, thickness: int) -> None:
    key = (r.w, r.h, radius, thickness)
    ring = _RINGS.get(key)
    if ring is None:
        outer = round_mask(r.w, r.h, radius).astype(np.int16)
        t = thickness
        inner = np.zeros_like(outer)
        if r.w > 2 * t and r.h > 2 * t:
            inner[t:-t, t:-t] = round_mask(r.w - 2 * t, r.h - 2 * t, max(0, radius - t))
        ring = _RINGS[key] = np.clip(outer - inner, 0, 255).astype(np.uint8)
        if len(_RINGS) > 16:
            _RINGS.popitem(last=False)
    band = min(r.h // 2, max(radius, thickness) + 2)
    # Blend only the four edge strips; the inside of a ring is empty.
    blend_mask(canvas, ring[:band], r.x, r.y, color)
    blend_mask(canvas, ring[r.h - band:], r.x, r.y2 - band, color)
    mid = ring[band: r.h - band]
    side = min(r.w // 2, thickness + 2)
    blend_mask(canvas, mid[:, :side], r.x, r.y + band, color)
    blend_mask(canvas, mid[:, r.w - side:], r.x2 - side, r.y + band, color)


def darken(canvas: np.ndarray, r: Rect, amount: float) -> None:
    roi = canvas[max(0, r.y): r.y2, max(0, r.x): r.x2]
    roi[:] = (roi * (1.0 - amount)).astype(np.uint8)


def paste_round(canvas: np.ndarray, img: np.ndarray, r: Rect, radius: int, under: BGR) -> None:
    """Paste ``img`` (already r.w x r.h) with anti-aliased rounded corners over ``under``."""
    canvas[r.y: r.y2, r.x: r.x2] = img
    rad = max(1, min(radius, r.w // 2, r.h // 2))
    m = round_mask(r.w, r.h, radius)
    a = m.astype(np.float32)[..., None] / 255.0
    bg = np.asarray(under, np.float32)
    for ys in (slice(0, rad), slice(r.h - rad, r.h)):
        for xs in (slice(0, rad), slice(r.w - rad, r.w)):
            roi = canvas[r.y: r.y2, r.x: r.x2][ys, xs]
            roi[:] = (roi * a[ys, xs] + bg * (1.0 - a[ys, xs]) + 0.5).astype(np.uint8)


def cover(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Centre-crop and resize to exactly w x h, filling the box."""
    ih, iw = img.shape[:2]
    scale = max(w / iw, h / ih)
    cw, ch = min(iw, round(w / scale)), min(ih, round(h / scale))
    x0, y0 = (iw - cw) // 2, (ih - ch) // 2
    crop = img[y0: y0 + ch, x0: x0 + cw]
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(crop, (w, h), interpolation=interp)


def contain(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Fit inside w x h without cropping, padding with the image's own corner colour."""
    ih, iw = img.shape[:2]
    if (iw, ih) == (w, h):
        return img
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    out = np.empty((h, w, 3), np.uint8)
    out[:] = img[0, 0]
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    out[y0: y0 + nh, x0: x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


# --------------------------------------------------------------------------- skeleton

Skeleton = Callable[..., object]


def fallback_skeleton(img: np.ndarray, pose: PersonPose, scores: Scores | None = None, *,
                      mirrored: bool) -> np.ndarray:
    """Limbs coloured by score, used until ``rth.vision.overlay`` is available."""
    h, w = img.shape[:2]
    thick = max(3, h // 90)

    def px(j: Joint) -> tuple[int, int] | None:
        p = pose.image_points.get(j)
        if p is None or pose.visibility.get(j, 0.0) < 0.5:
            return None
        x = 1.0 - p[0] if mirrored else p[0]
        return int(x * w * 4), int(p[1] * h * 4)

    torso = [(Joint.LEFT_SHOULDER, Joint.RIGHT_SHOULDER), (Joint.LEFT_HIP, Joint.RIGHT_HIP),
             (Joint.LEFT_SHOULDER, Joint.LEFT_HIP), (Joint.RIGHT_SHOULDER, Joint.RIGHT_HIP)]
    for a, b in torso:
        pa, pb = px(a), px(b)
        if pa and pb:
            cv2.line(img, pa, pb, (225, 220, 215), max(2, thick // 2), cv2.LINE_AA, 2)
    for limb in Limb:
        pa, pb = (px(j) for j in limb.joints)
        if not (pa and pb):
            continue
        s = scores.get(limb) if scores else None
        colour = UNSEEN if s is None or s.error_deg is None else (GOOD if s.ok else BAD)
        cv2.line(img, pa, pb, (20, 20, 20), thick + 4, cv2.LINE_AA, 2)
        cv2.line(img, pa, pb, colour, thick, cv2.LINE_AA, 2)
    for j in Joint:
        p = px(j)
        if p:
            cv2.circle(img, p, int(thick * 0.8 * 4), (245, 245, 245), -1, cv2.LINE_AA, 2)
    return img


def load_skeleton() -> tuple[Skeleton, BGR, BGR]:
    """The vision module's overlay and colours if present, else the local fallback."""
    try:
        from rth.vision import overlay

        return overlay.draw_skeleton, tuple(overlay.OK_BGR), tuple(overlay.BAD_BGR)
    except (ImportError, AttributeError):
        return fallback_skeleton, GOOD, BAD


# --------------------------------------------------------------------------- state


@dataclass
class ScreenState:
    """One frame's worth of everything the screen shows."""

    view: SessionView
    camera_bgr: np.ndarray | None = None  # unmirrored, as tracked
    pose: PersonPose | None = None
    scores: Scores | None = None
    robot_bgr: np.ndarray | None = None
    hint: str | None = None
    person_present: bool = False
    full_body: bool = False
    all_ok: bool = False
    mirror_display: bool = True
    mirror_view: bool = True
    sources: Mapping[str, str] = field(default_factory=dict)  # role -> "nebius" / "offline"
    recognised: tuple[str | None, float] | None = None  # None: no recogniser loaded
    fps: float | None = None
    typing: str | None = None  # text being typed, or None when not typing
    notice: str | None = None  # brief operator feedback, e.g. "Mirror view off"
    stance_keys: tuple[str, ...] = ()  # stance ids in number-key order


@dataclass(frozen=True)
class Layout:
    w: int
    h: int
    s: float
    pad: int
    top: Rect
    cam: Rect
    robot: Rect
    bottom: Rect


def make_layout(w: int, h: int) -> Layout:
    s = min(w / DESIGN_W, h / DESIGN_H)
    pad = round(28 * s)
    top_h = round(112 * s)
    bottom_h = round(176 * s)
    gap = round(20 * s)
    mid_h = h - top_h - bottom_h
    pw = (w - 2 * pad - gap) // 2
    return Layout(
        w=w,
        h=h,
        s=s,
        pad=pad,
        top=Rect(pad, 0, w - 2 * pad, top_h),
        cam=Rect(pad, top_h, pw, mid_h),
        robot=Rect(pad + pw + gap, top_h, w - 2 * pad - gap - pw, mid_h),
        bottom=Rect(pad, top_h + mid_h, w - 2 * pad, bottom_h),
    )


# --------------------------------------------------------------------------- renderer


class BoothUI:
    def __init__(self, stance_names: Mapping[str, str], *, tolerance_deg: float = 25.0,
                 skeleton: Skeleton | None = None, use_truetype: bool = True) -> None:
        self.names = dict(stance_names)
        self.tolerance_deg = tolerance_deg
        if skeleton is None:
            skeleton, good, bad = load_skeleton()
        else:
            good, bad = GOOD, BAD
        self.skeleton = skeleton
        self.good, self.bad = good, bad
        self.text = TextRenderer(use_truetype)
        self._s = 1.0
        self._bg: np.ndarray | None = None

    # ----------------------------------------------------------------- helpers

    def S(self, v: float) -> int:
        return round(v * self._s)

    def _t(self, canvas, text, x, y, size, color, weight="regular", anchor="lt", **kw) -> int:
        return self.text.draw(canvas, text, int(x), int(y), self.S(size), color, weight, anchor,
                              **kw)

    def _kicker(self, canvas, text, x, y, color=ACCENT, anchor="lt") -> int:
        return self._t(canvas, text.upper(), x, y, 17, color, "bold", anchor,
                       tracking=self.S(2.2))

    def _pill(self, canvas, text, x, y, size=17, fg=TEXT, bg=CARD, weight="bold",
              anchor="l", tracking=1.5, alpha=0.92) -> Rect:
        """Rounded label; (x, y) is its left/right edge and vertical centre."""
        tw = self.text.width(text, self.S(size), weight, self.S(tracking))
        ph, padx = self.S(size * 1.9), self.S(size * 0.8)
        w = tw + 2 * padx
        x0 = x - w if anchor == "r" else (x - w // 2 if anchor == "m" else x)
        r = Rect(int(x0), int(y - ph // 2), int(w), int(ph))
        fill_round(canvas, r, ph // 2, bg, alpha)
        self._t(canvas, text, r.x + padx, r.cy, size, fg, weight, "lm",
                tracking=self.S(tracking))
        return r

    def name(self, stance_id: str | None) -> str:
        if stance_id is None:
            return ""
        return self.names.get(stance_id, stance_id.replace("_", " ").capitalize())

    def _arrow(self, canvas, x_right: int, cy: int, length: int, color: BGR) -> int:
        """A right-pointing arrow ending at ``x_right``; drawn because not every font has one."""
        t = max(2, self.S(2))
        head = max(4, length // 3)
        x0 = x_right - length
        cv2.line(canvas, (x0, cy), (x_right, cy), color, t, cv2.LINE_AA)
        cv2.line(canvas, (x_right - head, cy - head), (x_right, cy), color, t, cv2.LINE_AA)
        cv2.line(canvas, (x_right - head, cy + head), (x_right, cy), color, t, cv2.LINE_AA)
        return length

    # ----------------------------------------------------------------- render

    def render(self, st: ScreenState, size: tuple[int, int] = (DESIGN_W, DESIGN_H)) -> np.ndarray:
        w, h = size
        lay = make_layout(w, h)
        self._s = lay.s
        if self._bg is None or self._bg.shape[:2] != (h, w):
            self._bg = np.empty((h, w, 3), np.uint8)
            self._bg[:] = BG
        canvas = self._bg.copy()
        if st.view.phase is Phase.SCORECARD and st.view.scorecard is not None:
            self._scorecard(canvas, lay, st)
        else:
            self._top_bar(canvas, lay, st)
            self._camera_panel(canvas, lay.cam, st)
            self._robot_panel(canvas, lay.robot, st)
            self._bottom_bar(canvas, lay.bottom, st)
        if st.notice:
            self._pill(canvas, st.notice, w // 2, lay.top.h - self.S(4), 18, TEXT, LINE, "medium",
                       "m", 0.5)
        if st.typing is not None:
            self._typing(canvas, lay, st)
        return canvas

    # ----------------------------------------------------------------- top bar

    def _top_bar(self, canvas, lay: Layout, st: ScreenState) -> None:
        v, r = st.view, lay.top
        x, y_kick, y_title = r.x + self.S(4), self.S(26), self.S(52)
        ex = v.exercise
        if v.phase is Phase.WAITING:
            kicker, title = "Robot teaches human", "Copy the robot, limb by limb"
        else:
            n = min(v.reps_done + (0 if v.phase is Phase.THINKING else 1), v.reps_total)
            step = {Phase.DEMONSTRATING: "Watch", Phase.COPYING: "Your turn",
                    Phase.THINKING: "Sensei is thinking"}[v.phase]
            if v.phase is Phase.THINKING and v.final_rep:
                step = "Last stance done"
            kicker = f"{step}  ·  stance {max(n, 1)} of {v.reps_total}"
            title = self.name(ex.stance_id if ex else None)
        self._kicker(canvas, kicker, x, y_kick)
        tw = self._t(canvas, title, x, y_title, 44, TEXT, "bold")
        if ex is not None and ex.mirrored and v.phase is not Phase.WAITING:
            th = self.text.line_height(self.S(44), "bold")
            self._pill(canvas, "OTHER SIDE", x + tw + self.S(18), y_title + th // 2, 15, ACCENT,
                       CARD)
        if v.phase is not Phase.WAITING:
            self._rep_dots(canvas, r, v)

    def _rep_dots(self, canvas, r: Rect, v: SessionView) -> None:
        rad, gap = self.S(11), self.S(34)
        n = v.reps_total
        cy = self.S(66)
        x0 = r.x2 - rad - (n - 1) * gap
        self._kicker(canvas, "Progress", r.x2, self.S(26), MUTED, "rt")
        for i in range(n):
            c = (x0 + i * gap) * 4, cy * 4
            if i < len(v.history):
                colour = self.good if v.history[i].held else self.bad
                cv2.circle(canvas, c, rad * 4, colour, -1, cv2.LINE_AA, 2)
            elif i == len(v.history) and v.phase is not Phase.THINKING:
                cv2.circle(canvas, c, rad * 4, ACCENT, max(2, self.S(3)), cv2.LINE_AA, 2)
                cv2.circle(canvas, c, int(rad * 0.4) * 4, ACCENT, -1, cv2.LINE_AA, 2)
            else:
                cv2.circle(canvas, c, rad * 4, LINE, max(2, self.S(2)), cv2.LINE_AA, 2)

    # ----------------------------------------------------------------- camera

    def _camera_panel(self, canvas, r: Rect, st: ScreenState) -> None:
        v = st.view
        radius = self.S(22)
        if st.camera_bgr is None:
            fill_round(canvas, r, radius, PANEL)
            self._t(canvas, "No camera", r.cx, r.cy, 30, MUTED, "medium", "mm")
        else:
            img = cv2.flip(st.camera_bgr, 1) if st.mirror_display else st.camera_bgr.copy()
            if st.pose is not None:
                out = self.skeleton(img, st.pose, st.scores, mirrored=st.mirror_display)
                if isinstance(out, np.ndarray):
                    img = out
            paste_round(canvas, cover(img, r.w, r.h), r, radius, BG)
        copying = v.phase is Phase.COPYING
        meter = copying or (v.phase is Phase.THINKING and v.hold_progress > 0)
        if not meter:
            self._pill(canvas, "YOU", r.x + self.S(18), r.y + self.S(30), 15, TEXT, (20, 16, 13),
                       alpha=0.7)
        if copying and st.all_ok:
            outline_round(canvas, r, radius, self.good, max(3, self.S(5)))

        if not st.person_present:
            if v.phase is Phase.WAITING:
                self._card(canvas, r, "Step onto the mark",
                           "so the camera can see you from head to toe")
            else:
                self._card(canvas, r, "Come back into view", "the session is paused")
        elif not st.full_body:
            self._banner(canvas, r, "Step back so I can see your whole body")
        elif v.phase is Phase.WAITING:
            self._banner(canvas, r, "Hold still, the robot is about to start", ACCENT)

        if meter:
            self._hold_meter(canvas, r, st)

    def _card(self, canvas, r: Rect, title: str, sub: str) -> None:
        cw = min(r.w - self.S(60), self.S(560))
        ch = self.S(150)
        c = Rect(r.cx - cw // 2, r.cy - ch // 2, cw, ch)
        fill_round(canvas, c, self.S(18), (20, 16, 13), 0.82)
        self._t(canvas, title, c.cx, c.y + self.S(56), 36, TEXT, "bold", "mm")
        self._t(canvas, sub, c.cx, c.y + self.S(104), 22, MUTED, "regular", "mm")

    def _banner(self, canvas, r: Rect, text: str, dot: BGR | None = None) -> None:
        size = 26
        dot = ACCENT if dot is None else dot
        tw = self.text.width(text, self.S(size), "bold")
        bh = self.S(64)
        bw = min(r.w - self.S(36), tw + self.S(84))
        # Near the bottom: banners show when the legs are out of view or nobody is copying yet,
        # so this covers the least useful part of the frame.
        b = Rect(r.cx - bw // 2, r.y2 - bh - self.S(28), bw, bh)
        fill_round(canvas, b, bh // 2, (20, 16, 13), 0.86)
        cv2.circle(canvas, ((b.x + self.S(32)) * 4, b.cy * 4), self.S(7) * 4, dot, -1,
                   cv2.LINE_AA, 2)
        text = self.text.fit(text, self.S(size), "bold", bw - self.S(70))
        self._t(canvas, text, b.x + self.S(54), b.cy, size, TEXT, "bold", "lm")

    def _hold_meter(self, canvas, r: Rect, st: ScreenState) -> None:
        v = st.view
        ex = v.exercise
        # Along the top edge: the bottom of the frame is where the visitor's feet are.
        m = Rect(r.x + self.S(14), r.y + self.S(14), r.w - self.S(28), self.S(78))
        fill_round(canvas, m, self.S(16), (20, 16, 13), 0.82)
        hold_s = ex.hold_s if ex else 3.0
        done = v.hold_progress >= 1.0
        held_s = v.hold_progress * hold_s
        label = "Held!" if done else ("Holding" if v.holding else "Hold every limb green")
        if v.robot_recovering and not done:
            label = "Paused"
        colour = self.good if done or st.all_ok else ACCENT
        self._t(canvas, label, m.x + self.S(22), m.y + self.S(28), 22, TEXT, "bold", "lm")
        self._t(canvas, f"{held_s:.1f} / {hold_s:.0f} s", m.x2 - self.S(22), m.y + self.S(28), 22,
                MUTED, "medium", "rm")
        track = Rect(m.x + self.S(22), m.y + self.S(50), m.w - self.S(44), self.S(12))
        fill_round(canvas, track, track.h // 2, LINE)
        fw = int(track.w * min(1.0, v.hold_progress))
        if fw > track.h:
            fill_round(canvas, Rect(track.x, track.y, fw, track.h), track.h // 2, colour)

    # ----------------------------------------------------------------- robot

    def _robot_panel(self, canvas, r: Rect, st: ScreenState) -> None:
        v = st.view
        radius = self.S(22)
        if st.robot_bgr is None:
            fill_round(canvas, r, radius, PANEL)
        else:
            paste_round(canvas, contain(st.robot_bgr, r.w, r.h), r, radius, BG)
        self._pill(canvas, "TEACHER  ·  UNITREE G1", r.x + self.S(18), r.y + self.S(30), 15, TEXT,
                   (20, 16, 13), alpha=0.7)
        if v.exercise is not None and v.exercise.mirrored and v.phase is not Phase.WAITING:
            self._pill(canvas, "OTHER SIDE", r.x2 - self.S(18), r.y + self.S(30), 15, ACCENT,
                       (20, 16, 13), anchor="r", alpha=0.7)
        if v.robot_recovering and v.phase in (Phase.DEMONSTRATING, Phase.COPYING):
            fill_round(canvas, r, self.S(22), (0, 0, 0), 0.45)
            self._t(canvas, "Getting back up", r.cx, r.cy, 40, TEXT, "bold", "mm")
            self._t(canvas, "the rep timer is paused", r.cx, r.cy + self.S(50), 24, MUTED,
                    "medium", "mm")
        elif v.phase is Phase.DEMONSTRATING:
            self._pill(canvas, "WATCH ME", r.cx, r.y2 - self.S(46), 18, (20, 16, 13), ACCENT,
                       anchor="m", tracking=2.5, alpha=1.0)
        elif v.phase is Phase.THINKING:
            self._thinking(canvas, r, v)

    def _thinking(self, canvas, r: Rect, v: SessionView) -> None:
        fill_round(canvas, r, self.S(22), (0, 0, 0), 0.55)
        rep = v.last_rep
        cy = r.cy - self.S(40)
        if rep is not None:
            colour = self.good if rep.held else ACCENT
            rad = self.S(58)
            ring = max(3, self.S(6))
            cv2.circle(canvas, (r.cx * 4, cy * 4), rad * 4, LINE, ring, cv2.LINE_AA, 2)
            frac = 1.0 if rep.held else min(1.0, v.best_hold_s / max(rep.hold_s, 1e-6))
            if frac > 0:
                # Arc from 12 o'clock: how much of the hold they managed at best.
                cv2.ellipse(canvas, (r.cx, cy), (rad, rad), -90, 0, 360 * frac, colour, ring,
                            cv2.LINE_AA)
            if rep.held:
                pts = np.array([(-0.42, 0.02), (-0.12, 0.32), (0.45, -0.3)])
                pts = ((pts * rad + (r.cx, cy)) * 4).astype(np.int32)
                cv2.polylines(canvas, [pts], False, colour, max(4, self.S(8)), cv2.LINE_AA, 2)
            else:
                self._t(canvas, f"{v.best_hold_s:.1f}s", r.cx, cy, 32, TEXT, "bold", "mm")
            word = "Held" if rep.held else "Not quite"
            self._t(canvas, word, r.cx, cy + self.S(98), 40, TEXT, "bold", "mm")
        if v.final_rep:
            sub = "Adding up your scores"
        else:
            sub = "Choosing what comes next"
        dots = "." * (1 + int(v.now * 2.5) % 3)
        sw = self.text.width(sub, self.S(24), "medium")
        x = r.cx - sw // 2
        self._t(canvas, sub, x, cy + self.S(150), 24, MUTED, "medium", "lm")
        self._t(canvas, dots, x + sw + self.S(2), cy + self.S(150), 24, MUTED, "medium", "lm")

    # ----------------------------------------------------------------- bottom bar

    def _primary_line(self, st: ScreenState) -> tuple[str, BGR | None]:
        """The single most useful instruction right now, and its status dot colour."""
        v = st.view
        copy_how = "copy it like a mirror" if st.mirror_view else "copy it with the same side"
        if v.phase is Phase.WAITING:
            if st.person_present and st.full_body:
                return "Great, stay right there", ACCENT
            return "Step onto the mark: the robot shows a stance, you copy it", None
        if v.phase is Phase.DEMONSTRATING:
            if v.robot_recovering:
                return "Sensei is getting back up", ACCENT
            return f"Watch the robot, then {copy_how}", ACCENT
        if v.phase is Phase.COPYING:
            if not st.person_present:
                return "Step into view", ACCENT
            if not st.full_body:
                return "Step back so your legs are in view", ACCENT
            if v.robot_recovering:
                return "Sensei is getting back up", ACCENT
            if st.all_ok:
                return "Perfect. Hold it!", self.good
            if st.hint:
                return st.hint, self.bad
            return "Match the robot, limb by limb", ACCENT
        rep = v.last_rep
        if rep is None:
            return "", None
        if rep.held:
            return f"Held for {rep.hold_s:.0f} seconds. Nice work.", self.good
        worst = rep.worst_limb
        if worst is not None:
            return f"Not held this time. Watch your {worst.label.lower()}", self.bad
        return "Not held this time", self.bad

    def _bottom_bar(self, canvas, r: Rect, st: ScreenState) -> None:
        v = st.view
        text, dot = self._primary_line(st)
        y1 = r.y + self.S(52)
        x = r.x + self.S(4)
        if dot is not None:
            cv2.circle(canvas, ((x + self.S(10)) * 4, y1 * 4), self.S(10) * 4, dot, -1,
                       cv2.LINE_AA, 2)
            x += self.S(36)
        text = self.text.fit(text, self.S(40), "bold", r.x2 - x)
        self._t(canvas, text, x, y1, 40, TEXT, "bold", "lm")

        # Second line: what the sensei said (or is doing).
        y2 = r.y + self.S(108)
        x = r.x + self.S(4)
        label, line = "Sensei", v.say or ""
        if v.pending_request:
            label, line = "Sensei", f"Finding a stance for “{v.pending_request}”…"
        elif v.phase is Phase.THINKING and v.tip:
            label, line = "Tip", v.tip
        elif v.phase is Phase.WAITING:
            label, line = "Try", "press T and type “something for balance”"
        if line:
            lw = self._kicker(canvas, label, x, y2, ACCENT, "lm")
            x += lw + self.S(16)
            line = self.text.fit(line, self.S(28), "medium", r.x2 - x)
            self._t(canvas, line, x, y2, 28, TEXT if label != "Try" else MUTED, "medium", "lm")
        self._footer(canvas, r, st)

    def _footer(self, canvas, r: Rect, st: ScreenState) -> None:
        y = r.y2 - self.S(26)
        n = len(st.stance_keys)
        keys = "T ask  ·  " + (f"1–{n} choose  ·  " if n else "") + "Space skip  ·  M mirror"
        self._t(canvas, keys, r.x + self.S(4), y, 16, DIM, "medium", "lm")
        x = r.x2
        if st.fps is not None:
            x = self._pill(canvas, f"{st.fps:.0f} FPS", x, y, 13, MUTED, PANEL, anchor="r").x
            x -= self.S(10)
        if st.recognised is not None:
            label, conf = st.recognised
            text = f"SEES  {self.name(label).upper()}  {conf * 100:.0f}%" if label else "SEES  —"
            x = self._pill(canvas, text, x, y, 13, MUTED, PANEL, anchor="r").x - self.S(10)
        for role, title in (("vision", "TIP"), ("sensei", "SENSEI"), ("picker", "PICKER")):
            src = st.sources.get(role)
            if src is None:
                continue
            nebius = src == "nebius"
            text = f"{title}  {'NEBIUS' if nebius else 'OFFLINE'}"
            x = self._pill(canvas, text, x, y, 13, ACCENT if nebius else MUTED, PANEL,
                           anchor="r").x - self.S(10)

    # ----------------------------------------------------------------- typing

    def _typing(self, canvas, lay: Layout, st: ScreenState) -> None:
        darken(canvas, Rect(0, 0, lay.w, lay.h), 0.62)
        cw, ch = min(lay.w - 2 * lay.pad, self.S(940)), self.S(330)
        c = Rect(lay.w // 2 - cw // 2, lay.h // 2 - ch // 2, cw, ch)
        fill_round(canvas, c, self.S(24), CARD)
        x = c.x + self.S(48)
        self._kicker(canvas, "Ask the sensei", x, c.y + self.S(40))
        self._t(canvas, "What would you like to practise?", x, c.y + self.S(66), 36, TEXT, "bold")
        box = Rect(x, c.y + self.S(132), c.w - self.S(96), self.S(76))
        fill_round(canvas, box, self.S(14), BG)
        outline_round(canvas, box, self.S(14), ACCENT, max(2, self.S(2)))
        typed = st.typing or ""
        tx = box.x + self.S(24)
        size = 32
        if typed:
            shown = typed
            while self.text.width(shown, self.S(size), "medium") > box.w - self.S(60) and shown:
                shown = shown[1:]  # keep the end visible while typing long requests
            tw = self._t(canvas, shown, tx, box.cy, size, TEXT, "medium", "lm")
        else:
            self._t(canvas, "something for balance", tx, box.cy, size, DIM, "medium", "lm")
            tw = 0
        if int(st.view.now * 2) % 2 == 0:
            cx = tx + tw + self.S(3)
            cv2.line(canvas, (cx, box.cy - self.S(20)), (cx, box.cy + self.S(20)), ACCENT,
                     max(2, self.S(3)), cv2.LINE_AA)
        self._t(canvas, "e.g. a strong stance  ·  gentle tai chi  ·  something for my legs", x,
                c.y + self.S(244), 22, MUTED, "regular", "lm")
        self._t(canvas, "Enter to choose   ·   Esc to cancel", x, c.y + self.S(284), 18, DIM,
                "medium", "lm")

    # ----------------------------------------------------------------- scorecard

    def _scorecard(self, canvas, lay: Layout, st: ScreenState) -> None:
        v = st.view
        sc: Scorecard = v.scorecard  # type: ignore[assignment]
        pad = lay.pad
        x0, w = pad + self.S(4), lay.w - 2 * pad
        self._kicker(canvas, "Session complete", x0, self.S(40))
        self._t(canvas, "Your scorecard", x0, self.S(64), 52, TEXT, "bold")
        # Headline numbers on the right.
        rx = pad + w
        self._t(canvas, f"{sc.held} / {sc.total}", rx, self.S(86), 60, TEXT, "bold", "rm")
        self._kicker(canvas, "Stances held", rx, self.S(40), MUTED, "rt")

        top = self.S(160)
        bottom = lay.h - self.S(96)
        left_w = int(w * 0.6)
        table = Rect(pad, top, left_w, bottom - top)
        side = Rect(pad + left_w + self.S(24), top, w - left_w - self.S(24), bottom - top)
        fill_round(canvas, table, self.S(22), PANEL)
        fill_round(canvas, side, self.S(22), PANEL)
        self._limb_table(canvas, table, sc)
        self._rep_list(canvas, side, sc)

        left = math.ceil(v.scorecard_left_s)
        self._t(canvas, f"Next visitor in {left} s   ·   Space to start now", lay.w // 2,
                lay.h - self.S(48), 22, MUTED, "medium", "mm")

    def _limb_table(self, canvas, r: Rect, sc: Scorecard) -> None:
        x = r.x + self.S(32)
        self._kicker(canvas, "Limb accuracy", x, r.y + self.S(36), MUTED)
        # Legend
        lx = r.x2 - self.S(32)
        lx -= self._t(canvas, "Last try", lx, r.y + self.S(44), 18, TEXT, "medium", "rm")
        lx -= self.S(12)
        cv2.circle(canvas, (lx * 4, (r.y + self.S(44)) * 4), self.S(7) * 4, ACCENT, -1,
                   cv2.LINE_AA, 2)
        lx -= self.S(28)
        lx -= self._t(canvas, "First try", lx, r.y + self.S(44), 18, MUTED, "medium", "rm")
        lx -= self.S(12)
        cv2.circle(canvas, (lx * 4, (r.y + self.S(44)) * 4), self.S(7) * 4, DIM, -1,
                   cv2.LINE_AA, 2)

        rows = list(sc.limbs)
        y0 = r.y + self.S(80)
        row_h = (r.y2 - self.S(24) - y0) // max(1, len(rows))
        label_w = self.S(190)
        num_w = self.S(150)
        bar_x = x + label_w
        bar_w = r.x2 - self.S(32) - num_w - bar_x
        green_at = accuracy_pct(self.tolerance_deg) or 0.0
        bh = max(6, min(self.S(12), row_h // 5))
        for i, ls in enumerate(rows):
            cy = y0 + i * row_h + row_h // 2
            if i == 4:  # arms above, legs below
                cv2.line(canvas, (x, y0 + i * row_h), (r.x2 - self.S(32), y0 + i * row_h), LINE,
                         1, cv2.LINE_AA)
            self._t(canvas, ls.limb.label, x, cy, 22, TEXT, "medium", "lm")
            for j, (pct, colour) in enumerate(((ls.first_pct, DIM), (ls.last_pct, ACCENT))):
                by = cy - bh - self.S(2) if j == 0 else cy + self.S(2)
                track = Rect(bar_x, by, bar_w, bh)
                fill_round(canvas, track, bh // 2, LINE)
                if pct is not None:
                    fw = max(bh, int(bar_w * pct / 100))
                    fill_round(canvas, Rect(bar_x, by, fw, bh), bh // 2, colour)
            # Tick where a limb turns green.
            gx = bar_x + int(bar_w * green_at / 100)
            cv2.line(canvas, (gx, cy - bh - self.S(6)), (gx, cy + bh + self.S(6)), MUTED, 1,
                     cv2.LINE_AA)
            first = "–" if ls.first_pct is None else f"{ls.first_pct:.0f}%"
            last = "–" if ls.last_pct is None else f"{ls.last_pct:.0f}%"
            nx = r.x2 - self.S(32)
            nx -= self._t(canvas, last, nx, cy, 24, TEXT, "bold", "rm")
            nx -= self._arrow(canvas, nx - self.S(10), cy, self.S(16), DIM) + self.S(20)
            self._t(canvas, first, nx, cy, 20, MUTED, "medium", "rm")
        self._t(canvas, "| marks where a limb turns green", bar_x, r.y2 - self.S(20), 15, DIM,
                "medium", "lm")

    def _rep_list(self, canvas, r: Rect, sc: Scorecard) -> None:
        x = r.x + self.S(32)
        self._kicker(canvas, "Your stances", x, r.y + self.S(36), MUTED)
        y = r.y + self.S(96)
        row_h = min(self.S(72), (r.h - self.S(200)) // max(1, len(sc.reps)))
        for rep in sc.reps:
            colour = self.good if rep.held else self.bad
            cv2.circle(canvas, ((x + self.S(9)) * 4, y * 4), self.S(9) * 4, colour, -1,
                       cv2.LINE_AA, 2)
            name = self.name(rep.stance_id) + (" (other side)" if rep.mirrored else "")
            name = self.text.fit(name, self.S(24), "medium", r.w - self.S(200))
            self._t(canvas, name, x + self.S(32), y, 24, TEXT, "medium", "lm")
            result = "held" if rep.held else "not held"
            self._t(canvas, result, r.x2 - self.S(32), y, 20, MUTED, "medium", "rm")
            y += row_h
        if sc.first_mean_deg is not None and sc.last_mean_deg is not None:
            y = r.y2 - self.S(96)
            cv2.line(canvas, (x, y - self.S(28)), (r.x2 - self.S(32), y - self.S(28)), LINE, 1,
                     cv2.LINE_AA)
            self._kicker(canvas, "Average limb error", x, y, MUTED, "lm")
            first, last = round(sc.first_mean_deg), round(sc.last_mean_deg)
            ty = y + self.S(44)
            ax = x + self._t(canvas, f"{first}°", x, ty, 36, MUTED, "bold", "lm") + self.S(16)
            ax += self._arrow(canvas, ax + self.S(26), ty, self.S(26), DIM) + self.S(16)
            self._t(canvas, f"{last}°", ax, ty, 36, TEXT, "bold", "lm")
            better = last < first
            note = "closer to the robot" if better else "first try to last"
            self._t(canvas, note, r.x2 - self.S(32), y + self.S(44), 20,
                    self.good if better else MUTED, "medium", "rm")
