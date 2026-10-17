"""Procedural dojo textures (tatami, shoji, wood, plaster, a calligraphy scroll), written as PNGs.

Each generator is seeded, so the files are reproducible. ``ensure_textures``
writes any missing file under ``assets/dojo/`` and returns their paths; the
version in each file name lets a changed generator replace stale images.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from rth.config import project_path

TEXTURE_DIR = "assets/dojo"
VERSION = 5  # bump when a generator changes so old PNGs are regenerated

PX_PER_M = 1024 / 3.6  # tatami tile: 3.6 m x 1.8 m (two mats wide, two rows)
MAT_L, MAT_W = 1.8, 0.9  # one tatami mat, metres
SHOJI_BAY = (0.9, 1.8)  # one shoji panel (width, height), metres
WOOD_TILE_M = 1.2

Image = np.ndarray  # RGB uint8 (h, w, 3); converted to BGR only when written


def _noise(rng: np.random.Generator, h: int, w: int, scale: int) -> np.ndarray:
    """Smooth value noise in [-1, 1]: random grid upsampled with bicubic interpolation."""
    gh, gw = max(h // scale, 2), max(w // scale, 2)
    g = rng.uniform(-1.0, 1.0, (gh, gw)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


def _tileable(img: np.ndarray, blend_px: int) -> np.ndarray:
    """Cross-fade the edges with the opposite side so a repeated image has no seams."""
    out = img.astype(np.float32).copy()
    for axis in (0, 1):
        n = out.shape[axis]
        b = min(blend_px, n // 4)
        ramp = np.linspace(0.0, 1.0, b, dtype=np.float32)
        shape = [1] * out.ndim
        shape[axis] = b
        ramp = ramp.reshape(shape)
        head = np.take(out, np.arange(b), axis=axis)
        tail = np.take(out, np.arange(n - b, n), axis=axis)
        # The last b pixels fade towards the first b, so the wrap is continuous.
        mixed = tail * (1 - ramp) + head * ramp
        idx = [slice(None)] * out.ndim
        idx[axis] = slice(n - b, n)
        out[tuple(idx)] = mixed
    return out


def _to_u8(rgb: np.ndarray) -> Image:
    return np.clip(rgb, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- tatami


def _straw(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Woven rush for one mat, long axis along the image width.

    The rush straws run across the mat (short direction), so each image column
    is one straw with its own shade; the warp threads show as faint lines along
    the length.
    """
    base = np.array([196, 176, 118], np.float32)  # aged golden rush, a touch of green
    col = rng.normal(0.0, 1.0, w).astype(np.float32)
    col = cv2.GaussianBlur(col.reshape(1, -1), (0, 0), 0.8).ravel()
    weave = 0.5 + 0.5 * np.cos(np.arange(w, dtype=np.float32) * (2 * np.pi / 3.0))
    shade = 1.0 + 0.045 * col[None, :] - 0.035 * weave[None, :]
    warp = np.ones(h, np.float32)
    warp[(np.arange(h) % 18) < 1] = 0.93
    mottle = 1.0 + 0.05 * _noise(rng, h, w, 40) + 0.025 * _noise(rng, h, w, 9)
    tint = 1.0 + 0.012 * rng.normal(0.0, 1.0, 3).astype(
        np.float32
    )  # each mat aged a bit differently
    light = (shade * warp[:, None] * mottle)[..., None]
    return base[None, None, :] * tint * light


def _heri(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """The dark cloth border on a mat's long edge."""
    base = np.array([66, 60, 46], np.float32)
    fine = 1.0 + 0.08 * _noise(rng, h, w, 3)
    return base[None, None, :] * fine[..., None]


def _mat(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """One mat (h = short side, w = long side): straw, cloth borders, darker short-end seams."""
    img = _straw(rng, h, w)
    border = max(round(0.024 * PX_PER_M), 3)
    img[:border] = _heri(rng, border, w)
    img[-border:] = _heri(rng, border, w)
    seam = 2
    img[:, :seam] *= 0.72
    img[:, -seam:] *= 0.72
    # A slight darkening just inside the border, where the straw is pressed.
    img[border : border + 2] *= 0.9
    img[-border - 2 : -border] *= 0.9
    return img


def tatami(seed: int = 1) -> Image:
    """A 3.6 m x 1.8 m tile of tatami laid in running bond (second row offset by half a mat)."""
    rng = np.random.default_rng(seed)
    w = round(2 * MAT_L * PX_PER_M)
    h = round(2 * MAT_W * PX_PER_M)
    mw, mh = w // 2, h // 2
    out = np.zeros((h, w, 3), np.float32)
    out[:mh, :mw] = _mat(rng, mh, mw)
    out[:mh, mw:] = _mat(rng, mh, w - mw)
    row = np.zeros((h - mh, w + mw, 3), np.float32)
    row[:, :mw] = _mat(rng, h - mh, mw)
    row[:, mw : 2 * mw] = _mat(rng, h - mh, mw)
    row[:, 2 * mw :] = _mat(rng, h - mh, w + mw - 2 * mw)
    # Shift the second row by half a mat; the wrap keeps the tile seamless.
    out[mh:] = np.roll(row[:, :w], mw // 2, axis=1)
    return _to_u8(out)


# --------------------------------------------------------------------------- walls


def _wood_grain(rng: np.random.Generator, h: int, w: int, base: tuple[int, int, int]) -> np.ndarray:
    """Vertical-grain wood: stretched noise for the grain, a few darker streaks."""
    grain = _noise(rng, h, w * 6, 2)
    grain = cv2.resize(grain, (w, h), interpolation=cv2.INTER_AREA)
    long = cv2.resize(rng.uniform(-1, 1, (4, w)).astype(np.float32), (w, h), cv2.INTER_CUBIC)
    rings = np.sin(np.linspace(0, 40 * np.pi, w, dtype=np.float32)[None, :] + 3.0 * long)
    light = 1.0 + 0.06 * grain + 0.05 * rings + 0.04 * _noise(rng, h, w, 64)
    return np.array(base, np.float32)[None, None, :] * light[..., None]


def wood(seed: int = 2, base: tuple[int, int, int] = (118, 78, 48)) -> Image:
    """Wainscot of vertical planks (15 cm), 1.2 m square tile."""
    rng = np.random.default_rng(seed)
    n = 512
    img = _wood_grain(rng, n, n, base)
    plank = n // 8
    for i in range(8):
        sl = slice(i * plank, (i + 1) * plank)
        img[:, sl] *= 1.0 + 0.06 * rng.normal()
        img[:, i * plank : i * plank + 2] *= 0.55  # the joint between planks
    return _to_u8(_tileable(img, 24))


def shoji(seed: int = 3) -> Image:
    """One shoji panel, 0.9 m x 1.8 m: warm rice paper in a dark kumiko lattice.

    The outer frame is half width on every edge, so neighbouring panels make a
    full-width stile when the texture repeats.
    """
    rng = np.random.default_rng(seed)
    ppm = 300
    w, h = int(SHOJI_BAY[0] * ppm), int(SHOJI_BAY[1] * ppm)
    paper = np.array([226, 214, 186], np.float32)
    fibre = 1.0 + 0.025 * _noise(rng, h, w, 6) + 0.02 * _noise(rng, h, w, 50)
    img = paper[None, None, :] * fibre[..., None]
    frame_rgb = _wood_grain(rng, h, w, (96, 64, 40))
    mask = np.zeros((h, w), bool)
    half_frame, bar = int(0.022 * ppm), int(0.012 * ppm)
    mask[:half_frame] = mask[-half_frame:] = True
    mask[:, :half_frame] = mask[:, -half_frame:] = True
    for k in (1, 2):
        x = int(k * w / 3)
        mask[:, x - bar // 2 : x + bar // 2] = True
    for k in range(1, 7):
        y = int(k * h / 7)
        mask[y - bar // 2 : y + bar // 2] = True
    img[mask] = frame_rgb[mask]
    # Light glowing through the paper is brighter towards the middle of each pane.
    glow = cv2.GaussianBlur((~mask).astype(np.float32), (0, 0), 6.0)
    img = img * (0.92 + 0.08 * glow)[..., None]
    return _to_u8(img)


def plaster(seed: int = 4) -> Image:
    """Warm clay plaster for the frieze above the shoji."""
    rng = np.random.default_rng(seed)
    n = 256
    base = np.array([196, 178, 150], np.float32)
    light = 1.0 + 0.035 * _noise(rng, n, n, 32) + 0.02 * _noise(rng, n, n, 4)
    return _to_u8(_tileable(base[None, None, :] * light[..., None], 32))


def scroll(seed: int = 5) -> Image:
    """A hanging scroll (kakejiku) with a brushed enso circle; 0.5 m x 1.3 m."""
    rng = np.random.default_rng(seed)
    w, h = 240, 624
    silk = np.array([92, 70, 52], np.float32)  # mounting cloth
    img = silk[None, None, :] * (1.0 + 0.04 * _noise(rng, h, w, 8))[..., None]
    x0, x1, y0, y1 = 22, w - 22, 80, h - 110
    paper = np.array([232, 224, 204], np.float32)
    img[y0:y1, x0:x1] = paper * (1.0 + 0.03 * _noise(rng, y1 - y0, x1 - x0, 12))[..., None]
    # The enso: one brush stroke, thick where the brush lands and dry at the tail.
    cx, cy, r = w // 2, (y0 + y1) // 2 - 40, 78
    ink = np.zeros((h, w), np.float32)
    for t in np.linspace(0.15, 2 * np.pi * 0.93, 400):
        frac = (t - 0.15) / (2 * np.pi * 0.93 - 0.15)
        thick = 14 * (1.0 - 0.75 * frac) + 3
        x = int(cx + r * np.cos(t + 1.9))
        y = int(cy + r * np.sin(t + 1.9))
        cv2.circle(ink, (x, y), int(thick), 1.0, -1, cv2.LINE_AA)
    dry = (_noise(rng, h, w, 2) > -0.55 - 0.4 * (ink > 0.5)).astype(np.float32)
    ink = np.clip(ink * dry, 0, 1)
    seal = np.zeros((h, w), np.float32)
    cv2.rectangle(seal, (cx + 40, y1 - 70), (cx + 62, y1 - 48), 1.0, -1)
    img = img * (1 - 0.85 * ink[..., None])
    img = img * (1 - seal[..., None]) + seal[..., None] * np.array([168, 40, 32], np.float32)
    return _to_u8(img)


GENERATORS: dict[str, Callable[[], Image]] = {
    "tatami": tatami,
    "wood": wood,
    "dark_wood": lambda: wood(seed=6, base=(78, 52, 34)),
    "shoji": shoji,
    "plaster": plaster,
    "scroll": scroll,
}


def texture_path(name: str, directory: str | Path = TEXTURE_DIR) -> Path:
    return project_path(directory) / f"{name}_v{VERSION}.png"


def _write_png(path: Path, rgb: Image) -> None:
    """Write via a temp file and rename, so a half-written PNG is never picked up."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.png")
    if not cv2.imwrite(str(tmp), rgb[:, :, ::-1]):
        raise OSError(f"could not write {tmp}")
    os.replace(tmp, path)


def ensure_textures(directory: str | Path = TEXTURE_DIR, *, force: bool = False) -> dict[str, Path]:
    """Generate any missing texture PNG; returns name -> absolute path."""
    out: dict[str, Path] = {}
    for name, make in GENERATORS.items():
        p = texture_path(name, directory)
        if force or not p.exists():
            _write_png(p, make())
        out[name] = p
    return out


def main(argv: list[str] | None = None) -> int:
    """Regenerate every texture (``python -m rth.sim.textures``)."""
    for name, p in ensure_textures(force=True).items():
        print(f"{name:10s} {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
