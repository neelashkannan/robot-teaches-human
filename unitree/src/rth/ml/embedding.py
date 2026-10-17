"""Pose embeddings for the stance recogniser, after MediaPipe's FullBodyPoseEmbedder.

A pose (MediaPipe's 33 world landmarks) becomes a vector of landmark-pair
differences, expressed in the body's own torso frame and divided by torso size.
That makes it independent of where the visitor stands, how tall they are and
which way they face the camera. MediaPipe's original keeps camera axes; we use
the torso frame (the same one as ``rth.limbs``) so a slightly turned visitor
still matches, and so mirroring is an exact reflection across the body's
sagittal plane rather than a guess about the camera's x axis.
"""

from __future__ import annotations

import numpy as np

N_LANDMARKS = 33
_EPS = 1e-6

NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# MediaPipe's left/right landmark pairs: eyes, ears, mouth, arms, hands, legs, feet.
_LR_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
    (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
]
MIRROR_INDEX = np.arange(N_LANDMARKS)
for _a, _b in _LR_PAIRS:
    MIRROR_INDEX[_a], MIRROR_INDEX[_b] = _b, _a

# Landmark pairs whose difference vectors form the embedding (MediaPipe's list, deduplicated,
# minus hips->shoulders which is constant in the torso frame, plus nose->wrists so fists
# by the face separate a guard from fists at the hips).
PAIRS: list[tuple[int, int]] = [
    (LEFT_SHOULDER, LEFT_ELBOW), (RIGHT_SHOULDER, RIGHT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_HIP, LEFT_KNEE), (RIGHT_HIP, RIGHT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE), (RIGHT_KNEE, RIGHT_ANKLE),
    (LEFT_SHOULDER, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_WRIST),
    (LEFT_HIP, LEFT_ANKLE), (RIGHT_HIP, RIGHT_ANKLE),
    (LEFT_HIP, LEFT_WRIST), (RIGHT_HIP, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_ANKLE), (RIGHT_SHOULDER, RIGHT_ANKLE),
    (LEFT_ELBOW, RIGHT_ELBOW), (LEFT_KNEE, RIGHT_KNEE),
    (LEFT_WRIST, RIGHT_WRIST), (LEFT_ANKLE, RIGHT_ANKLE),
    (NOSE, LEFT_WRIST), (NOSE, RIGHT_WRIST),
]
EMBEDDING_DIM = 3 * len(PAIRS)
FORWARD, LEFT, UP = 0, 1, 2  # embedding axes per pair, as in rth.limbs


def _mirror_pair_map() -> tuple[np.ndarray, np.ndarray]:
    """For each pair: index of its mirror pair and the sign (-1 when the pair is reversed)."""
    lookup = {p: i for i, p in enumerate(PAIRS)}
    index, sign = [], []
    for a, b in PAIRS:
        ma, mb = int(MIRROR_INDEX[a]), int(MIRROR_INDEX[b])
        if (ma, mb) in lookup:
            index.append(lookup[(ma, mb)])
            sign.append(1.0)
        else:
            index.append(lookup[(mb, ma)])  # KeyError here means PAIRS is not mirror-closed
            sign.append(-1.0)
    return np.array(index), np.array(sign)


_PAIR_MIRROR, _PAIR_SIGN = _mirror_pair_map()


def _as_batch(landmarks: np.ndarray) -> np.ndarray:
    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.ndim != 3 or lm.shape[1:] != (N_LANDMARKS, 3):
        raise ValueError(f"expected landmarks of shape (N, 33, 3), got {lm.shape}")
    return lm


def _unit(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    ok = (n[..., 0] > _EPS) & np.isfinite(n[..., 0])
    return v / np.where(ok[..., None], n, 1.0), ok


def torso_frames(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pose torso frame, matching ``rth.limbs.torso_frame``.

    Returns (frames (N, 3, 3) with rows forward/left/up, torso size (N,), valid (N,)).
    """
    lm = _as_batch(landmarks)
    hips = (lm[:, LEFT_HIP] + lm[:, RIGHT_HIP]) / 2
    shoulders = (lm[:, LEFT_SHOULDER] + lm[:, RIGHT_SHOULDER]) / 2
    torso = shoulders - hips
    up, ok_up = _unit(torso)
    across = (lm[:, LEFT_SHOULDER] - lm[:, RIGHT_SHOULDER]) + (lm[:, LEFT_HIP] - lm[:, RIGHT_HIP])
    across = across - np.sum(across * up, axis=1, keepdims=True) * up
    left, ok_left = _unit(across)
    forward = np.cross(left, up)
    frames = np.stack([forward, left, up], axis=1)
    return frames, np.linalg.norm(torso, axis=1), ok_up & ok_left


def embed_batch(landmarks: np.ndarray) -> np.ndarray:
    """(N, 33, 3) world landmarks -> (N, EMBEDDING_DIM) embeddings; degenerate poses are NaN rows."""
    lm = _as_batch(landmarks)
    frames, size, valid = torso_frames(lm)
    a = np.array([p[0] for p in PAIRS])
    b = np.array([p[1] for p in PAIRS])
    diffs = lm[:, b] - lm[:, a]  # (N, P, 3) in input coordinates
    local = np.einsum("nij,npj->npi", frames, diffs) / np.where(valid, size, 1.0)[:, None, None]
    out = local.reshape(len(lm), EMBEDDING_DIM)
    out[~valid] = np.nan
    return out


def embed(landmarks_world: np.ndarray) -> np.ndarray | None:
    """(33, 3) world landmarks -> (EMBEDDING_DIM,) embedding, or None if the torso is degenerate."""
    e = embed_batch(np.asarray(landmarks_world)[None])[0]
    return None if np.isnan(e).any() else e


def mirror_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """The other-side version of a pose: reflect across the sagittal plane and swap left/right.

    The plane goes through mid-hip, normal to the body's left axis (from hips and
    shoulders), so it works in any coordinate frame. Accepts (33, 3) or (N, 33, 3).
    Poses with a degenerate torso only get their left/right labels swapped.
    """
    lm = np.asarray(landmarks, dtype=np.float64)
    single = lm.ndim == 2
    batch = _as_batch(lm[None] if single else lm)
    frames, _, valid = torso_frames(batch)
    normal = np.where(valid[:, None], frames[:, LEFT], 0.0)
    centre = (batch[:, LEFT_HIP] + batch[:, RIGHT_HIP]) / 2
    offset = np.sum((batch - centre[:, None]) * normal[:, None], axis=2, keepdims=True)
    reflected = batch - 2 * offset * normal[:, None]
    out = reflected[:, MIRROR_INDEX]
    return out[0] if single else out


def mirror_embedding(embeddings: np.ndarray) -> np.ndarray:
    """Mirror embeddings directly; equals ``embed_batch(mirror_landmarks(lm))``. (D,) or (N, D)."""
    e = np.asarray(embeddings, dtype=np.float64)
    shaped = e.reshape(-1, len(PAIRS), 3)[:, _PAIR_MIRROR] * _PAIR_SIGN[None, :, None]
    shaped = shaped * np.array([1.0, -1.0, 1.0])  # reflection flips the left axis
    return shaped.reshape(e.shape)
