"""Skin / eye / hair color sampling + head pose matrix passthrough."""

from __future__ import annotations

import numpy as np

from face_mesh.domain.types import FaceAppearance, FaceDetection


def _px(lm, idx: int, w: int, h: int) -> tuple[int, int]:
    p = lm[idx]
    x = int(p.x * w)
    y = int(p.y * h)
    return max(0, min(w - 1, x)), max(0, min(h - 1, y))


def _patch_rgb(image_bgr: np.ndarray, x: int, y: int, radius: int) -> list[float]:
    h, w = image_bgr.shape[:2]
    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(w, x + radius + 1)
    y1 = min(h, y + radius + 1)
    patch = image_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return [0.0, 0.0, 0.0]
    b, g, r = patch.reshape(-1, 3).mean(axis=0)
    return [round(float(r) / 255.0, 4), round(float(g) / 255.0, 4), round(float(b) / 255.0, 4)]


def _avg(samples: list[list[float]]) -> list[float]:
    n = len(samples)
    return [round(sum(s[i] for s in samples) / n, 4) for i in range(3)]


class FaceAppearanceExtractor:
    """Detection + image → sampled appearance colors."""

    SKIN_POINTS = (234, 454, 10, 152)
    EYE_POINTS = (33, 133, 362, 263)

    def sample(self, detection: FaceDetection) -> FaceAppearance:
        lm = detection.landmarks
        img = detection.image_bgr
        w = detection.image_width
        h = detection.image_height

        skin = [_patch_rgb(img, *_px(lm, i, w, h), radius=4) for i in self.SKIN_POINTS]
        eyes = [_patch_rgb(img, *_px(lm, i, w, h), radius=2) for i in self.EYE_POINTS]

        fx, fy = _px(lm, 10, w, h)
        hair_y = max(0, fy - max(6, h // 40))
        hair = _patch_rgb(img, fx, hair_y, radius=6)

        return FaceAppearance(
            skin_color=_avg(skin),
            eye_color=_avg(eyes),
            hair_color=hair,
            head_pose_matrix=detection.transform_matrix,
        )
