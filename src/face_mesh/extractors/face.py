"""Pure function over a FaceDetection → raw geometric measurements + confidence.

Ported from legacy main.py. No normalization happens here — that belongs in the mapper.
"""

from __future__ import annotations

from collections.abc import Iterable

from face_mesh.domain.errors import InputError
from face_mesh.domain.types import FaceDetection, FaceMeasurements


def _xy(lm, idx: int) -> tuple[float, float]:
    p = lm[idx]
    return float(p.x), float(p.y)


def _dist(lm, a: int, b: int) -> float:
    ax, ay = _xy(lm, a)
    bx, by = _xy(lm, b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _mean(lm, indices: Iterable[int]) -> tuple[float, float]:
    pts = [_xy(lm, i) for i in indices]
    n = len(pts)
    return sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _norm_range(v: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return _clamp01((v - lo) / (hi - lo))


class FaceExtractor:
    """Detection → FaceMeasurements. No I/O, no MediaPipe dependency."""

    def measure(self, detection: FaceDetection) -> FaceMeasurements:
        lm = detection.landmarks
        if not lm:
            raise InputError("no face detected")

        # Scale: interocular center distance if available, else face width fallback.
        left_eye_center = _mean(lm, [33, 133, 159, 145, 160, 144])
        right_eye_center = _mean(lm, [263, 362, 386, 374, 387, 373])
        eye_dx = left_eye_center[0] - right_eye_center[0]
        eye_dy = left_eye_center[1] - right_eye_center[1]
        eye_scale = (eye_dx * eye_dx + eye_dy * eye_dy) ** 0.5
        fallback_scale = _dist(lm, 234, 454)
        scale = eye_scale if eye_scale > 1e-6 else max(fallback_scale, 1e-6)

        raw = {
            "faceWidth": _dist(lm, 234, 454) / scale,
            "jawWidth": _dist(lm, 172, 397) / scale,
            "chinLength": _dist(lm, 152, 17) / scale,
        }

        # Cheekbone: relative vertical position of cheek midpoint between eye line and chin.
        _, eye_y = _mean(lm, [159, 145, 386, 374])
        _, chin_y = _xy(lm, 152)
        _, cheek_y_l = _xy(lm, 116)
        _, cheek_y_r = _xy(lm, 345)
        cheek_y = 0.5 * (cheek_y_l + cheek_y_r)
        raw["cheekboneHeight"] = 1.0 - ((cheek_y - eye_y) / max(chin_y - eye_y, 1e-6))

        # Eye size: aperture / width, averaged across eyes.
        l_aperture = _dist(lm, 159, 145)
        r_aperture = _dist(lm, 386, 374)
        l_width = max(_dist(lm, 33, 133), 1e-6)
        r_width = max(_dist(lm, 263, 362), 1e-6)
        raw["eyeSize"] = 0.5 * ((l_aperture / l_width) + (r_aperture / r_width))
        raw["eyeSpacing"] = _dist(lm, 133, 362) / scale

        nose_width = _dist(lm, 98, 327)
        nose_length = _dist(lm, 6, 4)
        raw["noseSize"] = (nose_width + nose_length) / (2.0 * scale)
        raw["noseBridgeHeight"] = _dist(lm, 168, 6) / scale
        raw["noseTipSize"] = nose_width / scale

        raw["lipFullness"] = _dist(lm, 13, 14) / scale
        raw["mouthWidth"] = _dist(lm, 78, 308) / scale
        raw["browHeight"] = 0.5 * ((_dist(lm, 105, 159) / scale) + (_dist(lm, 334, 386) / scale))

        # Frontal score: how centered the nose tip is between the eye centers.
        nose_x, _ = _xy(lm, 1)
        mid_eye_x = 0.5 * (left_eye_center[0] + right_eye_center[0])
        center_offset = abs(nose_x - mid_eye_x) / max(scale, 1e-6)
        frontal_score = 1.0 - _norm_range(center_offset, 0.08, 0.22)

        # Confidence: base from frontal + expression penalty.
        sw = detection.shape_weights
        blink = max(sw.get("eyeBlinkLeft", 0.0), sw.get("eyeBlinkRight", 0.0))
        mouth_open = sw.get("jawOpen", 0.0)
        lip_motion = max(
            sw.get("mouthSmileLeft", 0.0),
            sw.get("mouthSmileRight", 0.0),
            sw.get("mouthPucker", 0.0),
            sw.get("mouthFunnel", 0.0),
        )
        expression_penalty = _clamp01(max(blink, mouth_open, lip_motion))
        base_conf = _clamp01(0.35 + 0.65 * frontal_score) * _clamp01(1.0 - 0.5 * expression_penalty)

        confidence = {
            "faceWidth": base_conf,
            "jawWidth": base_conf,
            "chinLength": base_conf * 0.85,
            "cheekboneHeight": base_conf * 0.75,
            "eyeSize": base_conf * _clamp01(1.0 - blink),
            "eyeSpacing": base_conf,
            "noseSize": base_conf * 0.9,
            "noseBridgeHeight": base_conf * 0.7,
            "noseTipSize": base_conf * 0.8,
            "lipFullness": base_conf * _clamp01(1.0 - max(mouth_open, lip_motion)),
            "mouthWidth": base_conf * _clamp01(1.0 - 0.5 * max(mouth_open, lip_motion)),
            "browHeight": base_conf * 0.9,
        }
        return FaceMeasurements(raw=raw, confidence=confidence, frontal_score=frontal_score)
