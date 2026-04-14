from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from face_mesh.domain.errors import InputError
from face_mesh.domain.types import FACE_PARAM_NAMES, FaceDetection
from face_mesh.extractors.face import FaceExtractor


@dataclass
class _LM:
    x: float
    y: float
    z: float = 0.0


def _symmetric_landmarks() -> list[_LM]:
    """Build 478 landmarks; override only the indices the extractor reads."""
    pts = [_LM(0.5, 0.5) for _ in range(478)]
    # horizontal axis: left > 0.5 (viewer-right), right < 0.5
    # face corners
    pts[234] = _LM(0.30, 0.50)  # face left
    pts[454] = _LM(0.70, 0.50)  # face right
    # jaw corners
    pts[172] = _LM(0.35, 0.70)
    pts[397] = _LM(0.65, 0.70)
    # chin / lower-lip
    pts[152] = _LM(0.50, 0.85)
    pts[17] = _LM(0.50, 0.65)
    # forehead mid
    pts[10] = _LM(0.50, 0.20)
    # eye left (inner, outer, upper, lower, extras)
    pts[33] = _LM(0.40, 0.45)
    pts[133] = _LM(0.46, 0.45)
    pts[159] = _LM(0.43, 0.43)
    pts[145] = _LM(0.43, 0.47)
    pts[160] = _LM(0.42, 0.44)
    pts[144] = _LM(0.42, 0.46)
    # eye right
    pts[263] = _LM(0.60, 0.45)
    pts[362] = _LM(0.54, 0.45)
    pts[386] = _LM(0.57, 0.43)
    pts[374] = _LM(0.57, 0.47)
    pts[387] = _LM(0.58, 0.44)
    pts[373] = _LM(0.58, 0.46)
    # nose: wings, bridge, tip
    pts[98] = _LM(0.46, 0.57)
    pts[327] = _LM(0.54, 0.57)
    pts[6] = _LM(0.50, 0.50)
    pts[4] = _LM(0.50, 0.58)
    pts[168] = _LM(0.50, 0.48)
    # nose x point
    pts[1] = _LM(0.50, 0.55)
    # lips
    pts[13] = _LM(0.50, 0.71)
    pts[14] = _LM(0.50, 0.75)
    pts[78] = _LM(0.44, 0.73)
    pts[308] = _LM(0.56, 0.73)
    # brows
    pts[105] = _LM(0.43, 0.38)
    pts[334] = _LM(0.57, 0.38)
    # cheeks
    pts[116] = _LM(0.38, 0.60)
    pts[345] = _LM(0.62, 0.60)
    return pts


def _detection(landmarks: list[_LM]) -> FaceDetection:
    return FaceDetection(
        landmarks=landmarks,
        shape_weights={},
        transform_matrix=None,
        image_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
        image_width=100,
        image_height=100,
    )


def test_face_extractor_returns_all_12_measurements():
    m = FaceExtractor().measure(_detection(_symmetric_landmarks()))
    for name in FACE_PARAM_NAMES:
        assert name in m.raw, f"missing {name}"
        assert name in m.confidence
    assert 0.0 <= m.frontal_score <= 1.0


def test_face_extractor_symmetric_face_has_high_frontal_score():
    m = FaceExtractor().measure(_detection(_symmetric_landmarks()))
    assert m.frontal_score > 0.8


def test_face_extractor_off_center_nose_drops_frontal_score():
    pts = _symmetric_landmarks()
    pts[1] = _LM(0.70, 0.55)  # nose far off-center
    m = FaceExtractor().measure(_detection(pts))
    assert m.frontal_score < 0.5


def test_face_extractor_raises_input_error_when_no_landmarks():
    det = FaceDetection(
        landmarks=[],
        shape_weights={},
        transform_matrix=None,
        image_bgr=np.zeros((10, 10, 3), dtype=np.uint8),
        image_width=10,
        image_height=10,
    )
    with pytest.raises(InputError):
        FaceExtractor().measure(det)


def test_face_extractor_blink_reduces_eye_confidence():
    det = _detection(_symmetric_landmarks())
    neutral = FaceExtractor().measure(det)
    det_blink = FaceDetection(
        landmarks=det.landmarks,
        shape_weights={"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.9},
        transform_matrix=None,
        image_bgr=det.image_bgr,
        image_width=100,
        image_height=100,
    )
    blinking = FaceExtractor().measure(det_blink)
    assert blinking.confidence["eyeSize"] < neutral.confidence["eyeSize"]
