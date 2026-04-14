from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

FACE_PARAM_NAMES: tuple[str, ...] = (
    "faceWidth",
    "jawWidth",
    "chinLength",
    "cheekboneHeight",
    "eyeSize",
    "eyeSpacing",
    "noseSize",
    "noseBridgeHeight",
    "noseTipSize",
    "lipFullness",
    "mouthWidth",
    "browHeight",
)

BODY_PARAM_NAMES: tuple[str, ...] = (
    "bodyFat",
    "muscleMass",
    "shoulderWidth",
    "chestSize",
    "waistSize",
    "hipSize",
    "armThickness",
    "legThickness",
    "legLength",
    "neckLength",
)


@dataclass(frozen=True)
class FaceDetection:
    """Raw MediaPipe face landmarker output plus the source image."""

    landmarks: list[Any]  # MediaPipe normalized landmarks (keep as list for mypy simplicity)
    shape_weights: dict[str, float]  # ARKit 52 blendshape scores
    transform_matrix: list[list[float]] | None
    image_bgr: np.ndarray
    image_width: int
    image_height: int


@dataclass(frozen=True)
class FaceMeasurements:
    raw: dict[str, float]  # 12 raw ratios
    confidence: dict[str, float]  # per-measurement confidence 0..1
    frontal_score: float  # 0..1

    def feature_vector(self, names: list[str]) -> np.ndarray:
        return np.array([self.raw.get(n, self.frontal_score) for n in names])


@dataclass(frozen=True)
class FaceAppearance:
    skin_color: list[float]
    eye_color: list[float]
    hair_color: list[float]
    head_pose_matrix: list[list[float]] | None


@dataclass(frozen=True)
class BodyViewMeasurements:
    view: str  # "front" | "side"
    raw: dict[str, float]
    confidence: dict[str, float]


@dataclass(frozen=True)
class BodyMeasurements:
    fused: dict[str, float]
    per_view: dict[str, dict[str, float]]  # {"front": {...}, "side": {...}}
    confidence: dict[str, float]


@dataclass(frozen=True)
class Job:
    id: str
    app_user_id: str
    retry_count: int
    face_asset_id: str | None = None
    body_front_asset_id: str | None = None
    body_side_asset_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
