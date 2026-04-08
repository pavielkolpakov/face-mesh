from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FaceDetectionResult:
    landmarks: Any
    shape_weights: dict[str, float]
    head_pose_matrix: list[list[float]] | None


@dataclass(frozen=True)
class FaceFeaturesPayload:
    shape_weights: dict[str, float]
    custom_blendshapes: dict[str, float]
    custom_blendshapes_raw: dict[str, float]
    custom_blendshape_confidence: dict[str, float]
    head_pose_matrix: list[list[float]] | None
    skin_color: list[float]
    eye_color: list[float]
    hair_color: list[float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "shapeWeights": self.shape_weights,
            "customBlendshapes": self.custom_blendshapes,
            "customBlendshapesRaw": self.custom_blendshapes_raw,
            "customBlendshapeConfidence": self.custom_blendshape_confidence,
            "headPoseMatrix": self.head_pose_matrix,
            "skinColor": self.skin_color,
            "eyeColor": self.eye_color,
            "hairColor": self.hair_color,
        }


@dataclass(frozen=True)
class BodyViewEstimate:
    betas: list[float] | None
    confidence: float | None
    ok: bool
    message: str | None = None


@dataclass(frozen=True)
class BodyShapeFused:
    betas: list[float]
    betas_front: list[float] | None
    betas_side: list[float] | None
    confidence_front: float | None
    confidence_side: float | None
    fusion_weights: dict[str, float]
    body_params: dict[str, float]
    ok: bool
    message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "bodyParams": self.body_params,
            "betas": self.betas,
            "betasFront": self.betas_front,
            "betasSide": self.betas_side,
            "confidenceFront": self.confidence_front,
            "confidenceSide": self.confidence_side,
            "fusionWeights": self.fusion_weights,
            "ok": self.ok,
            "message": self.message,
        }
