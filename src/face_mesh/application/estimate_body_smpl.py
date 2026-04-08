from pathlib import Path

import cv2

from face_mesh.application.body_params_unity import betas_to_unity_body_params
from face_mesh.domain.entities import BodyShapeFused, BodyViewEstimate
from face_mesh.domain.ports import BodyShapeEstimatorPort


def _fuse_two_vectors(
    a: list[float] | None,
    b: list[float] | None,
    weight_a: float,
    weight_b: float,
) -> tuple[list[float] | None, dict[str, float], str | None]:
    if not a and not b:
        return None, {"front": weight_a, "side": weight_b}, "No valid body estimate from either view"
    if not a:
        return list(b), {"front": 0.0, "side": 1.0}, None
    if not b:
        return list(a), {"front": 1.0, "side": 0.0}, None
    wa, wb = weight_a, weight_b
    total = wa + wb
    if total <= 0:
        wa, wb = 0.5, 0.5
        total = 1.0
    wa /= total
    wb /= total
    fused = [round(wa * x + wb * y, 6) for x, y in zip(a, b, strict=True)]
    return fused, {"front": round(wa, 4), "side": round(wb, 4)}, None


def estimate_body_smpl_fused(
    front_path: Path,
    side_path: Path,
    estimator: BodyShapeEstimatorPort,
    weight_front: float = 0.6,
    weight_side: float = 0.4,
) -> BodyShapeFused:
    front_img = cv2.imread(str(front_path))
    side_img = cv2.imread(str(side_path))
    front_est = BodyViewEstimate(betas=None, confidence=None, ok=False, message="Failed to load image")
    side_est = BodyViewEstimate(betas=None, confidence=None, ok=False, message="Failed to load image")
    if front_img is not None:
        front_est = estimator.estimate(front_img)
    if side_img is not None:
        side_est = estimator.estimate(side_img)

    wf, ws = weight_front, weight_side
    if not front_est.ok:
        wf, ws = 0.0, 1.0
    if not side_est.ok:
        wf, ws = 1.0, 0.0
    if not front_est.ok and not side_est.ok:
        return BodyShapeFused(
            betas=[],
            betas_front=front_est.betas,
            betas_side=side_est.betas,
            confidence_front=front_est.confidence,
            confidence_side=side_est.confidence,
            fusion_weights={"front": 0.0, "side": 0.0},
            body_params=betas_to_unity_body_params([]),
            ok=False,
            message=front_est.message or side_est.message or "Body estimation failed",
        )

    fused, weights, msg = _fuse_two_vectors(
        front_est.betas,
        side_est.betas,
        wf,
        ws,
    )
    assert fused is not None
    return BodyShapeFused(
        betas=fused,
        betas_front=front_est.betas,
        betas_side=side_est.betas,
        confidence_front=front_est.confidence,
        confidence_side=side_est.confidence,
        fusion_weights=weights,
        body_params=betas_to_unity_body_params(fused),
        ok=True,
        message=msg,
    )
