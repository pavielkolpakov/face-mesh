from pathlib import Path
from typing import Any, Protocol

from face_mesh.domain.entities import BodyViewEstimate, FaceDetectionResult


class FaceLandmarkPort(Protocol):
    def detect(self, image_bgr: Any, model_path: Path) -> FaceDetectionResult | None: ...


class BodyShapeEstimatorPort(Protocol):
    def estimate(self, image_bgr: Any) -> BodyViewEstimate: ...
