"""Orchestrates face extraction → mapping → payload build. Pure of I/O."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from face_mesh.domain.types import FaceDetection
from face_mesh.extractors.face import FaceExtractor
from face_mesh.extractors.face_appearance import FaceAppearanceExtractor
from face_mesh.mapping.mapper import FaceMapper

PAYLOAD_VERSION = "1.0.0"


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class FacePipeline:
    face_extractor: FaceExtractor
    appearance_extractor: FaceAppearanceExtractor
    face_mapper: FaceMapper

    def process(self, detection: FaceDetection, job_id: str) -> dict[str, Any]:
        measurements = self.face_extractor.measure(detection)
        appearance = self.appearance_extractor.sample(detection)
        face_params = self.face_mapper.predict(measurements)

        face_conf = measurements.confidence
        overall = sum(face_conf.values()) / max(len(face_conf), 1)

        return {
            "version": PAYLOAD_VERSION,
            "jobId": job_id,
            "generatedAt": _now_iso(),
            "faceParams": face_params,
            "bodyParams": None,
            "faceMeasurements": {
                "raw": measurements.raw,
                "frontalScore": measurements.frontal_score,
            },
            "bodyMeasurements": None,
            "confidence": {
                "face": face_conf,
                "body": None,
                "overall": overall,
            },
            "appearance": {
                "skinColor": appearance.skin_color,
                "eyeColor": appearance.eye_color,
                "hairColor": appearance.hair_color,
            },
            "shapeWeights": dict(detection.shape_weights),
            "headPoseMatrix": appearance.head_pose_matrix,
        }
