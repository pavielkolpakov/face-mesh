from dataclasses import dataclass

import numpy as np

from face_mesh.domain.types import FACE_PARAM_NAMES, FaceDetection
from face_mesh.extractors.face import FaceExtractor
from face_mesh.extractors.face_appearance import FaceAppearanceExtractor
from face_mesh.mapping.mapper import FaceMapper
from face_mesh.worker.pipeline import FacePipeline


@dataclass
class _LM:
    x: float
    y: float
    z: float = 0.0


def _lm():
    pts = [_LM(0.5, 0.5) for _ in range(478)]
    pts[234] = _LM(0.3, 0.5)
    pts[454] = _LM(0.7, 0.5)
    pts[172] = _LM(0.35, 0.7)
    pts[397] = _LM(0.65, 0.7)
    pts[152] = _LM(0.5, 0.85)
    pts[17] = _LM(0.5, 0.65)
    pts[10] = _LM(0.5, 0.2)
    pts[33] = _LM(0.4, 0.45); pts[133] = _LM(0.46, 0.45); pts[159] = _LM(0.43, 0.43)
    pts[145] = _LM(0.43, 0.47); pts[160] = _LM(0.42, 0.44); pts[144] = _LM(0.42, 0.46)
    pts[263] = _LM(0.6, 0.45); pts[362] = _LM(0.54, 0.45); pts[386] = _LM(0.57, 0.43)
    pts[374] = _LM(0.57, 0.47); pts[387] = _LM(0.58, 0.44); pts[373] = _LM(0.58, 0.46)
    pts[98] = _LM(0.46, 0.57); pts[327] = _LM(0.54, 0.57); pts[6] = _LM(0.5, 0.5)
    pts[4] = _LM(0.5, 0.58); pts[168] = _LM(0.5, 0.48); pts[1] = _LM(0.5, 0.55)
    pts[13] = _LM(0.5, 0.71); pts[14] = _LM(0.5, 0.75); pts[78] = _LM(0.44, 0.73)
    pts[308] = _LM(0.56, 0.73); pts[105] = _LM(0.43, 0.38); pts[334] = _LM(0.57, 0.38)
    pts[116] = _LM(0.38, 0.6); pts[345] = _LM(0.62, 0.6)
    return pts


def _detection():
    return FaceDetection(
        landmarks=_lm(),
        shape_weights={"jawOpen": 0.1},
        transform_matrix=[[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]],
        image_bgr=np.full((100, 100, 3), 120, dtype=np.uint8),
        image_width=100,
        image_height=100,
    )


def test_pipeline_builds_payload_with_version_and_face_fields():
    bounds = {n: (0.0, 2.0) for n in FACE_PARAM_NAMES}
    mapper = FaceMapper.from_heuristic_bounds(bounds)
    pipe = FacePipeline(
        face_extractor=FaceExtractor(),
        appearance_extractor=FaceAppearanceExtractor(),
        face_mapper=mapper,
    )
    payload = pipe.process(_detection(), job_id="job-1")

    assert payload["version"] == "1.0.0"
    assert payload["jobId"] == "job-1"
    assert "generatedAt" in payload
    assert set(payload["faceParams"].keys()) == set(FACE_PARAM_NAMES)
    assert payload["faceMeasurements"]["raw"]
    assert payload["confidence"]["overall"] >= 0.0
    assert payload["appearance"]["skinColor"]
    assert payload["bodyParams"] is None
    assert payload["shapeWeights"] == {"jawOpen": 0.1}
