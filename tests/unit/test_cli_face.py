import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from typer.testing import CliRunner

from face_mesh.cli.main import app
from face_mesh.domain.types import FaceDetection

runner = CliRunner()


def _synthetic_detection(image_path: Path) -> FaceDetection:
    # reuse the same landmark fixture as extractor tests
    from dataclasses import dataclass

    @dataclass
    class _LM:
        x: float
        y: float
        z: float = 0.0

    pts = [_LM(0.5, 0.5) for _ in range(478)]
    pts[234] = _LM(0.3, 0.5); pts[454] = _LM(0.7, 0.5)
    pts[172] = _LM(0.35, 0.7); pts[397] = _LM(0.65, 0.7)
    pts[152] = _LM(0.5, 0.85); pts[17] = _LM(0.5, 0.65); pts[10] = _LM(0.5, 0.2)
    pts[33] = _LM(0.4, 0.45); pts[133] = _LM(0.46, 0.45); pts[159] = _LM(0.43, 0.43)
    pts[145] = _LM(0.43, 0.47); pts[160] = _LM(0.42, 0.44); pts[144] = _LM(0.42, 0.46)
    pts[263] = _LM(0.6, 0.45); pts[362] = _LM(0.54, 0.45); pts[386] = _LM(0.57, 0.43)
    pts[374] = _LM(0.57, 0.47); pts[387] = _LM(0.58, 0.44); pts[373] = _LM(0.58, 0.46)
    pts[98] = _LM(0.46, 0.57); pts[327] = _LM(0.54, 0.57); pts[6] = _LM(0.5, 0.5)
    pts[4] = _LM(0.5, 0.58); pts[168] = _LM(0.5, 0.48); pts[1] = _LM(0.5, 0.55)
    pts[13] = _LM(0.5, 0.71); pts[14] = _LM(0.5, 0.75); pts[78] = _LM(0.44, 0.73)
    pts[308] = _LM(0.56, 0.73); pts[105] = _LM(0.43, 0.38); pts[334] = _LM(0.57, 0.38)
    pts[116] = _LM(0.38, 0.6); pts[345] = _LM(0.62, 0.6)

    return FaceDetection(
        landmarks=pts,
        shape_weights={},
        transform_matrix=None,
        image_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
        image_width=100,
        image_height=100,
    )


def test_cli_face_prints_payload_json(tmp_path: Path):
    img = tmp_path / "face.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")  # placeholder; detection is mocked

    with patch("face_mesh.runners.face_landmarker.FaceLandmarkerRunner.start", lambda self: None), \
         patch("face_mesh.runners.face_landmarker.FaceLandmarkerRunner.close", lambda self: None), \
         patch(
             "face_mesh.runners.face_landmarker.FaceLandmarkerRunner.detect_from_path",
             lambda self, p: _synthetic_detection(p),
         ):
        result = runner.invoke(app, ["face", str(img)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["version"] == "1.0.0"
    assert "faceParams" in payload
