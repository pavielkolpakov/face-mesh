"""Golden tests: run real MediaPipe against sample photos, compare to committed JSON.

Skips when the face_landmarker.task model is not present locally.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from face_mesh.composition import build_face_system

REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / "face_landmarker.task"
SAMPLES = [
    ("fat_face", REPO / "fat_face.png"),
    ("fit_face", REPO / "fit_face.png"),
    ("skinny_face", REPO / "skinny_face.png"),
]
GOLDEN_DIR = Path(__file__).parent / "expected"
TOL = 1e-3


def _assert_close(a, b, path="$"):
    if isinstance(a, dict):
        assert isinstance(b, dict), path
        assert a.keys() == b.keys(), f"keys differ at {path}: {a.keys() ^ b.keys()}"
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert isinstance(b, list), path
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            assert a == b, path
        else:
            assert math.isclose(float(a), float(b), abs_tol=TOL), f"{path}: {a} vs {b}"
    else:
        # ignore volatile fields
        if path.endswith(".generatedAt") or path.endswith(".jobId"):
            return
        assert a == b, f"{path}: {a} vs {b}"


@pytest.mark.skipif(not MODEL.exists(), reason="face_landmarker.task not present")
@pytest.mark.parametrize("name,image", SAMPLES)
def test_face_golden(name: str, image: Path, regen_golden: bool):
    if not image.exists():
        pytest.skip(f"sample {image} missing")

    system = build_face_system()
    try:
        system.runner.start()
        detection = system.runner.detect_from_path(image)
        payload = system.pipeline.process(detection, job_id=f"golden-{name}")
    finally:
        system.runner.close()

    # strip volatile fields for storage / comparison
    payload["generatedAt"] = None
    payload["jobId"] = f"golden-{name}"

    golden_path = GOLDEN_DIR / f"{name}.json"
    if regen_golden:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps(payload, indent=2, default=str))
        pytest.skip(f"regenerated {golden_path}")

    if not golden_path.exists():
        pytest.skip(f"no golden at {golden_path}; run with --regen-golden")

    expected = json.loads(golden_path.read_text())
    _assert_close(payload, expected)
