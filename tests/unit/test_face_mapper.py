from pathlib import Path

import numpy as np

from face_mesh.domain.types import FACE_PARAM_NAMES, FaceMeasurements
from face_mesh.mapping.mapper import FaceMapper


def _bounds() -> dict[str, tuple[float, float]]:
    return {name: (0.0, 2.0) for name in FACE_PARAM_NAMES}


def _measurements(value: float = 1.0) -> FaceMeasurements:
    raw = {name: value for name in FACE_PARAM_NAMES}
    conf = {name: 1.0 for name in FACE_PARAM_NAMES}
    return FaceMeasurements(raw=raw, confidence=conf, frontal_score=1.0)


def test_mapper_from_bounds_predicts_all_12_params():
    mapper = FaceMapper.from_heuristic_bounds(_bounds())
    out = mapper.predict(_measurements(1.0))
    assert set(out.keys()) == set(FACE_PARAM_NAMES)
    for name in FACE_PARAM_NAMES:
        assert out[name] == 0.5


def test_mapper_uses_trained_estimator_when_available():
    mapper = FaceMapper.from_heuristic_bounds(_bounds())

    class _Fixed:
        is_trained = True
        feature_names = ["faceWidth"]

        def predict(self, x: np.ndarray) -> float:
            return 0.77

        def fit(self, X, y):
            raise NotImplementedError

    mapper.set_estimator("faceWidth", _Fixed())
    out = mapper.predict(_measurements(1.0))
    assert out["faceWidth"] == 0.77
    assert out["jawWidth"] == 0.5  # still heuristic


def test_mapper_save_load_round_trip(tmp_path: Path):
    mapper = FaceMapper.from_heuristic_bounds(_bounds())
    path = tmp_path / "face_mapper.joblib"
    mapper.save(path)
    loaded = FaceMapper.load(path)
    assert loaded.predict(_measurements(1.5)) == mapper.predict(_measurements(1.5))
