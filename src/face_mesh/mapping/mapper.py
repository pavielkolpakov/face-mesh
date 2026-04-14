from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib

from face_mesh.domain.types import BODY_PARAM_NAMES, FACE_PARAM_NAMES, BodyMeasurements, FaceMeasurements
from face_mesh.mapping.estimator import Estimator
from face_mesh.mapping.heuristic import HeuristicEstimator


@dataclass
class _MapperBase:
    estimators: dict[str, Estimator]
    version: str = "1.0.0"

    def set_estimator(self, param: str, estimator: Estimator) -> None:
        self.estimators[param] = estimator

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"version": self.version, "estimators": self.estimators}, path)


class FaceMapper(_MapperBase):
    @classmethod
    def from_heuristic_bounds(cls, bounds: dict[str, tuple[float, float]]) -> FaceMapper:
        ests: dict[str, Estimator] = {
            name: HeuristicEstimator(feature_name=name, min_value=lo, max_value=hi)
            for name, (lo, hi) in bounds.items()
        }
        return cls(estimators=ests)

    @classmethod
    def load(cls, path: Path) -> FaceMapper:
        data = joblib.load(path)
        return cls(estimators=data["estimators"], version=data.get("version", "1.0.0"))

    def predict(self, measurements: FaceMeasurements) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in FACE_PARAM_NAMES:
            est = self.estimators.get(name)
            if est is None:
                continue
            vec = measurements.feature_vector(est.feature_names)
            out[name] = float(est.predict(vec))
        return out


class BodyMapper(_MapperBase):
    @classmethod
    def from_heuristic_bounds(cls, bounds: dict[str, tuple[float, float]]) -> BodyMapper:
        ests: dict[str, Estimator] = {
            name: HeuristicEstimator(feature_name=name, min_value=lo, max_value=hi)
            for name, (lo, hi) in bounds.items()
        }
        return cls(estimators=ests)

    @classmethod
    def load(cls, path: Path) -> BodyMapper:
        data = joblib.load(path)
        return cls(estimators=data["estimators"], version=data.get("version", "1.0.0"))

    def predict(self, measurements: BodyMeasurements) -> dict[str, float]:
        out: dict[str, float] = {}
        for name in BODY_PARAM_NAMES:
            est = self.estimators.get(name)
            if est is None:
                continue
            value = measurements.fused.get(name, 0.0)
            import numpy as np

            vec = np.array([measurements.fused.get(fn, value) for fn in est.feature_names])
            out[name] = float(est.predict(vec))
        return out
