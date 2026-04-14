from __future__ import annotations

import numpy as np


class HeuristicEstimator:
    """Single-feature min-max normalizer with 0..1 clamp."""

    def __init__(self, feature_name: str, min_value: float, max_value: float) -> None:
        if max_value <= min_value:
            raise ValueError(f"max_value ({max_value}) must exceed min_value ({min_value})")
        self._feature = feature_name
        self._min = float(min_value)
        self._max = float(max_value)

    @property
    def is_trained(self) -> bool:
        return False

    @property
    def feature_names(self) -> list[str]:
        return [self._feature]

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:  # noqa: N803, ARG002
        raise NotImplementedError("HeuristicEstimator does not learn")

    def predict(self, x: np.ndarray) -> float:
        value = float(x[0])
        normalized = (value - self._min) / (self._max - self._min)
        return max(0.0, min(1.0, normalized))
