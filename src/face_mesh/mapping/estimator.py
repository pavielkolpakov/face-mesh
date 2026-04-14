from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Estimator(Protocol):
    @property
    def is_trained(self) -> bool: ...

    @property
    def feature_names(self) -> list[str]: ...

    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...  # noqa: N803

    def predict(self, x: np.ndarray) -> float: ...
