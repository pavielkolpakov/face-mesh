import numpy as np
import pytest

from face_mesh.mapping.heuristic import HeuristicEstimator


def test_heuristic_predicts_at_midpoint_returns_half():
    est = HeuristicEstimator(feature_name="faceWidth", min_value=1.0, max_value=2.0)
    assert est.predict(np.array([1.5])) == pytest.approx(0.5)


def test_heuristic_clamps_below_min_to_zero():
    est = HeuristicEstimator(feature_name="faceWidth", min_value=1.0, max_value=2.0)
    assert est.predict(np.array([0.5])) == 0.0


def test_heuristic_clamps_above_max_to_one():
    est = HeuristicEstimator(feature_name="faceWidth", min_value=1.0, max_value=2.0)
    assert est.predict(np.array([99.0])) == 1.0


def test_heuristic_is_never_trained():
    est = HeuristicEstimator(feature_name="faceWidth", min_value=1.0, max_value=2.0)
    assert est.is_trained is False


def test_heuristic_feature_names_is_single_element():
    est = HeuristicEstimator(feature_name="faceWidth", min_value=1.0, max_value=2.0)
    assert est.feature_names == ["faceWidth"]
