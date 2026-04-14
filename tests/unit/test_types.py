import numpy as np

from face_mesh.domain.types import FaceMeasurements


def test_face_measurements_feature_vector_preserves_order():
    m = FaceMeasurements(
        raw={"faceWidth": 1.2, "jawWidth": 0.9, "chinLength": 1.4},
        confidence={"faceWidth": 0.9, "jawWidth": 0.8, "chinLength": 0.7},
        frontal_score=0.85,
    )
    vec = m.feature_vector(["jawWidth", "faceWidth", "chinLength"])
    np.testing.assert_array_almost_equal(vec, np.array([0.9, 1.2, 1.4]))


def test_feature_vector_substitutes_frontal_score_when_name_missing():
    m = FaceMeasurements(raw={"faceWidth": 1.2}, confidence={"faceWidth": 1.0}, frontal_score=0.5)
    vec = m.feature_vector(["faceWidth", "frontal_score"])
    np.testing.assert_array_almost_equal(vec, np.array([1.2, 0.5]))
