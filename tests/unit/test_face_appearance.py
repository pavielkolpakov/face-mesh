from dataclasses import dataclass

import numpy as np

from face_mesh.domain.types import FaceDetection
from face_mesh.extractors.face_appearance import FaceAppearanceExtractor


@dataclass
class _LM:
    x: float
    y: float
    z: float = 0.0


def _uniform_image(rgb: tuple[int, int, int]) -> np.ndarray:
    # OpenCV uses BGR
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:, :] = (rgb[2], rgb[1], rgb[0])
    return img


def _lm_for_appearance():
    pts = [_LM(0.5, 0.5) for _ in range(478)]
    pts[234] = _LM(0.3, 0.5)
    pts[454] = _LM(0.7, 0.5)
    pts[10] = _LM(0.5, 0.25)
    pts[152] = _LM(0.5, 0.85)
    pts[33] = _LM(0.4, 0.45)
    pts[133] = _LM(0.46, 0.45)
    pts[362] = _LM(0.54, 0.45)
    pts[263] = _LM(0.6, 0.45)
    return pts


def test_appearance_samples_skin_eye_hair_colors_normalized():
    det = FaceDetection(
        landmarks=_lm_for_appearance(),
        shape_weights={},
        transform_matrix=None,
        image_bgr=_uniform_image((200, 150, 120)),
        image_width=200,
        image_height=200,
    )
    app = FaceAppearanceExtractor().sample(det)
    # image is uniform 200,150,120 → skin/eye/hair all normalized to same triple
    np.testing.assert_array_almost_equal(app.skin_color, [200 / 255, 150 / 255, 120 / 255], decimal=2)
    np.testing.assert_array_almost_equal(app.eye_color, [200 / 255, 150 / 255, 120 / 255], decimal=2)
    np.testing.assert_array_almost_equal(app.hair_color, [200 / 255, 150 / 255, 120 / 255], decimal=2)


def test_appearance_passes_head_pose_matrix_through():
    matrix = [[1.0, 0.0, 0.0, 0.0]] * 4
    det = FaceDetection(
        landmarks=_lm_for_appearance(),
        shape_weights={},
        transform_matrix=matrix,
        image_bgr=_uniform_image((100, 100, 100)),
        image_width=200,
        image_height=200,
    )
    app = FaceAppearanceExtractor().sample(det)
    assert app.head_pose_matrix == matrix
