"""Thin MediaPipe wrapper — owns detector lifecycle, converts results to FaceDetection."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from face_mesh.domain.errors import InputError, TransientError
from face_mesh.domain.types import FaceDetection


class FaceLandmarkerRunner:
    def __init__(self, model_path: str | Path) -> None:
        self._model_path = Path(model_path)
        self._detector: vision.FaceLandmarker | None = None

    def __enter__(self) -> FaceLandmarkerRunner:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        if self._detector is not None:
            return
        if not self._model_path.exists():
            raise TransientError(f"face landmarker model missing at {self._model_path}")
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self._model_path)),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def detect_from_path(self, image_path: str | Path) -> FaceDetection:
        img = cv2.imread(str(image_path))
        if img is None:
            raise InputError(f"cannot read image: {image_path}")
        return self.detect(img)

    def detect(self, image_bgr: np.ndarray) -> FaceDetection:
        if self._detector is None:
            self.start()
        assert self._detector is not None
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)
        if not result.face_landmarks:
            raise InputError("no face detected")

        shape_weights: dict[str, float] = {}
        if result.face_blendshapes:
            for category in result.face_blendshapes[0]:
                shape_weights[category.category_name] = round(float(category.score), 4)

        transform = None
        if result.facial_transformation_matrixes:
            transform = [
                [round(float(v), 6) for v in row] for row in result.facial_transformation_matrixes[0]
            ]

        h, w = image_bgr.shape[:2]
        return FaceDetection(
            landmarks=result.face_landmarks[0],
            shape_weights=shape_weights,
            transform_matrix=transform,
            image_bgr=image_bgr,
            image_width=w,
            image_height=h,
        )
