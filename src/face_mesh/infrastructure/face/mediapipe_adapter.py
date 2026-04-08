import os
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from face_mesh.domain.entities import FaceDetectionResult
from face_mesh.domain.ports import FaceLandmarkPort


class MediaPipeFaceLandmarkAdapter(FaceLandmarkPort):
    def detect(self, image_bgr, model_path: Path) -> FaceDetectionResult | None:
        rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        detector = vision.FaceLandmarker.create_from_options(options)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        results = detector.detect(mp_image)
        detector.close()

        if not results.face_landmarks:
            return None
        landmarks = results.face_landmarks[0]
        shape_weights: dict[str, float] = {}
        if results.face_blendshapes:
            for category in results.face_blendshapes[0]:
                shape_weights[category.category_name] = round(float(category.score), 4)

        head_pose_matrix = None
        if results.facial_transformation_matrixes:
            head_pose_matrix = [
                [round(float(value), 6) for value in row]
                for row in results.facial_transformation_matrixes[0]
            ]
        return FaceDetectionResult(
            landmarks=landmarks,
            shape_weights=shape_weights,
            head_pose_matrix=head_pose_matrix,
        )
