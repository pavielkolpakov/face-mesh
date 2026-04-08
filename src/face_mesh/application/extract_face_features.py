from pathlib import Path

import cv2

from face_mesh.application.face_geometry import compute_custom_blendshapes, point_px, sample_patch_rgb
from face_mesh.domain.entities import FaceFeaturesPayload
from face_mesh.domain.ports import FaceLandmarkPort


def extract_face_features(
    image_path: Path,
    landmarker_model_path: Path,
    face_port: FaceLandmarkPort,
) -> FaceFeaturesPayload | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    detection = face_port.detect(image, landmarker_model_path)
    if detection is None:
        return None

    landmarks = detection.landmarks
    shape_weights = detection.shape_weights
    image_height, image_width = image.shape[:2]

    skin_points = [234, 454, 10, 152]
    eye_points = [33, 133, 362, 263]
    skin_samples = [
        sample_patch_rgb(image, *point_px(landmarks, idx, image_width, image_height), radius=4)
        for idx in skin_points
    ]
    eye_samples = [
        sample_patch_rgb(image, *point_px(landmarks, idx, image_width, image_height), radius=2)
        for idx in eye_points
    ]

    forehead_x, forehead_y = point_px(landmarks, 10, image_width, image_height)
    hair_y = max(0, forehead_y - max(6, image_height // 40))
    hair_color = sample_patch_rgb(image, forehead_x, hair_y, radius=6)

    skin_color = [
        round(sum(sample[0] for sample in skin_samples) / len(skin_samples), 4),
        round(sum(sample[1] for sample in skin_samples) / len(skin_samples), 4),
        round(sum(sample[2] for sample in skin_samples) / len(skin_samples), 4),
    ]
    eye_color = [
        round(sum(sample[0] for sample in eye_samples) / len(eye_samples), 4),
        round(sum(sample[1] for sample in eye_samples) / len(eye_samples), 4),
        round(sum(sample[2] for sample in eye_samples) / len(eye_samples), 4),
    ]

    custom_blendshapes, custom_blendshapes_raw, custom_blendshape_confidence = compute_custom_blendshapes(
        landmarks=landmarks,
        shape_weights=shape_weights,
    )

    return FaceFeaturesPayload(
        shape_weights=shape_weights,
        custom_blendshapes=custom_blendshapes,
        custom_blendshapes_raw=custom_blendshapes_raw,
        custom_blendshape_confidence=custom_blendshape_confidence,
        head_pose_matrix=detection.head_pose_matrix,
        skin_color=skin_color,
        eye_color=eye_color,
        hair_color=hair_color,
    )
