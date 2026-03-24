import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import cv2
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def point_px(landmarks, index: int, image_width: int, image_height: int) -> tuple[int, int]:
    landmark = landmarks[index]
    x = int(landmark.x * image_width)
    y = int(landmark.y * image_height)
    x = max(0, min(image_width - 1, x))
    y = max(0, min(image_height - 1, y))
    return x, y


def sample_patch_rgb(image_bgr, x: int, y: int, radius: int = 3) -> list[float]:
    h, w = image_bgr.shape[:2]
    x0 = max(0, x - radius)
    y0 = max(0, y - radius)
    x1 = min(w, x + radius + 1)
    y1 = min(h, y + radius + 1)
    patch = image_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return [0.0, 0.0, 0.0]
    b, g, r = patch.reshape(-1, 3).mean(axis=0)
    return [round(float(r) / 255.0, 4), round(float(g) / 255.0, 4), round(float(b) / 255.0, 4)]


def point_xy(landmarks, index: int) -> tuple[float, float]:
    landmark = landmarks[index]
    return float(landmark.x), float(landmark.y)


def distance_xy(landmarks, idx_a: int, idx_b: int) -> float:
    ax, ay = point_xy(landmarks, idx_a)
    bx, by = point_xy(landmarks, idx_b)
    dx = ax - bx
    dy = ay - by
    return (dx * dx + dy * dy) ** 0.5


def mean_point_xy(landmarks, indices: Iterable[int]) -> tuple[float, float]:
    points = [point_xy(landmarks, i) for i in indices]
    count = len(points)
    if count == 0:
        return 0.0, 0.0
    return sum(p[0] for p in points) / count, sum(p[1] for p in points) / count


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_range(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 0.0
    return clamp01((value - min_value) / (max_value - min_value))


def compute_custom_blendshapes(landmarks, shape_weights: dict[str, float]) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    left_eye_center = mean_point_xy(landmarks, [33, 133, 159, 145, 160, 144])
    right_eye_center = mean_point_xy(landmarks, [263, 362, 386, 374, 387, 373])
    eye_center_dx = left_eye_center[0] - right_eye_center[0]
    eye_center_dy = left_eye_center[1] - right_eye_center[1]
    eye_scale = (eye_center_dx * eye_center_dx + eye_center_dy * eye_center_dy) ** 0.5
    fallback_scale = distance_xy(landmarks, 234, 454)
    scale = eye_scale if eye_scale > 1e-6 else max(fallback_scale, 1e-6)

    face_width_raw = distance_xy(landmarks, 234, 454) / scale
    jaw_width_raw = distance_xy(landmarks, 172, 397) / scale
    chin_length_raw = distance_xy(landmarks, 152, 17) / scale

    left_eye_aperture = distance_xy(landmarks, 159, 145)
    right_eye_aperture = distance_xy(landmarks, 386, 374)
    left_eye_width = max(distance_xy(landmarks, 33, 133), 1e-6)
    right_eye_width = max(distance_xy(landmarks, 263, 362), 1e-6)
    eye_size_raw = 0.5 * ((left_eye_aperture / left_eye_width) + (right_eye_aperture / right_eye_width))
    eye_spacing_raw = distance_xy(landmarks, 133, 362) / scale

    nose_width = distance_xy(landmarks, 98, 327)
    nose_length = distance_xy(landmarks, 6, 4)
    nose_size_raw = (nose_width + nose_length) / (2.0 * scale)
    nose_bridge_height_raw = distance_xy(landmarks, 168, 6) / scale
    nose_tip_size_raw = nose_width / scale

    lip_fullness_raw = distance_xy(landmarks, 13, 14) / scale
    mouth_width_raw = distance_xy(landmarks, 78, 308) / scale
    brow_height_raw = 0.5 * (
        (distance_xy(landmarks, 105, 159) / scale) +
        (distance_xy(landmarks, 334, 386) / scale)
    )

    _, eye_y = mean_point_xy(landmarks, [159, 145, 386, 374])
    _, chin_y = point_xy(landmarks, 152)
    _, cheek_y_left = point_xy(landmarks, 116)
    _, cheek_y_right = point_xy(landmarks, 345)
    cheek_y = 0.5 * (cheek_y_left + cheek_y_right)
    denom = max(chin_y - eye_y, 1e-6)
    cheekbone_height_raw = 1.0 - ((cheek_y - eye_y) / denom)

    blink = max(shape_weights.get("eyeBlinkLeft", 0.0), shape_weights.get("eyeBlinkRight", 0.0))
    mouth_open = shape_weights.get("jawOpen", 0.0)
    lip_motion = max(
        shape_weights.get("mouthSmileLeft", 0.0),
        shape_weights.get("mouthSmileRight", 0.0),
        shape_weights.get("mouthPucker", 0.0),
        shape_weights.get("mouthFunnel", 0.0),
    )
    expression_penalty = clamp01(max(blink, mouth_open, lip_motion))

    nose_x, _ = point_xy(landmarks, 1)
    mid_eye_x = 0.5 * (left_eye_center[0] + right_eye_center[0])
    center_offset = abs(nose_x - mid_eye_x) / max(scale, 1e-6)
    frontal_score = 1.0 - normalize_range(center_offset, 0.08, 0.22)
    base_conf = clamp01(0.35 + 0.65 * frontal_score) * clamp01(1.0 - 0.5 * expression_penalty)

    raw = {
        "faceWidth": round(face_width_raw, 6),
        "jawWidth": round(jaw_width_raw, 6),
        "chinLength": round(chin_length_raw, 6),
        "cheekboneHeight": round(cheekbone_height_raw, 6),
        "eyeSize": round(eye_size_raw, 6),
        "eyeSpacing": round(eye_spacing_raw, 6),
        "noseSize": round(nose_size_raw, 6),
        "noseBridgeHeight": round(nose_bridge_height_raw, 6),
        "noseTipSize": round(nose_tip_size_raw, 6),
        "lipFullness": round(lip_fullness_raw, 6),
        "mouthWidth": round(mouth_width_raw, 6),
        "browHeight": round(brow_height_raw, 6),
    }

    normalized = {
        "faceWidth": round(normalize_range(face_width_raw, 3.1, 4.7), 4),
        "jawWidth": round(normalize_range(jaw_width_raw, 2.4, 4.1), 4),
        "chinLength": round(normalize_range(chin_length_raw, 1.0, 2.0), 4),
        "cheekboneHeight": round(clamp01(cheekbone_height_raw), 4),
        "eyeSize": round(normalize_range(eye_size_raw, 0.18, 0.40), 4),
        "eyeSpacing": round(normalize_range(eye_spacing_raw, 0.45, 0.90), 4),
        "noseSize": round(normalize_range(nose_size_raw, 0.55, 1.10), 4),
        "noseBridgeHeight": round(normalize_range(nose_bridge_height_raw, 0.20, 0.65), 4),
        "noseTipSize": round(normalize_range(nose_tip_size_raw, 0.35, 0.95), 4),
        "lipFullness": round(normalize_range(lip_fullness_raw, 0.08, 0.34), 4),
        "mouthWidth": round(normalize_range(mouth_width_raw, 1.0, 2.1), 4),
        "browHeight": round(normalize_range(brow_height_raw, 0.12, 0.52), 4),
    }

    confidence = {
        "faceWidth": round(base_conf, 4),
        "jawWidth": round(base_conf, 4),
        "chinLength": round(base_conf * 0.85, 4),
        "cheekboneHeight": round(base_conf * 0.75, 4),
        "eyeSize": round(base_conf * clamp01(1.0 - blink), 4),
        "eyeSpacing": round(base_conf, 4),
        "noseSize": round(base_conf * 0.9, 4),
        "noseBridgeHeight": round(base_conf * 0.7, 4),
        "noseTipSize": round(base_conf * 0.8, 4),
        "lipFullness": round(base_conf * clamp01(1.0 - max(mouth_open, lip_motion)), 4),
        "mouthWidth": round(base_conf * clamp01(1.0 - 0.5 * max(mouth_open, lip_motion)), 4),
        "browHeight": round(base_conf * 0.9, 4),
    }
    return normalized, raw, confidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Unity-relevant face params from one image."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default="input.jpeg",
        help="Path to input image file.",
    )
    parser.add_argument(
        "--output",
        default="landmarks.json",
        help="Path to output JSON file.",
    )
    parser.add_argument(
        "--model",
        default="face_landmarker.task",
        help="Path to MediaPipe face_landmarker.task model file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Missing model file: {args.model}")
        print("Provide MediaPipe face_landmarker.task via --model.")
        return

    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Failed to load image: {args.image_path}")
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_height, image_width = image.shape[:2]
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
        print("No face detected.")
        return
    landmarks = results.face_landmarks[0]
    shape_weights = {}
    if results.face_blendshapes:
        for category in results.face_blendshapes[0]:
            shape_weights[category.category_name] = round(float(category.score), 4)

    skin_points = [234, 454, 10, 152]
    eye_points = [33, 133, 362, 263]
    skin_samples = [sample_patch_rgb(image, *point_px(landmarks, idx, image_width, image_height), radius=4) for idx in skin_points]
    eye_samples = [sample_patch_rgb(image, *point_px(landmarks, idx, image_width, image_height), radius=2) for idx in eye_points]

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
    head_pose_matrix = None
    if results.facial_transformation_matrixes:
        head_pose_matrix = [
            [round(float(value), 6) for value in row]
            for row in results.facial_transformation_matrixes[0]
        ]
    custom_blendshapes, custom_blendshapes_raw, custom_blendshape_confidence = compute_custom_blendshapes(
        landmarks=landmarks,
        shape_weights=shape_weights,
    )

    payload = {
        "shapeWeights": shape_weights,
        "customBlendshapes": custom_blendshapes,
        "customBlendshapesRaw": custom_blendshapes_raw,
        "customBlendshapeConfidence": custom_blendshape_confidence,
        "headPoseMatrix": head_pose_matrix,
        "skinColor": skin_color,
        "eyeColor": eye_color,
        "hairColor": hair_color,
    }
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
