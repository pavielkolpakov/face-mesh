import argparse
import json
import os
from pathlib import Path

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

    payload = {
        "shapeWeights": shape_weights,
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
