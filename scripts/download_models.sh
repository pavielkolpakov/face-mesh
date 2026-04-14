#!/usr/bin/env bash
# Fetches MediaPipe model artifacts into ./models
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS_DIR="${ROOT}/models"
mkdir -p "${MODELS_DIR}"

declare -a SOURCES=(
  "face_landmarker.task|https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
  "pose_landmarker_lite.task|https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
  "selfie_segmenter.tflite|https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite"
)

for entry in "${SOURCES[@]}"; do
  name="${entry%%|*}"
  url="${entry##*|}"
  dest="${MODELS_DIR}/${name}"
  if [[ -f "${dest}" ]]; then
    echo "ok: ${name} already present"
    continue
  fi
  echo "downloading ${name} ..."
  curl -fL --retry 3 --output "${dest}" "${url}"
done

echo "done. models in ${MODELS_DIR}:"
ls -la "${MODELS_DIR}"
