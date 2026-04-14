# runners

Thin wrappers around MediaPipe. Own detector lifecycle (`start` / `close` / ctx-manager). Convert MediaPipe output into `domain.types` detections.

- `face_landmarker.py` — `FaceLandmarkerRunner`. Raises `InputError("no face detected")` when result is empty; `TransientError` when model file is missing. Reads image via OpenCV (`detect_from_path`) or a pre-loaded BGR ndarray (`detect`).
- Phase 3 (todo): `pose.py` (`PoseRunner`), `segmentation.py` (`SegmentationRunner`).

Rules:
- Not thread-safe. Worker is single-concurrency — one runner instance per process.
- Never compute measurements here; hand the `FaceDetection` to an extractor.
