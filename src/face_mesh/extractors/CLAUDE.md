# extractors

Pure functions over detection objects. No MediaPipe, no I/O.

- `face.py` — `FaceExtractor.measure(detection)` → `FaceMeasurements` (12 raw ratios + per-measurement confidence + `frontal_score`). Raises `InputError` on empty landmarks. Ports legacy `main.py` geometry. **No normalization here** — that lives in the mapper.
- `face_appearance.py` — `FaceAppearanceExtractor.sample(detection)` → `FaceAppearance` (skin/eye/hair RGB + head pose matrix).
- Phase 3 (todo): `body_view.py` (per-view pose+silhouette), `contour.py` (mask → width profile), `fusion.py` (`fuse_body_views(front, side)`).

Rules:
- Deterministic given the same detection input.
- Confidence is computed here, not in the mapper.
