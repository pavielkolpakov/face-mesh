# Face Mesh — Photo to 3D Avatar Blendshapes

## Overview
Python worker service that turns user photos into Unity blendshape parameters (12 face + 10 body) plus raw measurements, confidence, and appearance colors. Runs as a long-lived Supabase-connected worker for an Expo mobile app. A designer prepares the Unity model with custom blendshapes; this service bridges photo measurements → those abstract blendshapes via a calibratable regression layer that falls back to heuristic normalization when no calibration data exists.

## MVP Features
- Face blendshape extraction from 1 photo (12 params) via MediaPipe Face Landmarker
- Body blendshape extraction from 2 photos — front + side (10 params) via MediaPipe Pose + MediaPipe ImageSegmenter silhouette
- Heuristic fallback mapping (works with zero training data)
- Calibration system: designer provides photo + expected-values pairs, Ridge regression learns the mapping per-param
- Supabase worker: claims jobs from `agent_jobs`, fetches photos from Storage via `image_assets`, writes results to `user_avatar_preferences`
- CLI for development, extraction, and calibration workflows
- Full output payload: faceParams, bodyParams, raw measurements, per-param confidence, appearance colors, ARKit shapeWeights, head pose matrix

## Out of Scope (v1)
- Video or multi-angle input beyond front + side
- Real-time / streaming processing
- 3D face reconstruction (DECA/FLAME/SMPL)
- Clothing segmentation
- Hair/accessory detection
- Auto-retraining pipeline (manual retrain via CLI only)
- Concurrent job processing (single-concurrency worker; architecture leaves room for a pool later)
- Live Supabase integration tests in CI

## Tech Stack
| Layer | Choice | Reason |
|---|---|---|
| Face landmarks | MediaPipe Face Landmarker | 478 points + ARKit blendshapes + transform matrix |
| Body landmarks | MediaPipe Pose Landmarker | 33 landmarks, CPU-only |
| Body segmentation | MediaPipe ImageSegmenter (selfie / general) | Clean silhouette without clean-background requirement |
| Regression | scikit-learn Ridge | Tiny dataset (5–50 samples), regularized, retrains in ms |
| Persistence | joblib | Mapper artifacts (`models/face_mapper.joblib`, `models/body_mapper.joblib`) |
| Queue | Postgres `SELECT … FOR UPDATE SKIP LOCKED` | Atomic claim, concurrency-safe |
| Storage | Supabase Storage (via `image_assets` table) | Mobile app uploads, worker downloads |
| DB | Supabase Postgres | `agent_jobs` (queue) + `user_avatar_preferences` (results) + `image_assets` (photo registry) |
| Config | pydantic-settings + YAML | Layered defaults + env + `.env` dev overlay |
| CLI | typer | Subcommands, auto-help |
| Logging | stdlib `logging` + JSON formatter + contextvars | Structured stdout, job_id threaded everywhere |
| Tooling | uv + pyproject.toml + ruff + mypy | Fast deps, strict typing on pure modules |
| Language | Python 3.11+ | MediaPipe/OpenCV/sklearn ecosystem |

## Data flow

```
Mobile App (Expo)
  │ upload photos → Supabase Storage, register in image_assets,
  │ insert agent_jobs row (type='avatar_blendshapes', status='pending')
  ▼
Supabase (Postgres + Storage)
  │ worker polls every 2–5s with FOR UPDATE SKIP LOCKED
  ▼
Worker (persistent singleton runners, single concurrency)
  │
  ├─ FaceLandmarkerRunner.detect(face_image)           → FaceDetection
  │     ├─ FaceExtractor.measure(detection)           → FaceMeasurements + shapeWeights + confidence + frontal_score
  │     └─ FaceAppearanceExtractor.sample(detection)  → skinColor, eyeColor, hairColor, headPoseMatrix
  │
  ├─ PoseRunner.detect(front|side)  ─┐
  ├─ SegmentationRunner.segment(…)  ─┼→ BodyViewExtractor(front) → BodyViewMeasurements
  │                                   └→ BodyViewExtractor(side)  → BodyViewMeasurements
  │     └─ fuse_body_views(front, side) → BodyMeasurements + confidence
  │
  ├─ FaceMapper.predict(measurements)   → faceParams (12, 0..1)
  ├─ BodyMapper.predict(measurements)   → bodyParams (10, 0..1)
  │       (each param: RegressionEstimator if trained, else HeuristicEstimator)
  │
  └─ ResultStore.upsert(app_user_id, payload)
        → user_avatar_preferences.current_state = payload
        → user_avatar_preferences.initial_state = payload (only if empty)
```

## Folder structure

```
face-mesh/
  pyproject.toml
  uv.lock
  Dockerfile
  scripts/
    download_models.sh            # fetches face_landmarker.task, pose_landmarker.task, selfie_segmenter.tflite
  models/                         # gitignored, fetched locally / baked in image
  config/
    default.yaml                  # model paths, poll interval, retry cap, log level
    heuristic_bounds.yaml         # per-param (min, max) for 12 face + 10 body
  src/face_mesh/
    domain/
      types.py                    # FaceDetection, BodyDetection, FaceMeasurements, BodyMeasurements, FaceParams, BodyParams, Confidence, Payload
      protocols.py                # JobQueue, PhotoStore, ResultStore, Estimator
      errors.py                   # TransientError, InputError (InternalError = plain Exception)
    runners/
      face_landmarker.py          # FaceLandmarkerRunner (owns MediaPipe lifecycle)
      pose.py                     # PoseRunner
      segmentation.py             # SegmentationRunner
    extractors/
      face.py                     # FaceExtractor: detection → raw measurements + confidence + frontal_score
      face_appearance.py          # FaceAppearanceExtractor: skin/eye/hair colors + head pose matrix
      body_view.py                # BodyViewExtractor: pose + silhouette → BodyViewMeasurements (per view)
      contour.py                  # silhouette mask → largest component → width profile at landmark Y's
      fusion.py                   # fuse_body_views(front, side) → BodyMeasurements with confidence weighting
    mapping/
      estimator.py                # Estimator protocol (fit, predict, is_trained, feature_names)
      heuristic.py                # HeuristicEstimator(min, max) with normalize_range
      regression.py               # RegressionEstimator wrapping sklearn Ridge
      mapper.py                   # FaceMapper, BodyMapper: dict[param, Estimator], routing, save/load joblib
    calibration/
      dataset.py                  # load/save calibration cases (case JSON + photos)
      train.py                    # build X, y per param; fit Ridge; persist mapper
      evaluate.py                 # LeaveOneOut CV, MAE per param
    adapters/
      supabase.py                 # SupabaseAdapter implements JobQueue + PhotoStore + ResultStore
    worker/
      loop.py                     # main loop, SIGTERM handler, boot-time reaper
      pipeline.py                 # orchestrates extract → map → payload build (pure of IO)
    cli/
      main.py                     # typer app root
      commands/
        extract.py                # face, body, all
        calibrate.py              # calibrate add | train | evaluate
        worker.py                 # worker alias
    config/
      settings.py                 # pydantic-settings Settings, loads YAML + env + .env
    observability/
      logging.py                  # JSON logging config, contextvars job_id binding
      timing.py                   # @timed / timed(...) context manager emitting duration_ms
    composition.py                # wiring root — instantiates runners, extractors, mappers, adapter, worker
  calibration/
    cases/                        # case_001.json, …
    photos/                       # calibration photos
  tests/
    unit/                         # synthetic fixtures, no MediaPipe, no Supabase
    golden/                       # real photos → expected JSON, with --regen-golden hook
```

## Domain contracts (summary)

- **`FaceDetection`**: landmarks (478), shapeWeights (ARKit 52), facial_transformation_matrix, image_bgr, image_size.
- **`FaceMeasurements`**: 12 raw ratios (faceWidth, jawWidth, chinLength, cheekboneHeight, eyeSize, eyeSpacing, noseSize, noseBridgeHeight, noseTipSize, lipFullness, mouthWidth, browHeight) + `frontal_score` + per-measurement confidence.
- **`BodyViewMeasurements`**: view tag (`front`/`side`), pose-derived widths/lengths, silhouette-derived width profile at shoulder/chest/waist/hip Y's, per-measurement confidence.
- **`BodyMeasurements`**: fused output (shoulderWidth, chestWidth, chestDepth, waistWidth, waistDepth, hipWidth, hipDepth, armThickness, legThickness, torsoLength, legLength, neckLength, bellyDepth, height) normalized by pose height + fused per-measurement confidence.
- **`FaceParams` / `BodyParams`**: 12 + 10 values clamped to 0..1.
- **`Confidence`**: `{face: per-param dict, body: per-param dict, overall: float}`.
- **`Payload`** (output jsonb, `version: "1.0.0"`): see §Output payload below.

## Protocols (in `domain/protocols.py`)

```python
class JobQueue(Protocol):
    def claim_next(self) -> Job | None: ...
    def complete(self, job_id: UUID, payload: dict) -> None: ...
    def fail(self, job_id: UUID, error: str, requeue: bool) -> None: ...
    def reap_stuck(self, cutoff_minutes: int, max_retries: int) -> None: ...

class PhotoStore(Protocol):
    def fetch(self, asset_id: UUID) -> bytes: ...  # returns raw image bytes

class ResultStore(Protocol):
    def upsert_avatar(self, app_user_id: UUID, payload: dict) -> None: ...

class Estimator(Protocol):
    @property
    def is_trained(self) -> bool: ...
    @property
    def feature_names(self) -> list[str]: ...
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict(self, x: np.ndarray) -> float: ...
```

`SupabaseAdapter` implements `JobQueue + PhotoStore + ResultStore`. Tests use in-memory fakes satisfying the same protocols.

## Job I/O (hardcoded for v1, to be cleaned up later)

- Input: worker picks `image_assets` rows by `app_user_id` + a convention (latest-per-role, exact linkage TBD). Photo bytes downloaded from Supabase Storage using the asset's `bucket`/`path`.
- Output: upsert on `user_avatar_preferences` by `app_user_id`.
  - `current_state = payload` always.
  - `initial_state = payload` only if currently empty (`'{}'::jsonb` or null).

## Queue semantics

- `type = 'avatar_blendshapes'` distinguishes from other agent_job types.
- Claim query (atomic, concurrency-safe):
  ```sql
  WITH next AS (
    SELECT id FROM agent_jobs
    WHERE status='pending' AND type='avatar_blendshapes'
    ORDER BY created_at
    FOR UPDATE SKIP LOCKED LIMIT 1
  )
  UPDATE agent_jobs
    SET status='processing', started_at=now(), updated_at=now()
  FROM next WHERE agent_jobs.id = next.id
  RETURNING agent_jobs.*;
  ```
- Poll interval: 2–5s (configurable via `settings.poll_interval_s`).
- On success: `status='completed'`, `completed_at=now()`.
- On failure: see §Error handling.

## Error handling

Exceptions in `domain/errors.py`:

- **`TransientError`** — network blip, Supabase 5xx, MediaPipe init glitch. Worker increments `retry_count`; if `< max_retries` (default 3), set `status='pending'` (requeue), else `status='failed'` with `error='INTERNAL: retry cap exceeded: <msg>'`.
- **`InputError`** — no face/body detected, unreadable image, confidence below floor. Immediate `status='failed'`, `error='INPUT: <detail>'`. No retry. Client-facing (mobile can show "retake photo").
- **Anything else** — `status='failed'`, `error='INTERNAL: <type>: <msg>'` (no traceback in DB). Full traceback to stdout logs.

Extractors raise `InputError` when detection fails or confidence floor is not met. Adapters raise `TransientError` on Supabase/network failures.

## Worker loop

- Persistent singleton `FaceLandmarkerRunner`, `PoseRunner`, `SegmentationRunner` (each owns a MediaPipe detector; not thread-safe, one job at a time).
- On boot: reaper sweeps stuck jobs
  ```sql
  UPDATE agent_jobs
    SET status=CASE WHEN retry_count < 3 THEN 'pending' ELSE 'failed' END,
        retry_count = retry_count + 1,
        error = CASE WHEN retry_count >= 3 THEN 'INTERNAL: stuck job, exceeded retry cap' ELSE error END,
        updated_at = now()
  WHERE type='avatar_blendshapes'
    AND status='processing'
    AND started_at < now() - interval '10 minutes';
  ```
- SIGTERM/SIGINT → `threading.Event`; loop finishes the current job, then exits cleanly.
- `contextvars` binds `job_id` to every log line within a job.

## Mapper

- **`Estimator` protocol** with two implementations: `HeuristicEstimator(min, max)` and `RegressionEstimator(Ridge, feature_names)`.
- **`FaceMapper` / `BodyMapper`**: hold a `dict[param_name, Estimator]`. `predict` iterates params, routes to regression if `is_trained`, else heuristic.
- **Features for regression (option A)**: every param consumes the full measurement vector of its domain + `frontal_score` (face) / overall body confidence (body). Ridge regularization handles the wide-vs-shallow dataset. `feature_names` frozen per mapper artifact for ordering stability.
- **Heuristic bounds** live in `config/heuristic_bounds.yaml` (22 pairs). Ports the `normalize_range` logic from the current `main.py` for the 12 face params; body bounds added in Phase 3.
- **Persistence**: `models/face_mapper.joblib`, `models/body_mapper.joblib`. Each artifact stores the full `dict[param, Estimator]` (both heuristic and regression) plus a version tag.

## Calibration

- Case format (`calibration/cases/case_<id>.json`) unchanged from original PLAN:
  ```json
  {
    "id": "case_001",
    "facePhoto": "calibration/photos/case_001_face.jpg",
    "bodyFrontPhoto": "calibration/photos/case_001_front.jpg",
    "bodySidePhoto": "calibration/photos/case_001_side.jpg",
    "expected": { "faceParams": {...}, "bodyParams": {...} }
  }
  ```
- `calibrate add` writes the case JSON and copies photos into `calibration/photos/`.
- `calibrate train` iterates cases, runs extractors to get measurement vectors `X`, reads `expected.*` as `y`, fits Ridge per param, persists mapper.
- `calibrate evaluate` runs `LeaveOneOut` CV per param and prints an MAE table.

## Output payload

Written to `user_avatar_preferences.current_state` (and `initial_state` on first run):

```json
{
  "version": "1.0.0",
  "jobId": "<uuid>",
  "generatedAt": "2026-04-15T12:34:56Z",
  "faceParams":   { "faceWidth": 0.52, "jawWidth": 0.44, "...": 10 more },
  "bodyParams":   { "bodyFat": 0.34, "muscleMass": 0.56, "...": 8 more },
  "faceMeasurements": { /* raw ratios, pre-mapping */ },
  "bodyMeasurements": {
    "front": { /* raw per-view */ },
    "side":  { /* raw per-view */ },
    "fused": { /* fused */ }
  },
  "confidence": {
    "face":    { /* per-param 0..1 */ },
    "body":    { /* per-param 0..1 */ },
    "overall": 0.87
  },
  "appearance": { "skinColor": [r,g,b], "eyeColor": [r,g,b], "hairColor": [r,g,b] },
  "shapeWeights":   { /* ARKit 52 */ },
  "headPoseMatrix": [[...], [...], [...], [...]]
}
```

## Configuration

Layered via `pydantic-settings` `Settings` (frozen):
1. `config/default.yaml` — model paths, poll interval, retry cap, segmentation threshold, log level defaults.
2. `config/heuristic_bounds.yaml` — per-param `(min, max)`.
3. Environment variables — `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` (as `SecretStr`), `LOG_LEVEL`, `CONFIG_PROFILE`.
4. `.env` via `python-dotenv` in dev only (gated on non-`PRODUCTION`).

Worker uses the **service-role key** (bypasses RLS to read any user's photos and upsert their avatar prefs). Key never logged, never serialized.

## Observability

- Stdlib `logging` + JSON formatter → stdout (captured by container runtime).
- `contextvars` binds `job_id` to every log line inside a job.
- `timed("stage")` context manager emits `{event: "<stage>.done", duration_ms: N}` on exit. Used around face.detect, face.extract, body.pose, body.segment, body.extract, body.fuse, map.face, map.body, result.write.
- Standard events: `worker.boot`, `worker.reap.done`, `worker.shutdown.requested`, `worker.shutdown.complete`, `job.claimed`, `job.completed`, `job.failed` (with classification + error message, never traceback).
- No Sentry/OTel in v1; `observability/` isolates the later addition.

## Testing

- **Unit (`tests/unit/`)**: synthetic detection fixtures (hand-crafted landmark arrays) → extractor → assert measurements; fake `Estimator` → mapper → assert routing; fusion with front-only / side-only / both; in-memory `JobQueue`/`PhotoStore`/`ResultStore` fakes → worker happy/error paths. No MediaPipe, no Supabase. Runs <1s.
- **Golden (`tests/golden/`)**: existing `fat_face.png`, `fit_face.png`, `skinny_face.png` (plus sample body photos added in Phase 3) → expected JSON committed alongside; tolerance `1e-3`; skipped if `face_landmarker.task` not present locally; always run in CI. `pytest --regen-golden` regenerates.
- **Adapter contract**: same protocol tests run against `InMemoryAdapter` and (manually, in smoke script) `SupabaseAdapter`. No live Supabase in CI.
- **Tooling**: pytest, ruff (lint + format), mypy `--strict` scoped to `domain/`, `mapping/`, `extractors/fusion.py`, `extractors/contour.py`.

## Packaging & deployment

- `uv` + `pyproject.toml`, src layout, Python 3.11+.
- Entry points:
  - `face-mesh = face_mesh.cli.main:app`
  - `face-mesh-worker = face_mesh.worker.loop:main`
- `uv sync` / `uv lock` for reproducibility.
- `scripts/download_models.sh` fetches MediaPipe model files into `models/` (gitignored).
- Dockerfile (slim Python 3.11 base): `uv sync --frozen`, `COPY models/`, `CMD ["face-mesh-worker"]`.

## CLI commands (typer)

- `face-mesh face <image>` — full face pipeline, prints JSON.
- `face-mesh body <front> <side>` — full body pipeline, prints JSON.
- `face-mesh all <face> <front> <side>` — full payload matching the output contract.
- `face-mesh calibrate add --id <case_id> --face <img> --front <img> --side <img> --expected <json>` — writes case JSON + copies photos.
- `face-mesh calibrate train [--face] [--body]` — retrains selected mapper(s), persists joblib.
- `face-mesh calibrate evaluate` — LOO-CV per param, prints MAE table.
- `face-mesh worker` — alias for `face-mesh-worker` (dev convenience).

`face`, `body`, `all`, and `worker` all go through the same `pipeline.py` (swap `PhotoStore` for a local-file implementation, swap `ResultStore` for a stdout implementation). `calibrate *` are purpose-built against `calibration/`.

## Implementation phases

### Phase 0 — Skeleton
- `pyproject.toml` + `uv.lock` + src layout + Python 3.11 pin.
- `domain/` types, protocols, errors.
- `config/` package + `default.yaml` + `heuristic_bounds.yaml` (face bounds ported now; body bounds placeholder).
- `observability/` logging + timing.
- Stub modules in `runners/`, `extractors/`, `mapping/`, `adapters/`, `worker/`, `cli/`.
- `composition.py` wiring stub.
- Tests infra: pytest config, ruff + mypy configs, first smoke unit test.
- `scripts/download_models.sh` + `.gitignore` for `models/`.
- Dockerfile skeleton.
- CI config (lint + typecheck + unit + golden-guarded).

### Phase 1 — Face vertical slice
- `FaceLandmarkerRunner` (owns MediaPipe lifecycle, warm-up, close).
- `FaceExtractor` — ports current `main.py` geometric logic into a pure function over `FaceDetection`, returns raw measurements + per-measurement confidence + frontal_score. No normalization inside.
- `FaceAppearanceExtractor` — skin/eye/hair colors + head pose matrix.
- `HeuristicEstimator` + `FaceMapper` with YAML-driven bounds.
- `pipeline.py` face path + payload builder.
- CLI `face` command (local file in, JSON out).
- Unit tests on pure extractor + mapper routing.
- Golden tests on `fat_face.png`, `fit_face.png`, `skinny_face.png`.

### Phase 2 — Worker end-to-end (face-only)
- `SupabaseAdapter` — `JobQueue` with atomic claim + SKIP LOCKED, `PhotoStore` (Storage download), `ResultStore` (upsert `user_avatar_preferences`, initial_state-on-first-run).
- `worker/loop.py` — main loop, SIGTERM-safe `threading.Event`, boot-time reaper, structured logs with job_id contextvar.
- CLI `worker` command + `face-mesh-worker` entry point.
- Body fields in payload are null/absent for now.
- **First shippable milestone.**
- Smoke script exercising the real adapter manually; in-memory fakes for CI.

### Phase 3 — Body vertical
- `PoseRunner`, `SegmentationRunner`.
- `contour.py` — mask → largest component → width profile at pose-landmark Y's.
- `BodyViewExtractor` per view → `BodyViewMeasurements`.
- `fuse_body_views` — front/side fusion with confidence-weighted combination.
- `BodyMapper` heuristic path + body bounds in `heuristic_bounds.yaml`.
- Payload now includes `bodyParams`, `bodyMeasurements`, body confidence.
- CLI `body`, `all` commands.
- Golden tests for body (add sample photos).

### Phase 4 — Calibration
- `calibration/dataset.py` — case loader/saver.
- `calibration/train.py` — build `X`, `y` per param, fit Ridge, persist mapper.
- `RegressionEstimator` + mapper routing (regression when trained, else heuristic).
- `calibration/evaluate.py` — LeaveOneOut CV, MAE per param.
- CLI `calibrate add | train | evaluate`.

### Phase 5 — Polish
- Input validation: resolution floors, face-detected precheck, body-visible precheck, orientation sanity — all surfacing `InputError` with actionable messages.
- Confidence-based warnings inside payload (e.g. `warnings: ["low_frontal_score", "occluded_shoulder"]`).
- README: setup, calibration workflow, deployment, troubleshooting.
- Dockerfile hardening + multi-stage build.
- Deployment docs (env vars, service-role key rotation, log aggregation).

## Resolved questions (from original PLAN)
- **Minimum calibration cases**: 5 bare-minimum, 20+ recommended. Mapper auto-falls-back to heuristic below a per-param threshold.
- **Re-processing jobs**: yes — re-queue by inserting a new `agent_jobs` row; upsert on `user_avatar_preferences` overwrites `current_state` while preserving `initial_state`.
- **Photo requirements**: documented in README; frontal face, neutral expression, full body visible front + side, any background (segmentation handles it).
- **Silhouette background**: MediaPipe ImageSegmenter — no clean-background requirement for users.
- **Supabase project & tables**: project exists; `agent_jobs`, `image_assets`, `user_avatar_preferences` already present. Job-to-image linkage hardcoded in v1, proper schema deferred.
