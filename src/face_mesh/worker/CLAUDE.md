# worker

Job orchestration. Pipeline is pure; loop owns I/O + error classification.

- `pipeline.py` — `FacePipeline.process(detection, job_id)` builds the output payload (`version: "1.0.0"`, `faceParams`, `faceMeasurements`, `confidence`, `appearance`, `shapeWeights`, `headPoseMatrix`). Body fields null until Phase 3.
- `loop.py` — `Worker.run_once()` and `run_forever(stop, poll_interval_s)`. Error handling:
  - `InputError` → `queue.fail(..., requeue=False)` with `INPUT:` prefix.
  - `TransientError` under `max_retries` → `requeue=True`; over cap → `requeue=False` with `"INTERNAL: retry cap …"`.
  - Anything else → `requeue=False`, `"INTERNAL: <type>: <msg>"`; traceback to logs only.
  - SIGTERM/SIGINT sets `threading.Event`; finishes current job, then exits.
  - Boot-time `reap_stuck_jobs()` delegated to `JobQueue.reap_stuck`.

Rules:
- Worker never touches MediaPipe directly — uses the injected runner.
- Tests drive `run_once` against in-memory fakes in `tests/unit/test_worker_run_once.py`.
