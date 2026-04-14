# domain

Pure data + contracts. No I/O, no third-party deps beyond numpy.

- `types.py` — dataclasses (`FaceDetection`, `FaceMeasurements`, `FaceAppearance`, `BodyViewMeasurements`, `BodyMeasurements`, `Job`) and `FACE_PARAM_NAMES` / `BODY_PARAM_NAMES` tuples (order is the JSON key order).
- `errors.py` — `PipelineError` → `InputError` (prefix `INPUT`) / `TransientError` (prefix `INTERNAL`). `public_message` is what the worker writes to `agent_jobs.error`.
- `protocols.py` — `JobQueue`, `PhotoStore`, `ResultStore` (runtime-checkable). Adapters implement these; fakes in tests implement them too.

Rules:
- Dataclasses are frozen when feasible. Add fields at the end to keep joblib / pickle compat.
- Never import from `runners`, `adapters`, `worker`.
