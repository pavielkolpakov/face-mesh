# observability

Structured stdout logs + timing.

- `logging.py` — `configure_logging(level)` installs a JSON handler (marked with `_OWNED_ATTR`; idempotent, won't clobber pytest's caplog). `bind_job_id` / `unbind_job_id` thread `job_id` through via `contextvars`.
- `timing.py` — `with timed("stage"): ...` emits `"<stage>.done"` log with `duration_ms`.

Rules:
- Standard events: `worker.boot`, `worker.reap.done`, `worker.shutdown.complete`, `job.completed`, `job.failed.{input,transient.requeue,transient.cap,internal}`.
- No Sentry / OTel yet. When adding, put integration behind a config flag in this module only.
