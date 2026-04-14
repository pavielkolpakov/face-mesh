"""Worker loop: single-concurrency claim → pipeline → write, SIGTERM-safe."""

from __future__ import annotations

import signal
import threading
import traceback
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from face_mesh.domain.errors import InputError, TransientError
from face_mesh.domain.protocols import JobQueue, PhotoStore, ResultStore
from face_mesh.domain.types import FaceDetection, Job
from face_mesh.observability.logging import (
    bind_job_id,
    configure_logging,
    get_logger,
    unbind_job_id,
)
from face_mesh.observability.timing import timed

log = get_logger(__name__)


class _Runner(Protocol):
    def start(self) -> None: ...
    def close(self) -> None: ...
    def detect(self, image_bgr: np.ndarray) -> FaceDetection: ...


class _Pipeline(Protocol):
    def process(self, detection: FaceDetection, job_id: str) -> dict: ...


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise InputError("cannot decode image bytes")
    return img


@dataclass
class Worker:
    queue: JobQueue
    photo_store: PhotoStore
    result_store: ResultStore
    runner: _Runner
    pipeline: _Pipeline
    max_retries: int = 3
    stuck_cutoff_minutes: int = 10

    def reap_stuck_jobs(self) -> None:
        self.queue.reap_stuck(self.stuck_cutoff_minutes, self.max_retries)

    def run_once(self) -> bool:
        """Claim + process one job. Returns True if a job was processed."""
        job = self.queue.claim_next()
        if job is None:
            return False

        token = bind_job_id(job.id)
        try:
            self._process(job)
        finally:
            unbind_job_id(token)
        return True

    def run_forever(self, stop: threading.Event, poll_interval_s: float) -> None:
        log.info("worker.boot")
        self.runner.start()
        try:
            self.reap_stuck_jobs()
            log.info("worker.reap.done")
            while not stop.is_set():
                processed = self.run_once()
                if not processed:
                    stop.wait(poll_interval_s)
        finally:
            self.runner.close()
            log.info("worker.shutdown.complete")

    # --- internals ---

    def _process(self, job: Job) -> None:
        try:
            with timed("job.fetch_photo"):
                if not job.face_asset_id:
                    raise InputError("face asset id missing on job")
                data = self.photo_store.fetch(job.face_asset_id)
            image = _decode_image(data)
            with timed("job.detect"):
                detection = self.runner.detect(image)
            with timed("job.pipeline"):
                payload = self.pipeline.process(detection, job_id=job.id)
            with timed("job.write_result"):
                self.result_store.upsert_avatar(job.app_user_id, payload)
            self.queue.complete(job.id, payload)
            log.info("job.completed", extra={"job_id": job.id})
        except InputError as e:
            log.info("job.failed.input", extra={"job_id": job.id, "error": str(e)})
            self.queue.fail(job.id, e.public_message, requeue=False)
        except TransientError as e:
            if job.retry_count < self.max_retries:
                log.warning(
                    "job.failed.transient.requeue",
                    extra={"job_id": job.id, "retry_count": job.retry_count, "error": str(e)},
                )
                self.queue.fail(job.id, e.public_message, requeue=True)
            else:
                log.error(
                    "job.failed.transient.cap",
                    extra={"job_id": job.id, "retry_count": job.retry_count, "error": str(e)},
                )
                self.queue.fail(
                    job.id,
                    f"INTERNAL: retry cap exceeded: {e}",
                    requeue=False,
                )
        except Exception as e:  # noqa: BLE001
            log.error(
                "job.failed.internal",
                extra={"job_id": job.id, "error": str(e), "traceback": traceback.format_exc()},
            )
            self.queue.fail(job.id, f"INTERNAL: {type(e).__name__}: {e}", requeue=False)


def main() -> None:
    from face_mesh.composition import build_worker_system

    configure_logging()
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    system = build_worker_system()
    system.worker.run_forever(stop, poll_interval_s=system.settings.poll_interval_s)


if __name__ == "__main__":  # pragma: no cover
    main()
