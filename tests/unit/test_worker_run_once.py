"""Worker.run_once behavior: happy path + every error classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from face_mesh.domain.errors import InputError, TransientError
from face_mesh.domain.types import FaceDetection, Job
from face_mesh.worker.loop import Worker

# ---- fakes -----------------------------------------------------------------


@dataclass
class FakeQueue:
    jobs: list[Job] = field(default_factory=list)
    completed: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    failed: list[tuple[str, str, bool]] = field(default_factory=list)
    reaped: int = 0

    def claim_next(self) -> Job | None:
        return self.jobs.pop(0) if self.jobs else None

    def complete(self, job_id: str, payload: dict[str, Any]) -> None:
        self.completed.append((job_id, payload))

    def fail(self, job_id: str, error: str, *, requeue: bool) -> None:
        self.failed.append((job_id, error, requeue))

    def reap_stuck(self, cutoff_minutes: int, max_retries: int) -> None:
        self.reaped += 1


@dataclass
class FakeResultStore:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def upsert_avatar(self, app_user_id: str, payload: dict[str, Any]) -> None:
        self.calls.append((app_user_id, payload))


import cv2


def _valid_image_bytes() -> bytes:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


class FakePhotoStore:
    def fetch(self, asset_id: str) -> bytes:
        return _valid_image_bytes()


class FakeFacePipeline:
    def __init__(self, raise_: Exception | None = None, payload: dict | None = None) -> None:
        self._raise = raise_
        self._payload = payload or {"version": "1.0.0", "faceParams": {"faceWidth": 0.5}}

    def process(self, detection: FaceDetection, job_id: str) -> dict:
        if self._raise is not None:
            raise self._raise
        return {**self._payload, "jobId": job_id}


class FakeRunner:
    def __init__(self, raise_: Exception | None = None) -> None:
        self._raise = raise_

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def detect(self, image_bgr: np.ndarray) -> FaceDetection:
        if self._raise:
            raise self._raise
        return FaceDetection(
            landmarks=[],
            shape_weights={},
            transform_matrix=None,
            image_bgr=image_bgr,
            image_width=10,
            image_height=10,
        )

    def detect_from_bytes(self, data: bytes) -> FaceDetection:
        return self.detect(np.zeros((10, 10, 3), dtype=np.uint8))


def _job() -> Job:
    return Job(id="j1", app_user_id="u1", retry_count=0, face_asset_id="a1")


def _worker(queue, runner, pipeline, result_store=None, max_retries: int = 3) -> Worker:
    return Worker(
        queue=queue,
        photo_store=FakePhotoStore(),
        result_store=result_store or FakeResultStore(),
        runner=runner,
        pipeline=pipeline,
        max_retries=max_retries,
    )


# ---- tests -----------------------------------------------------------------


def test_run_once_returns_false_when_no_job_available():
    w = _worker(FakeQueue(), FakeRunner(), FakeFacePipeline())
    assert w.run_once() is False


def test_run_once_happy_path_completes_and_writes_result():
    queue = FakeQueue(jobs=[_job()])
    result_store = FakeResultStore()
    w = _worker(queue, FakeRunner(), FakeFacePipeline(), result_store=result_store)

    assert w.run_once() is True
    assert len(queue.completed) == 1
    assert queue.completed[0][0] == "j1"
    assert queue.completed[0][1]["jobId"] == "j1"
    assert queue.failed == []
    assert len(result_store.calls) == 1
    assert result_store.calls[0][0] == "u1"


def test_run_once_input_error_fails_without_requeue_with_input_prefix():
    queue = FakeQueue(jobs=[_job()])
    w = _worker(queue, FakeRunner(raise_=InputError("no face detected")), FakeFacePipeline())

    assert w.run_once() is True
    assert queue.completed == []
    assert len(queue.failed) == 1
    jid, err, requeue = queue.failed[0]
    assert jid == "j1"
    assert err.startswith("INPUT:")
    assert "no face" in err
    assert requeue is False


def test_run_once_transient_under_cap_requeues():
    queue = FakeQueue(jobs=[Job(id="j1", app_user_id="u1", retry_count=1, face_asset_id="a1")])
    w = _worker(queue, FakeRunner(raise_=TransientError("supabase 503")), FakeFacePipeline(), max_retries=3)

    assert w.run_once() is True
    assert len(queue.failed) == 1
    _, err, requeue = queue.failed[0]
    assert requeue is True
    assert err.startswith("INTERNAL:")


def test_run_once_transient_over_cap_marks_failed_no_requeue():
    queue = FakeQueue(jobs=[Job(id="j1", app_user_id="u1", retry_count=3, face_asset_id="a1")])
    w = _worker(queue, FakeRunner(raise_=TransientError("x")), FakeFacePipeline(), max_retries=3)

    assert w.run_once() is True
    _, err, requeue = queue.failed[0]
    assert requeue is False
    assert "retry cap" in err.lower() or "INTERNAL" in err


def test_run_once_unexpected_exception_fails_no_requeue_internal_prefix():
    queue = FakeQueue(jobs=[_job()])
    w = _worker(queue, FakeRunner(), FakeFacePipeline(raise_=ValueError("boom")))

    assert w.run_once() is True
    _, err, requeue = queue.failed[0]
    assert requeue is False
    assert err.startswith("INTERNAL:")


def test_reap_on_boot_is_delegated_to_queue():
    queue = FakeQueue()
    w = _worker(queue, FakeRunner(), FakeFacePipeline())
    w.reap_stuck_jobs()
    assert queue.reaped == 1
