from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from face_mesh.domain.types import Job


@runtime_checkable
class JobQueue(Protocol):
    def claim_next(self) -> Job | None: ...

    def complete(self, job_id: str, payload: dict[str, Any]) -> None: ...

    def fail(self, job_id: str, error: str, *, requeue: bool) -> None: ...

    def reap_stuck(self, cutoff_minutes: int, max_retries: int) -> None: ...


@runtime_checkable
class PhotoStore(Protocol):
    def fetch(self, asset_id: str) -> bytes: ...


@runtime_checkable
class ResultStore(Protocol):
    def upsert_avatar(self, app_user_id: str, payload: dict[str, Any]) -> None: ...
