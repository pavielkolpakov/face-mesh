"""SupabaseAdapter behavior, verified against an in-memory fake client.

The fake records method calls + returns canned rows; this lets us assert
the adapter issues the expected queries (status filters, upsert shape,
initial_state-on-first-run) without touching a real Supabase instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from face_mesh.adapters.supabase import SupabaseAdapter


class _Response:
    def __init__(self, data):
        self.data = data


class _TableFake:
    def __init__(self, rows: list[dict] | None = None, name: str = ""):
        self._rows = list(rows or [])
        self._name = name
        self._filters: list[tuple[str, Any, Any]] = []
        self._order: tuple[str, bool] | None = None
        self._limit: int | None = None
        self._single = False
        self._pending_update: dict | None = None
        self._pending_upsert: dict | None = None
        self.updates: list[dict] = []
        self.upserts: list[dict] = []

    def select(self, *_cols):
        return self

    def eq(self, col, val):
        self._filters.append(("eq", col, val))
        return self

    def lt(self, col, val):
        self._filters.append(("lt", col, val))
        return self

    def order(self, col, desc=False):
        self._order = (col, desc)
        return self

    def limit(self, n):
        self._limit = n
        return self

    def single(self):
        self._single = True
        return self

    def update(self, patch):
        self._pending_update = dict(patch)
        return self

    def upsert(self, patch, on_conflict=None):
        self._pending_upsert = dict(patch)
        return self

    def _reset(self):
        self._filters = []
        self._order = None
        self._limit = None
        self._single = False

    def execute(self):
        rows = [r for r in self._rows if all(_match(r, op, c, v) for op, c, v in self._filters)]
        try:
            if self._pending_update is not None:
                patch = self._pending_update
                for r in self._rows:
                    if all(_match(r, op, c, v) for op, c, v in self._filters):
                        r.update(patch)
                self.updates.append(patch)
                self._pending_update = None
                return _Response(rows)
            if self._pending_upsert is not None:
                self.upserts.append(self._pending_upsert)
                self._rows.append(self._pending_upsert)
                self._pending_upsert = None
                return _Response([self._rows[-1]])
            if self._order:
                rows = sorted(rows, key=lambda r: r.get(self._order[0], ""), reverse=self._order[1])
            if self._limit is not None:
                rows = rows[: self._limit]
            if self._single:
                return _Response(rows[0] if rows else None)
            return _Response(rows)
        finally:
            self._reset()


def _match(row, op, col, val):
    if op == "eq":
        return row.get(col) == val
    if op == "lt":
        return (row.get(col) or "") < val
    return True


class _StorageBucket:
    def __init__(self, data: bytes = b"img"):
        self._data = data

    def download(self, path: str) -> bytes:
        return self._data


class _Storage:
    def __init__(self, data: bytes = b"img"):
        self._data = data

    def from_(self, bucket: str) -> _StorageBucket:
        return _StorageBucket(self._data)


@dataclass
class _FakeClient:
    tables: dict[str, _TableFake] = field(default_factory=dict)
    storage: _Storage = field(default_factory=_Storage)

    def table(self, name: str) -> _TableFake:
        if name not in self.tables:
            self.tables[name] = _TableFake(name=name)
        return self.tables[name]


# ---- tests -----------------------------------------------------------


def _client_with_pending_job() -> _FakeClient:
    client = _FakeClient()
    client.tables["agent_jobs"] = _TableFake(
        rows=[
            {
                "id": "job-1",
                "type": "avatar_blendshapes",
                "status": "pending",
                "app_user_id": "u1",
                "retry_count": 0,
                "created_at": "2026-04-15T00:00:00Z",
                "updated_at": "2026-04-15T00:00:00Z",
            }
        ]
    )
    client.tables["image_assets"] = _TableFake(
        rows=[
            {
                "id": "a1",
                "owner_id": "u1",
                "type": "profile",
                "is_active": True,
                "bucket": "photos",
                "path": "u1/selfie.jpg",
                "created_at": "2026-04-15T00:00:00Z",
            }
        ]
    )
    return client


def test_claim_next_returns_job_and_marks_processing():
    client = _client_with_pending_job()
    adapter = SupabaseAdapter(client)

    job = adapter.claim_next()
    assert job is not None
    assert job.id == "job-1"
    assert job.app_user_id == "u1"
    assert job.face_asset_id == "a1"

    jobs_row = client.tables["agent_jobs"]._rows[0]
    assert jobs_row["status"] == "processing"
    assert "started_at" in jobs_row


def test_claim_next_returns_none_when_no_pending():
    client = _FakeClient()
    client.tables["agent_jobs"] = _TableFake(rows=[])
    assert SupabaseAdapter(client).claim_next() is None


def test_complete_updates_status_and_completed_at():
    client = _client_with_pending_job()
    adapter = SupabaseAdapter(client)
    adapter.claim_next()
    adapter.complete("job-1", {"any": "payload"})
    row = client.tables["agent_jobs"]._rows[0]
    assert row["status"] == "completed"
    assert "completed_at" in row


def test_fail_requeue_increments_retry_count_and_resets_pending():
    client = _client_with_pending_job()
    client.tables["agent_jobs"]._rows[0]["retry_count"] = 1
    adapter = SupabaseAdapter(client)
    adapter.fail("job-1", "INTERNAL: x", requeue=True)
    row = client.tables["agent_jobs"]._rows[0]
    assert row["status"] == "pending"
    assert row["retry_count"] == 2
    assert row["error"] == "INTERNAL: x"


def test_fail_no_requeue_marks_failed():
    client = _client_with_pending_job()
    adapter = SupabaseAdapter(client)
    adapter.fail("job-1", "INPUT: bad photo", requeue=False)
    row = client.tables["agent_jobs"]._rows[0]
    assert row["status"] == "failed"
    assert row["error"] == "INPUT: bad photo"


def test_upsert_avatar_sets_initial_state_only_on_first_run():
    client = _FakeClient()
    client.tables["user_avatar_preferences"] = _TableFake(rows=[])
    adapter = SupabaseAdapter(client)

    adapter.upsert_avatar("u1", {"version": "1.0.0", "value": 1})
    first = client.tables["user_avatar_preferences"].upserts[-1]
    assert first["initial_state"]["value"] == 1
    assert first["current_state"]["value"] == 1

    # simulate row now exists with initial_state set
    client.tables["user_avatar_preferences"]._rows = [
        {"app_user_id": "u1", "initial_state": {"value": 1}, "current_state": {"value": 1}}
    ]
    adapter.upsert_avatar("u1", {"version": "1.0.0", "value": 2})
    second = client.tables["user_avatar_preferences"].upserts[-1]
    assert "initial_state" not in second
    assert second["current_state"]["value"] == 2


def test_reap_stuck_requeues_under_cap_and_fails_over_cap():
    client = _FakeClient()
    client.tables["agent_jobs"] = _TableFake(
        rows=[
            {
                "id": "stuck-a",
                "type": "avatar_blendshapes",
                "status": "processing",
                "retry_count": 1,
                "started_at": "2000-01-01T00:00:00Z",
            },
            {
                "id": "stuck-b",
                "type": "avatar_blendshapes",
                "status": "processing",
                "retry_count": 3,
                "started_at": "2000-01-01T00:00:00Z",
            },
        ]
    )
    SupabaseAdapter(client).reap_stuck(cutoff_minutes=10, max_retries=3)
    rows = {r["id"]: r for r in client.tables["agent_jobs"]._rows}
    assert rows["stuck-a"]["status"] == "pending"
    assert rows["stuck-a"]["retry_count"] == 2
    assert rows["stuck-b"]["status"] == "failed"
    assert "retry cap" in rows["stuck-b"]["error"]


def test_fetch_downloads_from_storage_by_asset_id():
    client = _FakeClient()
    client.tables["image_assets"] = _TableFake(
        rows=[{"id": "a1", "bucket": "photos", "path": "u/x.jpg"}]
    )
    client.storage = _Storage(data=b"PNG-bytes")
    assert SupabaseAdapter(client).fetch("a1") == b"PNG-bytes"
