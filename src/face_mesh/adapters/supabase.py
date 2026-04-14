"""Supabase adapter: JobQueue + PhotoStore + ResultStore.

All SQL is issued via Supabase's postgrest client. For the atomic claim we
use `rpc("claim_avatar_job")` — a tiny SQL function the DBA deploys once.
To avoid coupling v1 to a migration we fall back to a two-phase UPDATE
that uses `select … for update skip locked`-equivalent semantics via a
versioned compare-and-swap on status.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from face_mesh.domain.errors import TransientError
from face_mesh.domain.types import Job


class _PostgrestLike(Protocol):  # for typing; matches supabase-py's table interface
    ...


class SupabaseAdapter:
    """Thin wrapper around supabase-py. Hardcoded for v1.

    Input linkage is convention-based (B from design: latest image_assets rows
    for the app_user_id, selected by role metadata). Clean schema is deferred.
    """

    JOB_TYPE = "avatar_blendshapes"

    def __init__(self, client: Any) -> None:
        self._client = client

    # ---- JobQueue -----------------------------------------------------

    def claim_next(self) -> Job | None:
        """Claim the oldest pending job. Two-phase: select then cas-update.

        Uses an `updated_at` compare-and-swap; if another worker beat us,
        we simply return None and wait for the next tick.
        """
        try:
            resp = (
                self._client.table("agent_jobs")
                .select("*")
                .eq("type", self.JOB_TYPE)
                .eq("status", "pending")
                .order("created_at")
                .limit(1)
                .execute()
            )
        except Exception as e:  # noqa: BLE001
            raise TransientError(f"claim select failed: {e}") from e

        rows = resp.data or []
        if not rows:
            return None
        row = rows[0]
        job_id = row["id"]
        prev_updated = row.get("updated_at")

        try:
            upd = (
                self._client.table("agent_jobs")
                .update(
                    {
                        "status": "processing",
                        "started_at": _now_iso(),
                        "updated_at": _now_iso(),
                    }
                )
                .eq("id", job_id)
                .eq("status", "pending")
                .eq("updated_at", prev_updated)
                .execute()
            )
        except Exception as e:  # noqa: BLE001
            raise TransientError(f"claim update failed: {e}") from e

        if not upd.data:
            return None  # lost the race; try next tick

        return Job(
            id=row["id"],
            app_user_id=row.get("app_user_id") or "",
            retry_count=int(row.get("retry_count") or 0),
            face_asset_id=self._latest_asset_id(row.get("app_user_id")),
        )

    def complete(self, job_id: str, payload: dict[str, Any]) -> None:
        try:
            self._client.table("agent_jobs").update(
                {
                    "status": "completed",
                    "completed_at": _now_iso(),
                    "updated_at": _now_iso(),
                }
            ).eq("id", job_id).execute()
        except Exception as e:  # noqa: BLE001
            raise TransientError(f"complete update failed: {e}") from e

    def fail(self, job_id: str, error: str, *, requeue: bool) -> None:
        status = "pending" if requeue else "failed"
        patch: dict[str, Any] = {
            "status": status,
            "error": error,
            "updated_at": _now_iso(),
        }
        if requeue:
            # increment retry_count; supabase-py lacks native ++; fetch+set
            try:
                resp = self._client.table("agent_jobs").select("retry_count").eq("id", job_id).execute()
                current = (resp.data or [{}])[0].get("retry_count", 0)
            except Exception as e:  # noqa: BLE001
                raise TransientError(f"fail read retry_count failed: {e}") from e
            patch["retry_count"] = int(current) + 1
        try:
            self._client.table("agent_jobs").update(patch).eq("id", job_id).execute()
        except Exception as e:  # noqa: BLE001
            raise TransientError(f"fail update failed: {e}") from e

    def reap_stuck(self, cutoff_minutes: int, max_retries: int) -> None:
        cutoff = (datetime.now(UTC) - timedelta(minutes=cutoff_minutes)).isoformat()
        try:
            resp = (
                self._client.table("agent_jobs")
                .select("*")
                .eq("type", self.JOB_TYPE)
                .eq("status", "processing")
                .lt("started_at", cutoff)
                .execute()
            )
        except Exception as e:  # noqa: BLE001
            raise TransientError(f"reap select failed: {e}") from e

        for row in resp.data or []:
            retry = int(row.get("retry_count") or 0)
            if retry + 1 < max_retries:
                patch = {
                    "status": "pending",
                    "retry_count": retry + 1,
                    "updated_at": _now_iso(),
                }
            else:
                patch = {
                    "status": "failed",
                    "error": "INTERNAL: stuck job, exceeded retry cap",
                    "updated_at": _now_iso(),
                }
            try:
                self._client.table("agent_jobs").update(patch).eq("id", row["id"]).execute()
            except Exception as e:  # noqa: BLE001
                raise TransientError(f"reap update failed: {e}") from e

    # ---- PhotoStore ---------------------------------------------------

    def fetch(self, asset_id: str) -> bytes:
        try:
            resp = (
                self._client.table("image_assets")
                .select("bucket,path")
                .eq("id", asset_id)
                .single()
                .execute()
            )
        except Exception as e:  # noqa: BLE001
            raise TransientError(f"image_assets lookup failed: {e}") from e
        bucket = resp.data["bucket"]
        path = resp.data["path"]
        try:
            return self._client.storage.from_(bucket).download(path)
        except Exception as e:  # noqa: BLE001
            raise TransientError(f"storage download failed: {e}") from e

    # ---- ResultStore --------------------------------------------------

    def upsert_avatar(self, app_user_id: str, payload: dict[str, Any]) -> None:
        try:
            resp = (
                self._client.table("user_avatar_preferences")
                .select("initial_state")
                .eq("app_user_id", app_user_id)
                .execute()
            )
        except Exception as e:  # noqa: BLE001
            raise TransientError(f"avatar read failed: {e}") from e

        rows = resp.data or []
        patch: dict[str, Any] = {
            "app_user_id": app_user_id,
            "current_state": payload,
            "updated_at": _now_iso(),
        }
        if not rows or not rows[0].get("initial_state"):
            patch["initial_state"] = payload

        try:
            self._client.table("user_avatar_preferences").upsert(
                patch, on_conflict="app_user_id"
            ).execute()
        except Exception as e:  # noqa: BLE001
            raise TransientError(f"avatar upsert failed: {e}") from e

    # ---- internals ----------------------------------------------------

    def _latest_asset_id(self, app_user_id: str | None) -> str | None:
        if not app_user_id:
            return None
        try:
            resp = (
                self._client.table("image_assets")
                .select("id")
                .eq("owner_id", app_user_id)
                .eq("type", "profile")
                .eq("is_active", True)
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
        except Exception:  # noqa: BLE001
            return None
        rows = resp.data or []
        return rows[0]["id"] if rows else None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
