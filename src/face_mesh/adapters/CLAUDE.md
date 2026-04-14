# adapters

External system boundaries. Implement `domain.protocols`.

- `supabase.py` — `SupabaseAdapter` implements `JobQueue + PhotoStore + ResultStore`.
  - `JOB_TYPE = "avatar_blendshapes"`.
  - Claim is two-phase CAS on `agent_jobs` (`status='pending'` + `updated_at` compare) — lose-races safely return `None`.
  - `reap_stuck`: sweeps `processing` rows older than cutoff; `retry_count+1 < max_retries` → back to `pending`, else `failed` with `"INTERNAL: stuck job, exceeded retry cap"`.
  - `upsert_avatar`: sets `initial_state` only when row missing or `initial_state` empty; `current_state` every time.
  - Photo lookup v1: `_latest_asset_id(app_user_id)` grabs newest `image_assets.type='profile'`. Per PLAN this is hardcoded; proper job→asset linkage is deferred.
  - Every external call is wrapped; raw errors → `TransientError`.

Rules:
- No business logic here — just translate between protocol calls and supabase-py.
- Tested via an in-memory `_FakeClient` (see `tests/unit/test_supabase_adapter.py`). No live Supabase in CI.
