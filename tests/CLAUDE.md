# tests

- `unit/` — synthetic fixtures, no MediaPipe, no Supabase. Must run in <2s.
- `golden/` — real MediaPipe on `fat_face.png` / `fit_face.png` / `skinny_face.png`. Skip-when-model-missing; tolerance `1e-3`. Fixtures in `tests/golden/expected/*.json`.
- `conftest.py` adds `--regen-golden` flag.

Rules:
- Drive behavior through public APIs only. No testing of privates.
- Worker & adapter tests use in-memory fakes from `tests/unit/test_worker_run_once.py` and `_FakeClient` in `test_supabase_adapter.py`.
- Regenerate goldens: `uv run pytest --regen-golden tests/golden`.
