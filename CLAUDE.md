# face-mesh — Claude guide

Photo → Unity blendshapes (12 face + 10 body). Supabase-connected Python worker. Full design in PLAN.md.

## Commands
- `uv sync --extra dev` — install deps
- `uv run pytest` — all tests (unit + golden)
- `uv run pytest --regen-golden tests/golden` — rewrite golden JSON fixtures
- `uv run ruff check src tests` — lint
- `uv run face-mesh face <img>` — face pipeline → JSON
- `uv run face-mesh-worker` — start worker loop (needs `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`)
- `./scripts/download_models.sh` — fetch MediaPipe models into `models/`

## Layout
- `src/face_mesh/{domain,runners,extractors,mapping,adapters,worker,cli,config,observability}` — see each folder's CLAUDE.md
- `composition.py` — single wiring root; build via `build_face_system()` / `build_worker_system()`
- `tests/unit` fast + synthetic; `tests/golden` real MediaPipe (skip when model missing)
- `config/default.yaml` + `config/heuristic_bounds.yaml`
- Model files live in `models/` (gitignored)

## Phase status
Phases 0–2 shipped (face extraction + mapper + worker end-to-end). Phases 3–5 (body / calibration / polish) pending — see PLAN.md.

## Architectural invariants
- Extractors are **pure** over detection objects; no MediaPipe import.
- Mappers route per-param: `RegressionEstimator` if `is_trained`, else `HeuristicEstimator`.
- Errors classify via `domain/errors.py`: `InputError` (no retry) / `TransientError` (retry up to cap) / anything else (internal, no retry). Tracebacks only to logs, never to DB.
- Worker is single-concurrency, sync. Runners are not thread-safe.
- Output payload version `"1.0.0"`; `initial_state` set only on first upsert, `current_state` always.

## Maintenance rule (IMPORTANT)
When you add features, change architecture, or learn non-obvious info, **update the relevant CLAUDE.md files and/or create new ones** for new modules. Keep every CLAUDE.md file as brief as possible — bullets over prose, no repetition of PLAN.md.
