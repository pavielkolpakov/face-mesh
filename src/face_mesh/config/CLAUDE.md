# config

Layered settings: `config/default.yaml` + env vars (+ `.env` in dev).

- `settings.py` — `Settings.load(config_dir)` returns a frozen dataclass. Secrets use `pydantic.SecretStr`.
- `load_heuristic_bounds(path)` → `{"face": {param: (lo, hi)}, "body": {...}}`.
- Required env: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` (service-role, bypasses RLS). Optional: `LOG_LEVEL`.
- YAMLs live at repo root `config/`, not inside the package — so deploys can override without rebuilding the wheel.

Rules:
- Never log secrets. Never serialize `Settings` to disk.
- Add new knobs to `default.yaml` with a sane default; new secrets via env only.
