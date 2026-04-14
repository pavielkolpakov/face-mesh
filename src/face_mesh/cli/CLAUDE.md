# cli

Typer app. Entry point: `face-mesh` → `face_mesh.cli.main:app`.

- `main.py` — subcommands:
  - `face <image>` — run face pipeline on a local file, print JSON.
  - `body`, `all` — Phase 3 stubs (exit 1).
  - `worker` — delegates to `face_mesh.worker.loop.main`.
- `commands/` — reserved for Phase 4 `calibrate add | train | evaluate`.

Rules:
- CLI goes through `composition.build_face_system()` — swap `PhotoStore` / `ResultStore` here later rather than duplicating pipeline code.
- `InputError` → exit 2 with the public message on stderr.
