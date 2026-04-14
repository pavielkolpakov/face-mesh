from pathlib import Path

from face_mesh.config.settings import Settings, load_heuristic_bounds


def test_load_heuristic_bounds_reads_yaml(tmp_path: Path):
    yaml = tmp_path / "b.yaml"
    yaml.write_text("face:\n  faceWidth: [1.0, 2.9]\n  jawWidth: [0.82, 2.35]\nbody: {}\n")
    bounds = load_heuristic_bounds(yaml)
    assert bounds["face"]["faceWidth"] == (1.0, 2.9)
    assert bounds["face"]["jawWidth"] == (0.82, 2.35)
    assert bounds["body"] == {}


def test_settings_reads_env(monkeypatch, tmp_path: Path):
    (tmp_path / "default.yaml").write_text(
        "poll_interval_s: 3\nmax_retries: 3\nmodels:\n  face_landmarker: models/face.task\n"
    )
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "k")
    s = Settings.load(config_dir=tmp_path)
    assert s.poll_interval_s == 3
    assert s.max_retries == 3
    assert s.supabase_url == "https://x.supabase.co"
    assert s.supabase_service_role_key.get_secret_value() == "k"
