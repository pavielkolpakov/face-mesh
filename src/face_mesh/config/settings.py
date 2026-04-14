from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import SecretStr


@dataclass(frozen=True)
class ModelPaths:
    face_landmarker: str = "models/face_landmarker.task"
    pose_landmarker: str = "models/pose_landmarker_lite.task"
    segmenter: str = "models/selfie_segmenter.tflite"


@dataclass(frozen=True)
class Settings:
    poll_interval_s: int
    max_retries: int
    stuck_job_cutoff_minutes: int
    log_level: str
    models: ModelPaths
    supabase_url: str
    supabase_service_role_key: SecretStr
    heuristic_bounds_path: Path

    @classmethod
    def load(cls, config_dir: Path | None = None) -> Settings:
        config_dir = config_dir or Path("config")
        default_yaml = config_dir / "default.yaml"
        data: dict[str, Any] = {}
        if default_yaml.exists():
            data = yaml.safe_load(default_yaml.read_text()) or {}

        models_cfg = data.get("models", {})
        return cls(
            poll_interval_s=int(data.get("poll_interval_s", 3)),
            max_retries=int(data.get("max_retries", 3)),
            stuck_job_cutoff_minutes=int(data.get("stuck_job_cutoff_minutes", 10)),
            log_level=os.getenv("LOG_LEVEL", data.get("log_level", "INFO")),
            models=ModelPaths(
                face_landmarker=models_cfg.get("face_landmarker", ModelPaths().face_landmarker),
                pose_landmarker=models_cfg.get("pose_landmarker", ModelPaths().pose_landmarker),
                segmenter=models_cfg.get("segmenter", ModelPaths().segmenter),
            ),
            supabase_url=os.getenv("SUPABASE_URL", ""),
            supabase_service_role_key=SecretStr(os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")),
            heuristic_bounds_path=config_dir / "heuristic_bounds.yaml",
        )


def load_heuristic_bounds(path: Path) -> dict[str, dict[str, tuple[float, float]]]:
    data = yaml.safe_load(path.read_text()) or {}
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for section in ("face", "body"):
        section_data = data.get(section) or {}
        out[section] = {k: (float(v[0]), float(v[1])) for k, v in section_data.items()}
    return out
