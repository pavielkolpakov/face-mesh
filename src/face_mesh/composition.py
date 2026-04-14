"""Wiring root — instantiates runners, extractors, mappers, pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from face_mesh.config.settings import Settings, load_heuristic_bounds
from face_mesh.extractors.face import FaceExtractor
from face_mesh.extractors.face_appearance import FaceAppearanceExtractor
from face_mesh.mapping.mapper import FaceMapper
from face_mesh.runners.face_landmarker import FaceLandmarkerRunner
from face_mesh.worker.loop import Worker
from face_mesh.worker.pipeline import FacePipeline


@dataclass
class FaceSystem:
    runner: FaceLandmarkerRunner
    pipeline: FacePipeline
    settings: Settings

    def close(self) -> None:
        self.runner.close()


def build_face_system(config_dir: Path | None = None) -> FaceSystem:
    settings = Settings.load(config_dir=config_dir)
    bounds_path = settings.heuristic_bounds_path
    all_bounds = load_heuristic_bounds(bounds_path) if bounds_path.exists() else {"face": {}, "body": {}}
    face_mapper = FaceMapper.from_heuristic_bounds(all_bounds.get("face", {}))
    mapper_artifact = Path("models/face_mapper.joblib")
    if mapper_artifact.exists():
        face_mapper = FaceMapper.load(mapper_artifact)

    runner = FaceLandmarkerRunner(settings.models.face_landmarker)
    pipeline = FacePipeline(
        face_extractor=FaceExtractor(),
        appearance_extractor=FaceAppearanceExtractor(),
        face_mapper=face_mapper,
    )
    return FaceSystem(runner=runner, pipeline=pipeline, settings=settings)


@dataclass
class WorkerSystem:
    worker: Worker
    settings: Settings


def build_worker_system(config_dir: Path | None = None) -> WorkerSystem:
    face = build_face_system(config_dir)
    from supabase import create_client

    from face_mesh.adapters.supabase import SupabaseAdapter

    client = create_client(
        face.settings.supabase_url,
        face.settings.supabase_service_role_key.get_secret_value(),
    )
    adapter = SupabaseAdapter(client)
    worker = Worker(
        queue=adapter,
        photo_store=adapter,
        result_store=adapter,
        runner=face.runner,
        pipeline=face.pipeline,
        max_retries=face.settings.max_retries,
        stuck_cutoff_minutes=face.settings.stuck_job_cutoff_minutes,
    )
    return WorkerSystem(worker=worker, settings=face.settings)
