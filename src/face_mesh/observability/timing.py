from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

from face_mesh.observability.logging import get_logger


@contextmanager
def timed(stage: str, logger: logging.Logger | None = None) -> Iterator[dict]:
    log = logger or get_logger("face_mesh.timing")
    started = time.perf_counter()
    ctx: dict = {}
    try:
        yield ctx
    finally:
        duration_ms = int((time.perf_counter() - started) * 1000)
        log.info(f"{stage}.done", extra={"stage": stage, "duration_ms": duration_ms, **ctx})
