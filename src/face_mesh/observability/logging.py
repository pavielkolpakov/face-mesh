from __future__ import annotations

import contextvars
import logging
import sys
from typing import Any

try:
    from pythonjsonlogger import json as jsonlogger  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    from pythonjsonlogger import jsonlogger  # type: ignore[no-redef]

_job_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("job_id", default=None)


class _JobIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = _job_id_var.get() or ""
        return True


def bind_job_id(job_id: str | None) -> contextvars.Token[str | None]:
    return _job_id_var.set(job_id)


def unbind_job_id(token: contextvars.Token[str | None]) -> None:
    _job_id_var.reset(token)


_OWNED_ATTR = "_face_mesh_owned"


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, _OWNED_ATTR, False):
            root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    setattr(handler, _OWNED_ATTR, True)
    fmt = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s %(job_id)s",
        rename_fields={"asctime": "ts", "levelname": "level"},
    )
    handler.setFormatter(fmt)
    handler.addFilter(_JobIdFilter())
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    logger.info(event, extra=fields)
