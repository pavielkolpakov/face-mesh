from __future__ import annotations


class PipelineError(Exception):
    """Base class for classified worker errors."""

    prefix: str = "INTERNAL"

    @property
    def public_message(self) -> str:
        return f"{self.prefix}: {self}"


class InputError(PipelineError):
    """Bad user input — photo unreadable, no face/body detected, etc. Not retried."""

    prefix = "INPUT"


class TransientError(PipelineError):
    """Transient infra failure — retryable up to the worker's retry cap."""

    prefix = "INTERNAL"
