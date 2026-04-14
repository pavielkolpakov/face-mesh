import logging

from face_mesh.observability.logging import bind_job_id, configure_logging, unbind_job_id
from face_mesh.observability.timing import timed


def test_timed_logs_duration(caplog):
    configure_logging("INFO")
    caplog.set_level(logging.INFO)
    with timed("demo"):
        pass
    events = [r for r in caplog.records if "demo.done" in r.getMessage()]
    assert len(events) == 1
    assert hasattr(events[0], "duration_ms")


def test_bind_and_unbind_job_id_round_trip():
    token = bind_job_id("job-123")
    try:
        pass
    finally:
        unbind_job_id(token)
