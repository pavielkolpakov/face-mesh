import threading

from face_mesh.worker.loop import Worker
from tests.unit.test_worker_run_once import (
    FakeFacePipeline,
    FakePhotoStore,
    FakeQueue,
    FakeResultStore,
    FakeRunner,
    _job,
)


def test_run_forever_stops_on_event_after_processing_queue():
    queue = FakeQueue(jobs=[_job(), _job()])
    worker = Worker(
        queue=queue,
        photo_store=FakePhotoStore(),
        result_store=FakeResultStore(),
        runner=FakeRunner(),
        pipeline=FakeFacePipeline(),
    )
    stop = threading.Event()

    # let the loop drain the 2 jobs then set stop when queue runs dry
    def watch():
        # busy-wait until both jobs are done, then signal stop
        import time

        for _ in range(100):
            if len(queue.completed) >= 2:
                stop.set()
                return
            time.sleep(0.01)
        stop.set()

    t = threading.Thread(target=watch)
    t.start()
    worker.run_forever(stop, poll_interval_s=0.01)
    t.join()

    assert len(queue.completed) == 2
    assert queue.reaped == 1
