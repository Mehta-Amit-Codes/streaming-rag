"""
Ingestion queue: the "cron + queue" half of the suggested stack.

Pollers (scheduler), webhooks and manual pushes all SUBMIT work here instead of ingesting inline,
so fetching/embedding is decoupled from request handling and from each other: one slow feed can't
stall the API, and a single worker thread serialises writes (no two ingestions racing on the same
item). With INGEST_WORKER=0 tasks run inline in the caller, which keeps tests deterministic.
Swap this class for a Kafka/Redis/Celery consumer without touching the callers.
"""
import queue
import threading

from app.config import INGEST_WORKER


class Task:
    def __init__(self, fn, args):
        self.fn, self.args = fn, args
        self.done = threading.Event()
        self.result = None
        self.error: str | None = None

    def run(self):
        try:
            self.result = self.fn(*self.args)
        except Exception as e:  # last-resort guard: the worker thread must never die
            self.error = f"{type(e).__name__}: {e}"
        finally:
            self.done.set()

    def wait(self, timeout: float | None = None):
        self.done.wait(timeout)
        return self.result


class IngestQueue:
    def __init__(self, threaded: bool):
        self.threaded = threaded
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self):
        if self.threaded and self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="ingest-worker")
            self._thread.start()

    def stop(self):
        self._stop.set()
        self._q.put(None)

    def submit(self, fn, *args) -> Task:
        task = Task(fn, args)
        if self.threaded:
            self._q.put(task)
        else:
            task.run()
        return task

    def pending(self) -> int:
        return self._q.qsize()

    def _loop(self):
        while not self._stop.is_set():
            task = self._q.get()
            if task is None:
                break
            task.run()


ingest_queue = IngestQueue(threaded=INGEST_WORKER)
