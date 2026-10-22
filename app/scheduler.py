"""
APScheduler: periodic polling + the retention purge. It only decides WHEN; the work itself goes
through the ingest queue. `sync_jobs()` reconciles scheduler jobs with the sources table, so it is
called at startup and after any source is created, changed or deleted.
"""
from app import registry
from app.connectors import POLLABLE_TYPES

_scheduler = None


def _poll_job(source_id: str):
    from app import ingestion
    from app.worker import ingest_queue
    ingest_queue.submit(ingestion.poll_source, source_id, "schedule")


def _purge_job():
    from app import ingestion
    from app.worker import ingest_queue
    ingest_queue.submit(ingestion.purge_expired)


def start():
    global _scheduler
    from apscheduler.schedulers.background import BackgroundScheduler
    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(_purge_job, "interval", hours=1, id="purge", replace_existing=True)
    sync_jobs()
    _scheduler.start()


def shutdown():
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None


def sync_jobs():
    if _scheduler is None:
        return
    wanted = {}
    for s in registry.list_all_sources():
        if s["enabled"] and s["type"] in POLLABLE_TYPES and s["poll_interval_seconds"] > 0:
            wanted[f"poll:{s['id']}"] = s
    for job in _scheduler.get_jobs():
        if job.id.startswith("poll:") and job.id not in wanted:
            job.remove()
    for job_id, s in wanted.items():
        _scheduler.add_job(_poll_job, "interval", seconds=s["poll_interval_seconds"], args=[s["id"]], id=job_id,
                           replace_existing=True, max_instances=1, coalesce=True)


def next_run(source_id: str) -> float | None:
    if _scheduler is None:
        return None
    job = _scheduler.get_job(f"poll:{source_id}")
    return job.next_run_time.timestamp() if job and job.next_run_time else None
