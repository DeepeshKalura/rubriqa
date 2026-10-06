"""Worker: claims jobs from Postgres with FOR UPDATE SKIP LOCKED and runs the
pipeline. No Redis/Celery: the jobs table is the queue.

Usage (inside the app image):
    python -m worker.main
"""
from __future__ import annotations

import asyncio
import signal
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import dispose_engine, get_sessionmaker
from app.logging import configure_logging, get_logger, timed
from app.models import Document, Job, JobState, JobType, LlmCall, Page, Submission
from app.pipeline.readers import build_reader, build_repairer
from app.providers.llm import build_provider
from app.services import jobs as jobs_service
from app.services.app_settings import get_effective_settings
from app.services.pipeline import PipelineError, SubmissionPipeline
from app.services.storage import Storage

logger = get_logger("exam_eval.worker")

_shutdown = asyncio.Event()


async def usage_hook(payload: dict[str, Any]) -> None:
    """Persist LLM usage. Never contains prompts or credentials."""
    maker = get_sessionmaker()
    async with maker() as session:
        session.add(LlmCall(**payload))
        await session.commit()


class Worker:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.storage = Storage()
        self.storage.ensure_dirs()
        self.llm = None
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        configure_logging(self.settings.log_level, self.settings.log_json)
        self.llm = build_provider(self.settings, usage_hook=usage_hook)
        logger.info(
            "worker_start",
            worker_id=jobs_service.WORKER_ID,
            concurrency=self.settings.worker_concurrency,
            provider=self.settings.llm_provider,
        )
        self._tasks = [
            asyncio.create_task(self._loop(i), name=f"worker-{i}")
            for i in range(self.settings.worker_concurrency)
        ]
        self._tasks.append(asyncio.create_task(self._sweeper(), name="sweeper"))
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def stop(self) -> None:
        _shutdown.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self.llm is not None and hasattr(self.llm, "aclose"):
            await self.llm.aclose()
        await dispose_engine()
        logger.info("worker_stop")

    # ------------------------------------------------------------------ loops
    async def _loop(self, index: int) -> None:
        maker = get_sessionmaker()
        idle_delay = 1.0
        while not _shutdown.is_set():
            try:
                async with maker() as session:
                    job = await jobs_service.claim_next_job(session)
                if job is None:
                    await asyncio.sleep(idle_delay)
                    idle_delay = min(idle_delay * 1.5, 5.0)
                    continue
                idle_delay = 1.0
                await self._run_job(job.id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the loop must survive
                logger.error("worker_loop_error", worker=index, error=str(exc)[:500])
                await asyncio.sleep(2.0)

    async def _sweeper(self) -> None:
        maker = get_sessionmaker()
        while not _shutdown.is_set():
            try:
                async with maker() as session:
                    count = await jobs_service.sweep_stale_jobs(session)
                if count:
                    logger.warning("sweeper_requeued", jobs=count)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error("sweeper_error", error=str(exc)[:300])
            try:
                await asyncio.wait_for(
                    _shutdown.wait(), timeout=self.settings.sweeper_interval_seconds
                )
            except asyncio.TimeoutError:
                pass

    # ---------------------------------------------------------------- job run
    async def _run_job(self, job_id: int) -> None:
        maker = get_sessionmaker()
        async with maker() as session:
            job = await session.get(Job, job_id)
            if job is None:
                return
            heartbeat_task = asyncio.create_task(self._heartbeat(job_id))
            try:
                settings = await get_effective_settings(session)
                reader = build_reader(settings.reader_mode, llm=self.llm, settings=settings)
                repairer = build_repairer(self.llm, settings)
                with timed(
                    logger, f"job_{job.type.value}", job_id=job.id, worker=jobs_service.WORKER_ID
                ):
                    if job.type is JobType.PROCESS_SUBMISSION:
                        await self._run_pipeline(session, job, settings, reader, repairer)
                    elif job.type is JobType.RESCORE_SUBMISSION:
                        await self._run_rescore(session, job, settings)
                    elif job.type is JobType.REPROCESS_PAGE:
                        await self._run_page_reprocess(session, job, settings, reader, repairer)
                    elif job.type is JobType.INGEST_IMPORT:
                        await self._run_import(session, job, settings)
                    else:
                        raise PipelineError(f"Unknown job type {job.type}", code="unknown_job")
            except PipelineError as exc:
                await jobs_service.fail(session, job, str(exc), code=exc.code)
            except Exception as exc:  # noqa: BLE001
                logger.error("job_crashed", job_id=job_id, error=str(exc)[:500])
                await session.rollback()
                job = await session.get(Job, job_id)
                if job is not None:
                    await jobs_service.fail(
                        session, job, f"{type(exc).__name__}: {exc}", code="crash"
                    )
            finally:
                heartbeat_task.cancel()

    async def _heartbeat(self, job_id: int) -> None:
        maker = get_sessionmaker()
        while True:
            await asyncio.sleep(self.settings.job_heartbeat_seconds)
            try:
                async with maker() as session:
                    await jobs_service.heartbeat(session, job_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("heartbeat_failed", job_id=job_id, error=str(exc)[:200])

    async def _run_pipeline(self, session, job, settings, reader, repairer) -> None:
        pipeline = SubmissionPipeline(
            session, settings, reader=reader, repairer=repairer, llm=self.llm,
            storage=self.storage,
        )
        await pipeline.run(job)

    async def _run_rescore(self, session, job, settings) -> None:
        from app.services.rescore import rescore_submission

        await rescore_submission(session, settings, self.llm, job)

    async def _run_page_reprocess(self, session, job, settings, reader, repairer) -> None:
        """Re-read a single page and re-run the rest of the pipeline."""
        page = await session.get(Page, job.page_id) if job.page_id else None
        if page is None:
            raise PipelineError("Page for reprocess job not found", code="missing_page")
        document = await session.get(Document, page.document_id)
        submission = await session.get(Submission, document.submission_id) if document else None
        if submission is None:
            raise PipelineError("Submission for reprocess job not found", code="missing_submission")
        job.checkpoint = {"read": False}
        job.params_json = {**(job.params_json or {}), "force_pages": True}
        await session.commit()
        pipeline = SubmissionPipeline(
            session, settings, reader=reader, repairer=repairer, llm=self.llm,
            storage=self.storage,
        )
        await pipeline.run(job)

    async def _run_import(self, session, job, settings) -> None:
        from app.services.importer import enqueue_submissions, register_documents, scan_folder

        params = job.params_json or {}
        root = params.get("root_path")
        exam_id = params.get("exam_id")
        if not root or not exam_id:
            raise PipelineError("Import job missing root_path/exam_id", code="bad_import_params")
        await jobs_service.transition(session, job, JobState.RUNNING, f"Scanning {root}")
        plan = scan_folder(self.storage.resolve(root))
        await jobs_service.transition(
            session,
            job,
            JobState.RUNNING,
            f"Registering {plan.discovered} file(s)",
            progress_total=plan.discovered,
        )
        batch = await register_documents(session, exam_id=exam_id, plan=plan)
        job.import_batch_id = batch.id
        await session.commit()
        await jobs_service.transition(session, job, JobState.RUNNING, "Queueing submissions")
        queued = await enqueue_submissions(session, exam_id)
        await jobs_service.complete(
            session,
            job,
            f"Imported {batch.new_documents} new document(s), {batch.already_imported} duplicate(s) "
            f"skipped, {queued} submission(s) queued",
        )


async def main() -> None:
    worker = Worker()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: _shutdown.set())
    try:
        await worker.start()
    finally:
        await worker.stop()


if __name__ == "__main__":
    asyncio.run(main())
