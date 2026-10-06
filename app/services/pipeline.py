"""Submission pipeline: READ -> VALIDATE -> EXTRACT -> SCORE.

Each stage commits its own transaction and records its completion in the job
checkpoint, so a retry resumes instead of repeating work. Page-level reading is
idempotent per (page, reading_version, provider).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.logging import get_logger, timed
from app.models import (
    Answer,
    Document,
    DocumentKind,
    Exam,
    Job,
    JobState,
    Page,
    PageState,
    Question,
    ReadingRun,
    ReadingStatus,
    Repair,
    Score,
    ScoringRun,
    Submission,
    SubmissionState,
    ValidationIssue,
)
from app.pipeline.extraction import LLMExtractor
from app.pipeline.readers.base import ReaderContext, ReaderError
from app.pipeline.scoring.factory import build_scoring
from app.pipeline.validation import (
    ExtractedAnswerView,
    PageReading,
    has_blocking,
    run_validators,
)
from app.schemas import CanonicalDocument, CanonicalPage, ReadingResult, SourceType
from app.services import jobs as jobs_service
from app.services.storage import Storage
from app.services.textutil import sanitize_text

logger = get_logger("exam_eval.pipeline")


class PipelineResult:
    def __init__(self):
        self.deferred_pages = 0
        self.blocking_issues = 0
        self.warning_issues = 0
        self.answers = 0
        self.scored = 0


class SubmissionPipeline:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        *,
        reader,
        repairer=None,
        llm,
        storage: Storage | None = None,
    ):
        self.session = session
        self.settings = settings
        self.reader = reader
        self.repairer = repairer
        self.llm = llm
        self.storage = storage or Storage()
        self.extractor = LLMExtractor(llm)

    # ------------------------------------------------------------------ entry
    async def run(self, job: Job) -> PipelineResult:
        result = PipelineResult()
        submission = await self.session.get(Submission, job.submission_id)
        if submission is None:
            raise PipelineError("Submission no longer exists", code="missing_submission")
        checkpoint = dict(job.checkpoint or {})
        force_pages = bool((job.params_json or {}).get("force_pages"))

        submission.state = SubmissionState.PROCESSING
        await self.session.commit()

        if not checkpoint.get("read"):
            await jobs_service.transition(
                self.session, job, JobState.READING, "Reading documents"
            )
            deferred = await self._read(job, submission, force=force_pages)
            result.deferred_pages = deferred
            checkpoint["read"] = True
            job.checkpoint = checkpoint
            await self.session.commit()

        # ---- validation (pass 1: pages + exam structure) --------------------
        await jobs_service.transition(self.session, job, JobState.VALIDATING, "Validating reading")
        issues = await self._validate(job, submission, check_answers=False)
        result.blocking_issues = sum(
            1 for issue in issues if issue.severity and issue.severity.value == "error"
        )
        result.warning_issues = len(issues) - result.blocking_issues
        if result.blocking_issues:
            await self._mark_needs_review(submission, job, result)
            return result
        checkpoint["validate"] = True

        # ---- extraction -----------------------------------------------------
        if not checkpoint.get("extract"):
            await jobs_service.transition(self.session, job, JobState.EXTRACTING)
            result.answers = await self._extract(job, submission)
            checkpoint["extract"] = True
            job.checkpoint = checkpoint
            await self.session.commit()

        # ---- validation (pass 2: extracted answers) -------------------------
        await jobs_service.transition(
            self.session, job, JobState.VALIDATING, "Validating extracted answers"
        )
        answer_issues = await self._validate(job, submission, check_answers=True)
        blocking = [i for i in answer_issues if i.severity.value == "error"]
        if blocking:
            result.blocking_issues += len(blocking)
            await self._mark_needs_review(submission, job, result)
            return result
        result.warning_issues = max(
            result.warning_issues,
            sum(1 for i in answer_issues if i.severity.value == "warning"),
        )
        checkpoint["validate_answers"] = True

        # ---- scoring --------------------------------------------------------
        await jobs_service.transition(self.session, job, JobState.SCORING)
        result.scored = await self._score(job, submission)
        checkpoint["score"] = True
        job.checkpoint = checkpoint
        await self.session.commit()

        submission.state = (
            SubmissionState.NEEDS_REVIEW
            if (result.deferred_pages or result.warning_issues)
            else SubmissionState.COMPLETED
        )
        if result.deferred_pages:
            submission.needs_review = True
            submission.notes = (
                f"{result.deferred_pages} page(s) could not be read (network/provider). "
                "They will be retried automatically when the provider is reachable."
            )
        await self.session.commit()
        await jobs_service.complete(self.session, job, "Pipeline finished")
        return result

    async def _mark_needs_review(
        self, submission: Submission, job: Job, result: PipelineResult
    ) -> None:
        submission.state = SubmissionState.NEEDS_REVIEW
        submission.needs_review = True
        submission.notes = (
            f"Blocked by {result.blocking_issues} validation error(s); scoring skipped."
        )
        await self.session.commit()
        await jobs_service.complete(
            self.session, job, f"Needs review: {result.blocking_issues} blocking issue(s)"
        )

    # -------------------------------------------------------------------- read
    async def _read(self, job: Job, submission: Submission, *, force: bool = False) -> int:
        pages = (
            (
                await self.session.execute(
                    select(Page)
                    .join(Document, Page.document_id == Document.id)
                    .where(Document.submission_id == submission.id)
                    .order_by(Document.id, Page.index)
                )
            )
            .scalars()
            .all()
        )
        total = len(pages)
        done = 0
        deferred = 0
        semaphore = asyncio.Semaphore(self.settings.ocr_threads)

        for page in pages:
            needs = force or page.state in (
                PageState.PENDING,
                PageState.ERROR,
                PageState.DEFERRED,
            ) or page.reading_version != self.settings.reading_version
            if not needs:
                done += 1
                continue
            document = await self.session.get(Document, page.document_id)
            assert document is not None
            try:
                reading = await self._read_page(document, page, semaphore=semaphore, force=force)
            except ReaderError as exc:
                if exc.retryable:
                    page.state = PageState.DEFERRED
                    await self.session.commit()
                    deferred += 1
                    await jobs_service.add_event(
                        self.session,
                        job,
                        "read",
                        "warning",
                        f"Page {page.index + 1} deferred: {exc}",
                        data={"page_id": page.id, "code": exc.code},
                    )
                    await self.session.commit()
                    continue
                page.state = PageState.ERROR
                await self.session.commit()
                continue
            except Exception as exc:  # noqa: BLE001 - isolate a bad page, not the whole job
                logger.error(
                    "page_read_failed",
                    job_id=job.id,
                    page_id=page.id,
                    error=str(exc)[:300],
                )
                page.state = PageState.ERROR
                await self.session.commit()
                continue

            page.state = PageState.READ
            page.reading_version = self.settings.reading_version
            # keep the winning reading text on the page for fast extraction
            page.direct_text = reading.text
            await self.session.commit()
            done += 1
            await jobs_service.transition(
                self.session,
                job,
                JobState.READING,
                f"Read {done}/{total} page(s)",
                progress_current=done,
                progress_total=total,
            )
        return deferred

    async def _read_page(
        self, document: Document, page: Page, *, semaphore: asyncio.Semaphore, force: bool
    ):
        # Text-based sources (txt/md/google forms) already have a text layer;
        # they bypass image readers entirely regardless of READER_MODE.
        if document.kind in (DocumentKind.TEXT, DocumentKind.GOOGLE_FORM):
            text = page.direct_text or ""
            reading = ReadingResult(
                text=text,
                confidence=1.0 if text.strip() else 0.0,
                status=ReadingStatus.OK if text.strip() else ReadingStatus.EMPTY,
                engine="text_layer",
                provider="builtin",
                mode="text_layer",
            )
            self.session.add(
                ReadingRun(
                    page_id=page.id,
                    provider=reading.provider,
                    mode=reading.mode,
                    version=self.settings.reading_version,
                    status=reading.status,
                    text=reading.text,
                    confidence=reading.confidence,
                    is_selected=True,
                )
            )
            await self.session.commit()
            return reading

        image_path = await self._ensure_page_image(document, page, semaphore)
        context = ReaderContext(
            document=CanonicalDocument(
                source_type=SourceType(document.kind.value),
                original_filename=document.original_filename,
                sha256=document.sha256,
            ),
            original_path=self.storage.resolve(document.storage_path),
            page_image_path=image_path,
        )
        canonical_page = CanonicalPage(
            index=page.index,
            has_text_layer=page.has_text_layer,
            text=page.direct_text if page.has_text_layer else None,
        )
        with timed(logger, "read_page", job_document_id=document.id, page_index=page.index) as _:
            reading = await self.reader.read(canonical_page, context)
        reading.text = sanitize_text(reading.text)
        run = ReadingRun(
            page_id=page.id,
            provider=reading.provider,
            model=reading.model,
            mode=reading.mode,
            version=self.settings.reading_version,
            status=reading.status,
            text=reading.text,
            confidence=reading.confidence,
            language=reading.language,
            segments_json=reading.raw,
            warnings=", ".join(reading.warnings) or None,
            latency_ms=reading.latency_ms,
            prompt_tokens=reading.prompt_tokens,
            completion_tokens=reading.completion_tokens,
            is_selected=True,
        )
        self.session.add(run)
        await self.session.commit()

        # ---- repair path ----------------------------------------------------
        if (
            self.repairer is not None
            and reading.text.strip()
            and reading.status in (ReadingStatus.LOW_CONFIDENCE, ReadingStatus.EMPTY)
        ):
            try:
                repaired = await self.repairer.repair(
                    reading, context, reason=f"reading status={reading.status.value}"
                )
            except ReaderError as exc:
                await jobs_service.add_event(
                    self.session,
                    job,
                    "repair",
                    "warning",
                    f"Repair failed for page {page.index + 1}: {exc}",
                )
                await self.session.commit()
                return reading
            accepted = bool(repaired.text.strip()) and (
                repaired.confidence >= self.settings.repair_auto_accept_min_legibility
            )
            self.session.add(
                Repair(
                    reading_run_id=run.id,
                    provider=repaired.provider,
                    model=repaired.model,
                    original_text=sanitize_text(reading.text),
                    repaired_text=sanitize_text(repaired.text),
                    original_confidence=reading.confidence,
                    repaired_confidence=repaired.confidence,
                    reason=repaired.raw.get("reason") if repaired.raw else None,
                    accepted=accepted,
                    accepted_by="auto" if accepted else None,
                    latency_ms=repaired.latency_ms,
                    prompt_tokens=repaired.prompt_tokens,
                    completion_tokens=repaired.completion_tokens,
                )
            )
            await self.session.commit()
            if accepted:
                run.is_selected = False
                repair_run = ReadingRun(
                    page_id=page.id,
                    provider=repaired.provider,
                    model=repaired.model,
                    mode="repair",
                    version=self.settings.reading_version,
                    status=repaired.status,
                    text=sanitize_text(repaired.text),
                    confidence=repaired.confidence,
                    segments_json=repaired.raw,
                    warnings="repaired, original_kept",
                    latency_ms=repaired.latency_ms,
                    is_selected=True,
                )
                self.session.add(repair_run)
                await self.session.commit()
                return repaired
        return reading

    async def _ensure_page_image(
        self, document: Document, page: Page, semaphore: asyncio.Semaphore
    ) -> Path:
        if page.image_path:
            existing = self.storage.resolve(page.image_path)
            if existing and existing.exists():
                return existing

        from app.pipeline.preprocessing import (
            load_image,
            render_pdf_page,
            save_page_image,
            save_thumbnail,
        )

        original = self.storage.resolve(document.storage_path)
        assert original is not None and original.exists(), "original file missing"

        async with semaphore:
            if document.kind is DocumentKind.PDF:
                image = await asyncio.to_thread(render_pdf_page, original, page.index)
            else:
                image = await asyncio.to_thread(load_image, original)
            target = self.storage.page_image_path(document.id, page.index)
            await asyncio.to_thread(save_page_image, image, target)
            thumb = self.storage.page_thumb_path(document.id, page.index)
            await asyncio.to_thread(save_thumbnail, image, thumb)

        page.image_path = self.storage.relative(target)
        page.width, page.height = image.size
        await self.session.commit()
        return target

    # -------------------------------------------------------------- validation
    async def _validate(self, job: Job, submission: Submission, *, check_answers: bool = False):
        documents = (
            (
                await self.session.execute(
                    select(Document).where(Document.submission_id == submission.id)
                )
            )
            .scalars()
            .all()
        )
        page_readings: list[PageReading] = []
        document_id = documents[0].id if documents else None
        for document in documents:
            pages = (
                (
                    await self.session.execute(
                        select(Page).where(Page.document_id == document.id).order_by(Page.index)
                    )
                )
                .scalars()
                .all()
            )
            for page in pages:
                selected = await self._selected_reading(page.id)
                page_readings.append(
                    PageReading(
                        page_id=page.id,
                        index=page.index,
                        text=selected.text if selected else (page.direct_text or ""),
                        confidence=selected.confidence if selected else 0.0,
                        state="error" if page.state is PageState.ERROR else page.state.value,
                        warnings=[],
                    )
                )

        questions = (
            (
                await self.session.execute(
                    select(Question).where(Question.exam_id == submission.exam_id)
                )
            )
            .scalars()
            .all()
        )

        answer_views: list[ExtractedAnswerView] = []
        if check_answers:
            from app.models import Answer as AnswerModel

            answers = (
                (
                    await self.session.execute(
                        select(AnswerModel).where(
                            AnswerModel.submission_id == submission.id
                        )
                    )
                )
                .scalars()
                .all()
            )
            question_numbers = {q.id: q.q_no for q in questions}
            answer_views = [
                ExtractedAnswerView(
                    question_id=a.question_id,
                    question_no=question_numbers.get(a.question_id, str(a.question_id)),
                    answer_text=a.text,
                )
                for a in answers
            ]

        payload = {
            "submission_id": submission.id,
            "document_id": document_id,
            "pages": page_readings,
            "expected_questions": [{"id": q.id, "q_no": q.q_no} for q in questions],
            "answers": answer_views,
            "check_answers": check_answers,
        }
        issues = run_validators(payload)

        # replace unresolved issues for this submission (idempotent re-validation)
        await self.session.execute(
            delete(ValidationIssue).where(
                ValidationIssue.submission_id == submission.id,
                ValidationIssue.resolved.is_(False),
            )
        )
        for issue in issues:
            self.session.add(
                ValidationIssue(
                    submission_id=submission.id,
                    document_id=issue.document_id,
                    page_id=issue.page_id,
                    question_id=issue.question_id,
                    severity=issue.severity.value,
                    code=issue.code,
                    message=issue.message,
                    stage=issue.stage,
                    details=issue.details,
                )
            )
        await self.session.commit()
        return issues

    async def _selected_reading(self, page_id: int) -> ReadingRun | None:
        return (
            await self.session.execute(
                select(ReadingRun)
                .where(ReadingRun.page_id == page_id, ReadingRun.is_selected.is_(True))
                .order_by(ReadingRun.id.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

    # -------------------------------------------------------------- extraction
    async def _extract(self, job: Job, submission: Submission) -> int:
        questions = (
            (
                await self.session.execute(
                    select(Question)
                    .where(Question.exam_id == submission.exam_id)
                    .order_by(Question.id)
                )
            )
            .scalars()
            .all()
        )
        text_parts: list[str] = []
        documents = (
            (
                await self.session.execute(
                    select(Document)
                    .where(Document.submission_id == submission.id)
                    .order_by(Document.id)
                )
            )
            .scalars()
            .all()
        )
        for document in documents:
            pages = (
                (
                    await self.session.execute(
                        select(Page).where(Page.document_id == document.id).order_by(Page.index)
                    )
                )
                .scalars()
                .all()
            )
            for page in pages:
                selected = await self._selected_reading(page.id)
                text = (selected.text if selected else None) or page.direct_text or ""
                if text.strip():
                    text_parts.append(f"--- page {page.index + 1} of {document.original_filename} ---\n{text}")
        transcript = "\n\n".join(text_parts)

        extracted = await self.extractor.extract(
            transcript,
            [{"id": q.id, "q_no": q.q_no, "text": q.text} for q in questions],
            context={"job_id": job.id, "submission_id": submission.id},
        )
        for answer in extracted:
            existing = (
                await self.session.execute(
                    select(Answer).where(
                        Answer.submission_id == submission.id,
                        Answer.question_id == answer.question_id,
                    )
                )
            ).scalar_one_or_none()
            if existing is None:
                self.session.add(
                    Answer(
                        submission_id=submission.id,
                        question_id=answer.question_id,
                        text=sanitize_text(answer.answer_text),
                        method=answer.method,
                        confidence=answer.confidence,
                    )
                )
            else:
                existing.text = sanitize_text(answer.answer_text)
                existing.method = answer.method
                existing.confidence = answer.confidence
        await self.session.commit()
        return len(extracted)

    # ----------------------------------------------------------------- scoring
    async def _score(self, job: Job, submission: Submission) -> int:
        questions = (
            (
                await self.session.execute(
                    select(Question)
                    .where(Question.exam_id == submission.exam_id)
                    .order_by(Question.id)
                )
            )
            .scalars()
            .all()
        )
        answers = (
            (
                await self.session.execute(
                    select(Answer).where(Answer.submission_id == submission.id)
                )
            )
            .scalars()
            .all()
        )
        answers_by_question = {a.question_id: a for a in answers}

        # supersede any previous scoring run (history is preserved)
        previous_runs = (
            (
                await self.session.execute(
                    select(ScoringRun).where(
                        ScoringRun.submission_id == submission.id, ScoringRun.is_current.is_(True)
                    )
                )
            )
            .scalars()
            .all()
        )
        for previous in previous_runs:
            previous.is_current = False

        exam = await self.session.get(Exam, submission.exam_id)
        strategy_name = (exam.scoring_strategy if exam else None) or self.settings.scoring_strategy
        run = ScoringRun(
            submission_id=submission.id,
            strategy=strategy_name,
            provider=self.settings.llm_provider,
            model=self.settings.minimax_text_model
            if self.settings.llm_provider == "minimax"
            else "fake",
            is_current=True,
            params_json={"enable_feedback": self.settings.scoring_enable_feedback},
        )
        self.session.add(run)
        await self.session.commit()

        total_awarded = 0.0
        total_max = 0.0
        scored = 0
        strategy = None
        strategy_cache: dict[str, object] = {}
        for question in questions:
            answer = answers_by_question.get(question.id)
            rubric = await self._rubric_spec(question)
            question_payload = {
                "id": question.id,
                "q_no": question.q_no,
                "text": question.text,
                "max_score": question.max_score,
                "answer_key": question.answer_key,
                "answer_key_alternatives": question.answer_key_alternatives,
                "strategy": question.strategy or exam.scoring_strategy if exam else question.strategy,
            }
            strategy_name = question_payload["strategy"] or strategy_name
            if strategy_name not in strategy_cache:
                strategy_cache[strategy_name] = build_scoring(
                    strategy_name, llm=self.llm, settings=self.settings
                )
            strategy = strategy_cache[strategy_name]
            answer_text = answer.text if answer else ""
            result = await strategy.score(
                question_payload,
                answer_text,
                rubric,
                context={
                    "job_id": job.id,
                    "submission_id": submission.id,
                    "question_id": question.id,
                },
            )
            total_max += float(question.max_score)
            if result.error:
                await jobs_service.add_event(
                    self.session,
                    job,
                    "score",
                    "warning",
                    f"Question {question.q_no}: {result.error}",
                )
            else:
                total_awarded += result.total_awarded
                scored += 1
            # persist per-criterion scores
            for criterion in result.criteria or []:
                if answer is not None:
                    self.session.add(
                        Score(
                            scoring_run_id=run.id,
                            answer_id=answer.id,
                            criterion=criterion.criterion,
                            awarded=criterion.awarded,
                            max_score=criterion.max_score,
                            rationale=sanitize_text(criterion.rationale) or None,
                            feedback=sanitize_text(criterion.feedback) or None,
                            confidence=criterion.confidence,
                            strategy=result.strategy or strategy_name,
                            provider=result.provider,
                            model=result.model,
                        )
                    )
        run.total_awarded = round(total_awarded, 2)
        run.total_max = round(total_max, 2)
        submission.total_score = round(total_awarded, 2)
        submission.max_score = round(total_max, 2)
        submission.percentage = (
            round(100.0 * total_awarded / total_max, 1) if total_max else 0.0
        )
        if submission.state is not SubmissionState.NEEDS_REVIEW:
            submission.state = SubmissionState.COMPLETED
        await self.session.commit()
        return scored

    async def _rubric_spec(self, question: Question) -> list[dict]:
        from app.models import Rubric, RubricCriterion

        rubric_id = question.rubric_id
        if rubric_id is None:
            return [
                {
                    "name": "correctness",
                    "max_score": float(question.max_score),
                    "weight": 1.0,
                    "description": "Correctness against the reference answer.",
                    "knockout": False,
                    "knockout_below_fraction": 0.0,
                }
            ]
        criteria = (
            (
                await self.session.execute(
                    select(RubricCriterion)
                    .where(RubricCriterion.rubric_id == rubric_id)
                    .order_by(RubricCriterion.id)
                )
            )
            .scalars()
            .all()
        )
        total_weight = sum(c.weight for c in criteria) or 1.0
        specs = []
        for criterion in criteria:
            specs.append(
                {
                    "name": criterion.name,
                    "max_score": round(float(question.max_score) * float(criterion.weight) / total_weight, 2),
                    "weight": 1.0,
                    "description": criterion.description,
                    "knockout": criterion.knockout,
                    "knockout_below_fraction": criterion.knockout_below_fraction,
                }
            )
        return specs


class PipelineError(Exception):
    def __init__(self, message: str, code: str = "pipeline_error"):
        super().__init__(message)
        self.code = code
