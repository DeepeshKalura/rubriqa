"""Jobs, import batches, LLM call log and runtime settings."""
from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, enum_column


class JobState(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    READING = "reading"
    VALIDATING = "validating"
    EXTRACTING = "extracting"
    SCORING = "scoring"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, enum.Enum):
    INGEST_IMPORT = "ingest_import"
    PROCESS_SUBMISSION = "process_submission"
    RESCORE_SUBMISSION = "rescore_submission"
    REPROCESS_PAGE = "reprocess_page"


ACTIVE_STATES = {
    JobState.QUEUED,
    JobState.RUNNING,
    JobState.READING,
    JobState.VALIDATING,
    JobState.EXTRACTING,
    JobState.SCORING,
}


class Job(Base, TimestampMixin):
    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_state_created", "state", "created_at"),
        # a submission can have at most one active processing job
        Index(
            "uq_jobs_active_submission",
            "submission_id",
            unique=True,
            postgresql_where=text(
                "state IN ('queued','running','reading','validating','extracting','scoring')"
            ),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    type: Mapped[JobType] = mapped_column(enum_column(JobType, "job_type"), nullable=False)
    state: Mapped[JobState] = mapped_column(
        enum_column(JobState, "job_state"), default=JobState.QUEUED, nullable=False
    )
    submission_id: Mapped[int | None] = mapped_column(
        ForeignKey("submissions.id", ondelete="CASCADE"), index=True
    )
    document_id: Mapped[int | None] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    page_id: Mapped[int | None] = mapped_column(
        ForeignKey("pages.id", ondelete="CASCADE"), index=True
    )
    import_batch_id: Mapped[int | None] = mapped_column(
        ForeignKey("import_batches.id", ondelete="CASCADE"), index=True
    )
    priority: Mapped[int] = mapped_column(Integer, default=100)
    progress_current: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    checkpoint: Mapped[dict | None] = mapped_column(JSON, default=dict)
    params_json: Mapped[dict | None] = mapped_column(JSON)
    locked_by: Mapped[str | None] = mapped_column(String(120))
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(60))
    error: Mapped[str | None] = mapped_column(Text)

    events: Mapped[list["JobEvent"]] = relationship(
        back_populates="job", cascade="all, delete-orphan", order_by="JobEvent.id"
    )


class JobEvent(Base, TimestampMixin):
    __tablename__ = "job_events"
    __table_args__ = (Index("ix_job_events_job", "job_id", "id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    stage: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # start|ok|error|info
    message: Mapped[str | None] = mapped_column(Text)
    data_json: Mapped[dict | None] = mapped_column(JSON)
    duration_ms: Mapped[int | None] = mapped_column(Integer)

    job: Mapped[Job] = relationship(back_populates="events")


class ImportBatch(Base, TimestampMixin):
    __tablename__ = "import_batches"

    id: Mapped[int] = mapped_column(primary_key=True)
    exam_id: Mapped[int] = mapped_column(
        ForeignKey("exams.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source: Mapped[str] = mapped_column(String(40), nullable=False)  # upload|path|google
    root_path: Mapped[str | None] = mapped_column(String(600))
    manifest_json: Mapped[dict | None] = mapped_column(JSON)  # student -> [files]
    discovered: Mapped[int] = mapped_column(Integer, default=0)
    already_imported: Mapped[int] = mapped_column(Integer, default=0)
    new_documents: Mapped[int] = mapped_column(Integer, default=0)
    submissions_created: Mapped[int] = mapped_column(Integer, default=0)
    unmatched: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(30), default="draft")  # draft|committed|failed


class LlmCall(Base, TimestampMixin):
    """Usage log. Never contains the API key or full prompts with PII by default."""

    __tablename__ = "llm_calls"
    __table_args__ = (Index("ix_llm_calls_created", "created_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(60), nullable=False)
    model: Mapped[str | None] = mapped_column(String(120))
    stage: Mapped[str] = mapped_column(String(40), nullable=False)
    job_id: Mapped[int | None] = mapped_column(Integer, index=True)
    document_id: Mapped[int | None] = mapped_column(Integer)
    page_id: Mapped[int | None] = mapped_column(Integer)
    submission_id: Mapped[int | None] = mapped_column(Integer)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # ok|error
    error: Mapped[str | None] = mapped_column(Text)
    cost_usd: Mapped[float | None] = mapped_column(Float)


class Setting(Base):
    """Runtime-adjustable feature flags (overrides .env, e.g. reader mode)."""

    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(80), primary_key=True)
    value: Mapped[str] = mapped_column(String(400), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
