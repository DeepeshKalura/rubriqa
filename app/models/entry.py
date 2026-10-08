"""Entry-evaluation models (Program, Entry, OCR runs, cleaning, batches, samples).

The exam pipeline (`submission.py`) is kept for backwards compatibility; new flows
use these models. Each `Entry` represents one student's project idea submission
with full demographics, theme, situation text and proposed AI solution text.
The OCR runs keep both the original (audit) and cleaned text (for review),
along with token / confidence / status columns matching the client's sheet.
"""
from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, enum_column


class ProgramState(str, enum.Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    CLOSED = "closed"


class EntryState(str, enum.Enum):
    PENDING = "pending"
    READING = "reading"
    CLEANING = "cleaning"
    SCORING = "scoring"
    READY_FOR_REVIEW = "ready_for_review"
    NEEDS_REVIEW = "needs_review"
    COMPLETED = "completed"
    FAILED = "failed"


class OcrMethod(str, enum.Enum):
    TESSERACT = "tesseract"
    VISION_LLM = "vision_llm"
    HYBRID = "hybrid"


class OcrStatus(str, enum.Enum):
    OK = "ok"
    LOW_CONFIDENCE = "low_confidence"
    EMPTY = "empty"
    FAILED = "failed"


class BatchState(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    AWAITING_REVIEW = "awaiting_review"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class SampleKind(str, enum.Enum):
    RANDOM = "random"
    AI_FLAGGED = "ai_flagged"
    LOW_CONFIDENCE = "low_confidence"
    OVERRIDE = "override"


class ReviewStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    CORRECTED = "corrected"
    REJECTED = "rejected"


class Program(Base, TimestampMixin):
    """An AI Innovation Project program (e.g. 'HP pilot 2026')."""

    __tablename__ = "programs"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    state: Mapped[ProgramState] = mapped_column(
        enum_column(ProgramState, "program_state"), default=ProgramState.ACTIVE, nullable=False
    )
    rubric_id: Mapped[int | None] = mapped_column(
        ForeignKey("rubrics.id", ondelete="SET NULL")
    )
    batch_size: Mapped[int] = mapped_column(Integer, default=250)
    forms_csv_path: Mapped[str | None] = mapped_column(String(600))

    rubric: Mapped["Rubric"] = relationship()  # noqa: F821
    entries: Mapped[list["Entry"]] = relationship(
        back_populates="program", cascade="all, delete-orphan"
    )
    batches: Mapped[list["EvaluationBatch"]] = relationship(
        back_populates="program", cascade="all, delete-orphan"
    )


class Entry(Base, TimestampMixin):
    """One student submission to a Program."""

    __tablename__ = "entries"
    __table_args__ = (
        UniqueConstraint("program_id", "external_ref"),
        Index("ix_entries_program_state", "program_id", "state"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    program_id: Mapped[int] = mapped_column(
        ForeignKey("programs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    external_ref: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    state: Mapped[EntryState] = mapped_column(
        enum_column(EntryState, "entry_state"), default=EntryState.PENDING, nullable=False
    )
    # --- form answers
    form_kind: Mapped[str] = mapped_column(String(40), default="himachal")  # himachal | ashramshala
    full_name: Mapped[str | None] = mapped_column(String(200))
    gender: Mapped[str | None] = mapped_column(String(20))
    phone: Mapped[str | None] = mapped_column(String(40))
    school_name: Mapped[str | None] = mapped_column(String(300))
    class_name: Mapped[str | None] = mapped_column(String(40))
    state_name: Mapped[str | None] = mapped_column(String(80))
    district: Mapped[str | None] = mapped_column(String(120))
    block_name: Mapped[str | None] = mapped_column(String(120))
    division: Mapped[str | None] = mapped_column(String(120))
    project_office: Mapped[str | None] = mapped_column(String(120))
    ashramshala: Mapped[str | None] = mapped_column(String(200))
    language: Mapped[str | None] = mapped_column(String(40))
    theme: Mapped[str | None] = mapped_column(String(80), index=True)
    ai_description: Mapped[str | None] = mapped_column(Text)
    resources: Mapped[dict | None] = mapped_column(JSON)
    situation: Mapped[str | None] = mapped_column(Text)
    proposed_solution: Mapped[str | None] = mapped_column(Text)
    declaration: Mapped[bool] = mapped_column(Boolean, default=False)
    consent: Mapped[bool] = mapped_column(Boolean, default=False)
    po_name: Mapped[str | None] = mapped_column(String(200))
    school_name_form: Mapped[str | None] = mapped_column(String(300))
    submission_count: Mapped[int | None] = mapped_column(Integer)
    email: Mapped[str | None] = mapped_column(String(200))
    # --- content
    submitted_text: Mapped[str | None] = mapped_column(Text)  # final cleaned text snapshot
    # --- evaluation
    total_score: Mapped[float | None] = mapped_column(Float)
    max_score: Mapped[float | None] = mapped_column(Float)
    percentage: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(30), default="pending")
    # --- sources
    drive_url: Mapped[str | None] = mapped_column(String(800))
    drive_file_id: Mapped[str | None] = mapped_column(String(120), index=True)
    local_file_path: Mapped[str | None] = mapped_column(String(600))
    drive_sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    review_notes: Mapped[str | None] = mapped_column(Text)
    review_status: Mapped[str] = mapped_column(String(20), default="pending")  # pending|approved|corrected|rejected
    human_score: Mapped[float | None] = mapped_column(Float)
    human_score_notes: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict | None] = mapped_column(JSON)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    program: Mapped[Program] = relationship(back_populates="entries")
    ocr_runs: Mapped[list["EntryOCR"]] = relationship(
        back_populates="entry", cascade="all, delete-orphan", order_by="EntryOCR.id"
    )
    cleaning_runs: Mapped[list["CleaningRun"]] = relationship(
        back_populates="entry", cascade="all, delete-orphan", order_by="CleaningRun.id"
    )
    scores: Mapped[list["EntryScore"]] = relationship(
        back_populates="entry", cascade="all, delete-orphan", order_by="EntryScore.id"
    )
    samples: Mapped[list["ReviewSample"]] = relationship(
        back_populates="entry", cascade="all, delete-orphan"
    )


class EntryOCR(Base, TimestampMixin):
    """One OCR attempt for an entry. Both original and reconstructed text are kept."""

    __tablename__ = "entry_ocr"
    __table_args__ = (Index("ix_entry_ocr_entry", "entry_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    entry_id: Mapped[int] = mapped_column(
        ForeignKey("entries.id", ondelete="CASCADE"), nullable=False, index=True
    )
    method: Mapped[OcrMethod] = mapped_column(
        enum_column(OcrMethod, "ocr_method"), nullable=False
    )
    status: Mapped[OcrStatus] = mapped_column(
        enum_column(OcrStatus, "ocr_status"), nullable=False
    )
    language: Mapped[str | None] = mapped_column(String(20))
    # raw OCR text (the "OCR" column in the client sheet)
    original_text: Mapped[str | None] = mapped_column(Text)
    # cleaned + reconstructed text (the "OCR Reconstruction" column in the client sheet)
    reconstructed_text: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    provider: Mapped[str | None] = mapped_column(String(60))
    model: Mapped[str | None] = mapped_column(String(120))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    is_selected: Mapped[bool] = mapped_column(Boolean, default=True)

    entry: Mapped[Entry] = relationship(back_populates="ocr_runs")


class CleaningRun(Base, TimestampMixin):
    """One cleaning pass. Keeps original vs cleaned text for audit."""

    __tablename__ = "cleaning_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    entry_id: Mapped[int] = mapped_column(
        ForeignKey("entries.id", ondelete="CASCADE"), nullable=False, index=True
    )
    method: Mapped[str] = mapped_column(String(40), nullable=False)  # rules | llm
    original_text: Mapped[str] = mapped_column(Text, nullable=False)
    cleaned_text: Mapped[str] = mapped_column(Text, nullable=False)
    # what was changed (diff or notes)
    notes: Mapped[str | None] = mapped_column(Text)
    provider: Mapped[str | None] = mapped_column(String(60))
    model: Mapped[str | None] = mapped_column(String(120))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    applied: Mapped[bool] = mapped_column(Boolean, default=True)
    applied_by: Mapped[str] = mapped_column(String(40), default="auto")  # auto | human

    entry: Mapped[Entry] = relationship(back_populates="cleaning_runs")


class EntryScore(Base, TimestampMixin):
    """Whole-entry rubric scoring run. One row per run; criteria live in EntryScoreCriterion."""

    __tablename__ = "entry_scores"
    __table_args__ = (
        Index("ix_entry_scores_entry_current", "entry_id", "is_current"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    entry_id: Mapped[int] = mapped_column(
        ForeignKey("entries.id", ondelete="CASCADE"), nullable=False, index=True
    )
    batch_id: Mapped[int | None] = mapped_column(
        ForeignKey("evaluation_batches.id", ondelete="SET NULL"), index=True
    )
    strategy: Mapped[str] = mapped_column(String(60), default="entry_rubric_llm")
    provider: Mapped[str | None] = mapped_column(String(60))
    model: Mapped[str | None] = mapped_column(String(120))
    rubric_id: Mapped[int | None] = mapped_column(
        ForeignKey("rubrics.id", ondelete="SET NULL")
    )
    total_awarded: Mapped[float | None] = mapped_column(Float)
    total_max: Mapped[float | None] = mapped_column(Float)
    percentage: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    strengths: Mapped[str | None] = mapped_column(Text)
    gaps: Mapped[str | None] = mapped_column(Text)
    observations: Mapped[str | None] = mapped_column(Text)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    error: Mapped[str | None] = mapped_column(Text)

    entry: Mapped[Entry] = relationship(back_populates="scores")
    criteria: Mapped[list["EntryScoreCriterion"]] = relationship(
        back_populates="score", cascade="all, delete-orphan", order_by="EntryScoreCriterion.id"
    )


class EntryScoreCriterion(Base, TimestampMixin):
    __tablename__ = "entry_score_criteria"
    __table_args__ = (UniqueConstraint("score_id", "criterion"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    score_id: Mapped[int] = mapped_column(
        ForeignKey("entry_scores.id", ondelete="CASCADE"), nullable=False, index=True
    )
    criterion: Mapped[str] = mapped_column(String(80), nullable=False)
    awarded: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    max_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rationale: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)

    score: Mapped[EntryScore] = relationship(back_populates="criteria")


class EvaluationBatch(Base, TimestampMixin):
    """200-300 entries grouped for review + scoring with a checkpoint between batches."""

    __tablename__ = "evaluation_batches"
    __table_args__ = (Index("ix_eval_batches_program_state", "program_id", "state"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    program_id: Mapped[int] = mapped_column(
        ForeignKey("programs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    state: Mapped[BatchState] = mapped_column(
        enum_column(BatchState, "batch_state"), default=BatchState.PENDING, nullable=False
    )
    target_size: Mapped[int] = mapped_column(Integer, default=250)
    size: Mapped[int] = mapped_column(Integer, default=0)
    completed: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    flagged: Mapped[int] = mapped_column(Integer, default=0)
    notes: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    paused_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    program: Mapped[Program] = relationship(back_populates="batches")
    entries: Mapped[list["Entry"]] = relationship(
        back_populates=None, overlaps="batch"
    )  # entries.batch_id assigned via batch_id FK below
    samples: Mapped[list["ReviewSample"]] = relationship(
        back_populates="batch", cascade="all, delete-orphan"
    )


# Add batch_id FK to Entry (the column already exists in entry table via ad-hoc SQL;
# we add the FK relationship here as a forward reference). The actual column lives
# on the Entry model: batch_id (see the migration that creates it).
Entry.batch_id = mapped_column(  # type: ignore[attr-defined]
    ForeignKey("evaluation_batches.id", ondelete="SET NULL"), nullable=True, index=True
)
Entry.batch = relationship("EvaluationBatch", back_populates=None)  # type: ignore[attr-defined]


class ReviewSample(Base, TimestampMixin):
    """Human review sample for one entry in one batch."""

    __tablename__ = "review_samples"
    __table_args__ = (UniqueConstraint("batch_id", "entry_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    batch_id: Mapped[int] = mapped_column(
        ForeignKey("evaluation_batches.id", ondelete="CASCADE"), nullable=False, index=True
    )
    entry_id: Mapped[int] = mapped_column(
        ForeignKey("entries.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[SampleKind] = mapped_column(
        enum_column(SampleKind, "sample_kind"), nullable=False
    )
    status: Mapped[ReviewStatus] = mapped_column(
        enum_column(ReviewStatus, "review_status"),
        default=ReviewStatus.PENDING,
        nullable=False,
    )
    reviewer: Mapped[str | None] = mapped_column(String(120))
    notes: Mapped[str | None] = mapped_column(Text)
    final_score: Mapped[float | None] = mapped_column(Float)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    batch: Mapped[EvaluationBatch] = relationship(back_populates="samples")
    entry: Mapped[Entry] = relationship(back_populates="samples")


# ---------------------------------------------------------------------------
# Drive file mirror metadata
# ---------------------------------------------------------------------------

class DriveFile(Base, TimestampMixin):
    """Metadata about a Drive file (real or local mirror)."""

    __tablename__ = "drive_files"

    id: Mapped[int] = mapped_column(primary_key=True)
    drive_file_id: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(400), nullable=False)
    mime: Mapped[str | None] = mapped_column(String(120))
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    local_path: Mapped[str | None] = mapped_column(String(600))
    download_url: Mapped[str | None] = mapped_column(String(800))
    parent_folder: Mapped[str | None] = mapped_column(String(120), index=True)
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending|fetched|failed
    error: Mapped[str | None] = mapped_column(Text)


# Avoid circular import for type checker: Rubric is defined in app/models/exam.py
from app.models.exam import Rubric  # noqa: E402  (intentional late import)