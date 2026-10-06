"""Submissions, documents, pages, reading runs and repairs.

Traceability chain:
  submission -> document -> page -> reading_run -> repair
                                 -> answer -> score
"""
from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    BigInteger,
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


class SubmissionState(str, enum.Enum):
    PENDING = "pending"          # imported, pipeline not started
    PROCESSING = "processing"
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class DocumentKind(str, enum.Enum):
    TEXT = "text"
    PDF = "pdf"
    IMAGE = "image"
    GOOGLE_FORM = "google_form"


class PageState(str, enum.Enum):
    PENDING = "pending"
    READING = "reading"
    READ = "read"
    ERROR = "error"
    DEFERRED = "deferred"  # network/API unavailable, retry later


class ReadingStatus(str, enum.Enum):
    OK = "ok"
    LOW_CONFIDENCE = "low_confidence"
    EMPTY = "empty"
    FAILED = "failed"


class Submission(Base, TimestampMixin):
    __tablename__ = "submissions"
    __table_args__ = (
        # an exam can only be submitted once per student (idempotency anchor)
        UniqueConstraint("exam_id", "student_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    exam_id: Mapped[int] = mapped_column(
        ForeignKey("exams.id", ondelete="CASCADE"), nullable=False, index=True
    )
    student_id: Mapped[int] = mapped_column(
        ForeignKey("students.id", ondelete="CASCADE"), nullable=False, index=True
    )
    state: Mapped[SubmissionState] = mapped_column(
        enum_column(SubmissionState, "submission_state"),
        default=SubmissionState.PENDING,
        nullable=False,
        index=True,
    )
    total_score: Mapped[float | None] = mapped_column(Float)
    max_score: Mapped[float | None] = mapped_column(Float)
    percentage: Mapped[float | None] = mapped_column(Float)
    needs_review: Mapped[bool] = mapped_column(default=False)
    notes: Mapped[str | None] = mapped_column(Text)

    exam: Mapped["Exam"] = relationship()  # noqa: F821
    student: Mapped["Student"] = relationship(back_populates="submissions")  # noqa: F821
    documents: Mapped[list["Document"]] = relationship(
        back_populates="submission", cascade="all, delete-orphan"
    )
    answers: Mapped[list["Answer"]] = relationship(  # noqa: F821
        back_populates="submission", cascade="all, delete-orphan"
    )


class Document(Base, TimestampMixin):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("submission_id", "sha256"),
        Index("ix_documents_sha256", "sha256"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    submission_id: Mapped[int] = mapped_column(
        ForeignKey("submissions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[DocumentKind] = mapped_column(
        enum_column(DocumentKind, "document_kind"), nullable=False
    )
    original_filename: Mapped[str] = mapped_column(String(400), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(600), nullable=False)
    mime: Mapped[str | None] = mapped_column(String(120))
    size_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    source: Mapped[str] = mapped_column(String(40), default="upload")
    metadata_json: Mapped[dict | None] = mapped_column(JSON)
    processed_version: Mapped[str | None] = mapped_column(String(40))

    submission: Mapped[Submission] = relationship(back_populates="documents")
    pages: Mapped[list["Page"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="Page.index"
    )


class Page(Base, TimestampMixin):
    __tablename__ = "pages"
    __table_args__ = (UniqueConstraint("document_id", "index"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    index: Mapped[int] = mapped_column(Integer, nullable=False)  # 0-based
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    has_text_layer: Mapped[bool] = mapped_column(default=False)
    direct_text: Mapped[str | None] = mapped_column(Text)  # from PDF text layer
    image_path: Mapped[str | None] = mapped_column(String(600))  # rendered page image
    state: Mapped[PageState] = mapped_column(
        enum_column(PageState, "page_state"), default=PageState.PENDING, nullable=False
    )
    reading_version: Mapped[str | None] = mapped_column(String(40))

    document: Mapped[Document] = relationship(back_populates="pages")
    reading_runs: Mapped[list["ReadingRun"]] = relationship(
        back_populates="page", cascade="all, delete-orphan", order_by="ReadingRun.id"
    )


class ReadingRun(Base, TimestampMixin):
    """One attempt by one provider to read one page. Immutable record."""

    __tablename__ = "reading_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    page_id: Mapped[int] = mapped_column(
        ForeignKey("pages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    provider: Mapped[str] = mapped_column(String(60), nullable=False)
    model: Mapped[str | None] = mapped_column(String(120))
    mode: Mapped[str] = mapped_column(String(40), nullable=False)  # reader mode used
    version: Mapped[str | None] = mapped_column(String(40))  # processing version
    status: Mapped[ReadingStatus] = mapped_column(
        enum_column(ReadingStatus, "reading_status"), nullable=False
    )
    text: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    language: Mapped[str | None] = mapped_column(String(20))
    segments_json: Mapped[dict | None] = mapped_column(JSON)  # word boxes etc.
    warnings: Mapped[str | None] = mapped_column(Text)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    is_selected: Mapped[bool] = mapped_column(default=False)  # winning reading

    page: Mapped[Page] = relationship(back_populates="reading_runs")
    repairs: Mapped[list["Repair"]] = relationship(  # noqa: F821
        back_populates="reading_run", cascade="all, delete-orphan"
    )


class Repair(Base, TimestampMixin):
    """Original OCR text vs candidate repair. Both are kept forever."""

    __tablename__ = "repairs"

    id: Mapped[int] = mapped_column(primary_key=True)
    reading_run_id: Mapped[int] = mapped_column(
        ForeignKey("reading_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    provider: Mapped[str] = mapped_column(String(60), nullable=False)
    model: Mapped[str | None] = mapped_column(String(120))
    original_text: Mapped[str] = mapped_column(Text, nullable=False)
    repaired_text: Mapped[str] = mapped_column(Text, nullable=False)
    original_confidence: Mapped[float | None] = mapped_column(Float)
    repaired_confidence: Mapped[float | None] = mapped_column(Float)
    reason: Mapped[str | None] = mapped_column(Text)
    accepted: Mapped[bool] = mapped_column(default=False)
    accepted_by: Mapped[str | None] = mapped_column(String(40))  # auto | human
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[int | None] = mapped_column(Integer)

    reading_run: Mapped[ReadingRun] = relationship(back_populates="repairs")


class ValidationIssue(Base, TimestampMixin):
    __tablename__ = "validation_issues"

    id: Mapped[int] = mapped_column(primary_key=True)
    submission_id: Mapped[int | None] = mapped_column(
        ForeignKey("submissions.id", ondelete="CASCADE"), index=True
    )
    document_id: Mapped[int | None] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    page_id: Mapped[int | None] = mapped_column(
        ForeignKey("pages.id", ondelete="CASCADE"), index=True
    )
    question_id: Mapped[int | None] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"), index=True
    )
    severity: Mapped[str] = mapped_column(String(20), nullable=False)  # info|warning|error
    code: Mapped[str] = mapped_column(String(60), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    stage: Mapped[str] = mapped_column(String(30), default="validation")
    details: Mapped[dict | None] = mapped_column(JSON)
    resolved: Mapped[bool] = mapped_column(default=False)
    resolution: Mapped[str | None] = mapped_column(String(200))
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Answer(Base, TimestampMixin):
    __tablename__ = "answers"
    __table_args__ = (
        # exactly one extracted answer per question per submission
        UniqueConstraint("submission_id", "question_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    submission_id: Mapped[int] = mapped_column(
        ForeignKey("submissions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    question_id: Mapped[int] = mapped_column(
        ForeignKey("questions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    method: Mapped[str] = mapped_column(String(40), default="llm")  # llm|regex|manual
    confidence: Mapped[float | None] = mapped_column(Float)
    source_page_id: Mapped[int | None] = mapped_column(
        ForeignKey("pages.id", ondelete="SET NULL")
    )
    evidence: Mapped[dict | None] = mapped_column(JSON)  # spans/offsets

    submission: Mapped[Submission] = relationship(back_populates="answers")
    question: Mapped["Question"] = relationship()  # noqa: F821
    scores: Mapped[list["Score"]] = relationship(  # noqa: F821
        back_populates="answer", cascade="all, delete-orphan"
    )


class ScoringRun(Base, TimestampMixin):
    __tablename__ = "scoring_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    submission_id: Mapped[int] = mapped_column(
        ForeignKey("submissions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    strategy: Mapped[str] = mapped_column(String(60), nullable=False)
    provider: Mapped[str | None] = mapped_column(String(60))
    model: Mapped[str | None] = mapped_column(String(120))
    rubric_id: Mapped[int | None] = mapped_column(ForeignKey("rubrics.id", ondelete="SET NULL"))
    params_json: Mapped[dict | None] = mapped_column(JSON)
    total_awarded: Mapped[float | None] = mapped_column(Float)
    total_max: Mapped[float | None] = mapped_column(Float)
    is_current: Mapped[bool] = mapped_column(default=True)
    error: Mapped[str | None] = mapped_column(Text)

    scores: Mapped[list["Score"]] = relationship(  # noqa: F821
        back_populates="scoring_run", cascade="all, delete-orphan"
    )


class Score(Base, TimestampMixin):
    """Score for one criterion of one answer within one scoring run."""

    __tablename__ = "scores"
    __table_args__ = (UniqueConstraint("scoring_run_id", "answer_id", "criterion"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    scoring_run_id: Mapped[int] = mapped_column(
        ForeignKey("scoring_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    answer_id: Mapped[int] = mapped_column(
        ForeignKey("answers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    criterion: Mapped[str] = mapped_column(String(80), nullable=False, default="overall")
    awarded: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    max_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rationale: Mapped[str | None] = mapped_column(Text)
    feedback: Mapped[str | None] = mapped_column(Text)  # optional, flag-gated
    confidence: Mapped[float | None] = mapped_column(Float)
    strategy: Mapped[str] = mapped_column(String(60), nullable=False)
    provider: Mapped[str | None] = mapped_column(String(60))
    model: Mapped[str | None] = mapped_column(String(120))

    scoring_run: Mapped[ScoringRun] = relationship(back_populates="scores")
    answer: Mapped[Answer] = relationship(back_populates="scores")
