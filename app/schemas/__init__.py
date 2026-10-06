"""Canonical contract objects shared across the pipeline.

These are deliberately provider-agnostic: ingestion adapters produce a
CanonicalDocument, readers produce ReadingResult, scoring strategies produce
ScoreResult. Markdown is never the internal representation.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    TEXT = "text"
    PDF = "pdf"
    IMAGE = "image"
    GOOGLE_FORM = "google_form"


class TextBlock(BaseModel):
    text: str
    block_no: int = 0
    bbox: tuple[float, float, float, float] | None = None
    kind: str = "text"  # text | heading | table | form_field


class CanonicalPage(BaseModel):
    index: int
    width: int | None = None
    height: int | None = None
    has_text_layer: bool = False
    text: str | None = None
    blocks: list[TextBlock] = Field(default_factory=list)
    image_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalDocument(BaseModel):
    source_type: SourceType
    original_filename: str
    sha256: str
    size_bytes: int = 0
    mime: str | None = None
    pages: list[CanonicalPage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def full_text(self) -> str:
        parts = []
        for page in self.pages:
            if page.text:
                parts.append(page.text)
            elif page.blocks:
                parts.append("\n".join(b.text for b in page.blocks))
        return "\n\n".join(parts).strip()


class ReadingStatus(str, Enum):
    OK = "ok"
    LOW_CONFIDENCE = "low_confidence"
    EMPTY = "empty"
    FAILED = "failed"


class ReadingResult(BaseModel):
    text: str = ""
    confidence: float = 0.0
    status: ReadingStatus = ReadingStatus.OK
    blocks: list[TextBlock] = Field(default_factory=list)
    engine: str = ""
    provider: str = ""
    model: str | None = None
    mode: str = ""
    version: str | None = None
    language: str | None = None
    warnings: list[str] = Field(default_factory=list)
    latency_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: dict[str, Any] | None = None


class RepairResult(BaseModel):
    original_text: str
    repaired_text: str
    reason: str
    confidence: float | None = None
    accepted: bool = False
    accepted_by: str | None = None


class ValidationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ValidationIssueData(BaseModel):
    severity: ValidationSeverity
    code: str
    message: str
    stage: str = "validation"
    document_id: int | None = None
    page_id: int | None = None
    question_id: int | None = None
    details: dict[str, Any] | None = None


class ExtractedAnswer(BaseModel):
    question_id: int
    question_no: str
    answer_text: str
    confidence: float | None = None
    method: str = "llm"
    source_page_id: int | None = None
    evidence: dict[str, Any] | None = None


class CriteriaScore(BaseModel):
    criterion: str
    awarded: float
    max_score: float
    rationale: str | None = None
    feedback: str | None = None
    confidence: float | None = None


class ScoreResult(BaseModel):
    question_id: int
    question_no: str
    total_awarded: float
    total_max: float
    criteria: list[CriteriaScore] = Field(default_factory=list)
    strategy: str = ""
    provider: str | None = None
    model: str | None = None
    error: str | None = None


class LLMMessage(BaseModel):
    role: str
    content: str


class LLMResult(BaseModel):
    text: str
    provider: str
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: int | None = None
    raw: dict[str, Any] | None = None
