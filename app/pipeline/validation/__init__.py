"""Validation layer.

Structural validation is pure data-in/data-out so it can be unit tested and
replaced. Content/semantic validation is intentionally a separate hook
(`ContentValidator`) that can be enabled later without touching this module.

Blocking errors stop scoring; warnings are recorded and surfaced in the UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.schemas import ValidationIssueData, ValidationSeverity

MIN_PAGE_CHARS = 3
LOW_CONFIDENCE_THRESHOLD = 0.5


@dataclass
class PageReading:
    page_id: int | None
    index: int
    text: str
    confidence: float
    state: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class ExtractedAnswerView:
    question_id: int
    question_no: str
    answer_text: str


class Validator(Protocol):
    name: str

    def validate(self, payload: dict) -> list[ValidationIssueData]: ...


class StructuralValidator:
    """Checks that pages, readings and extracted answers are internally sane."""

    name = "structural"
    stage = "validation"

    def validate(self, payload: dict) -> list[ValidationIssueData]:
        issues: list[ValidationIssueData] = []
        issues.extend(self._validate_pages(payload))
        issues.extend(self._validate_answers(payload))
        issues.extend(self._validate_exam(payload))
        return issues

    # -- pages ---------------------------------------------------------------
    def _validate_pages(self, payload: dict) -> list[ValidationIssueData]:
        issues: list[ValidationIssueData] = []
        pages: list[PageReading] = payload.get("pages", [])
        document_id = payload.get("document_id")
        if not pages:
            issues.append(
                ValidationIssueData(
                    severity=ValidationSeverity.ERROR,
                    code="no_pages",
                    message="Document has no processable pages.",
                    document_id=document_id,
                )
            )
            return issues
        for page in pages:
            if page.state == "error":
                issues.append(
                    ValidationIssueData(
                        severity=ValidationSeverity.ERROR,
                        code="page_processing_failed",
                        message=f"Page {page.index + 1} failed during reading.",
                        document_id=document_id,
                        page_id=page.page_id,
                    )
                )
                continue
            text = (page.text or "").strip()
            if len(text) < MIN_PAGE_CHARS:
                issues.append(
                    ValidationIssueData(
                        severity=ValidationSeverity.WARNING,
                        code="empty_page_extraction",
                        message=f"Page {page.index + 1} produced empty or near-empty text "
                        f"(got {len(text)} chars).",
                        document_id=document_id,
                        page_id=page.page_id,
                    )
                )
            if page.confidence < LOW_CONFIDENCE_THRESHOLD and text:
                issues.append(
                    ValidationIssueData(
                        severity=ValidationSeverity.WARNING,
                        code="low_confidence_reading",
                        message=f"Page {page.index + 1} reading confidence is "
                        f"{page.confidence:.2f}.",
                        document_id=document_id,
                        page_id=page.page_id,
                        details={"confidence": page.confidence},
                    )
                )
        return issues

    # -- answers -------------------------------------------------------------
    def _validate_answers(self, payload: dict) -> list[ValidationIssueData]:
        issues: list[ValidationIssueData] = []
        answers: list[ExtractedAnswerView] = payload.get("answers", [])
        submission_id = payload.get("submission_id")
        seen: set[str] = set()
        for answer in answers:
            if answer.question_no in seen:
                issues.append(
                    ValidationIssueData(
                        severity=ValidationSeverity.ERROR,
                        code="duplicate_question_ids",
                        message=f"Duplicate extracted answer for question "
                        f"{answer.question_no}.",
                        question_id=answer.question_id,
                        details={"question_no": answer.question_no},
                    )
                )
            seen.add(answer.question_no)
            if not (answer.answer_text or "").strip():
                issues.append(
                    ValidationIssueData(
                        severity=ValidationSeverity.WARNING,
                        code="empty_answer",
                        message=f"Question {answer.question_no} has an empty extracted answer.",
                        question_id=answer.question_id,
                    )
                )
        if submission_id is not None and payload.get("check_answers", False):
            expected: list[dict] = payload.get("expected_questions", [])
            answered = {a.question_id for a in answers}
            for question in expected:
                if question["id"] not in answered:
                    issues.append(
                        ValidationIssueData(
                            severity=ValidationSeverity.WARNING,
                            code="missing_answer",
                            message=f"No answer extracted for question {question['q_no']}.",
                            question_id=question["id"],
                            details={"question_no": question["q_no"]},
                        )
                    )
        return issues

    def _validate_exam(self, payload: dict) -> list[ValidationIssueData]:
        issues: list[ValidationIssueData] = []
        expected: list[dict] = payload.get("expected_questions", [])
        seen: set[str] = set()
        for question in expected:
            if question["q_no"] in seen:
                issues.append(
                    ValidationIssueData(
                        severity=ValidationSeverity.ERROR,
                        code="duplicate_question_ids",
                        message=f"Exam defines duplicate question number {question['q_no']}.",
                        question_id=question["id"],
                    )
                )
            seen.add(question["q_no"])
        if not expected and payload.get("answers"):
            issues.append(
                ValidationIssueData(
                    severity=ValidationSeverity.WARNING,
                    code="exam_has_no_questions",
                    message="Answers were extracted but the exam defines no questions.",
                )
            )
        return issues


class ContentValidator:
    """Placeholder for semantic validation (language sanity, answer relevance).

    Kept as a separate replaceable seam. It is deliberately not wired into the
    default pipeline yet: there is no labeled data to calibrate it against.
    """

    name = "content"

    def validate(self, payload: dict) -> list[ValidationIssueData]:
        return []


def run_validators(payload: dict, validators: list[Validator] | None = None) -> list[ValidationIssueData]:
    validators = validators or [StructuralValidator()]
    issues: list[ValidationIssueData] = []
    for validator in validators:
        issues.extend(validator.validate(payload))
    return issues


def has_blocking(issues: list[ValidationIssueData]) -> bool:
    return any(issue.severity is ValidationSeverity.ERROR for issue in issues)
