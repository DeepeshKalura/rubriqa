"""Validation tests: valid result, missing answer, duplicate question, malformed."""
from __future__ import annotations

from app.pipeline.validation import (
    ExtractedAnswerView,
    PageReading,
    StructuralValidator,
    has_blocking,
    run_validators,
)
from app.schemas import ValidationSeverity

validator = StructuralValidator()


def payload(**overrides):
    base = {
        "submission_id": 1,
        "document_id": 1,
        "pages": [PageReading(page_id=1, index=0, text="Q1. 56", confidence=0.9, state="read")],
        "expected_questions": [{"id": 1, "q_no": "Q1"}, {"id": 2, "q_no": "Q2"}],
        "answers": [],
        "check_answers": False,
    }
    base.update(overrides)
    return base


def test_valid_pages_pass():
    issues = validator.validate(payload())
    assert not has_blocking(issues)
    assert issues == []


def test_no_pages_is_blocking():
    issues = validator.validate(payload(pages=[]))
    assert has_blocking(issues)
    assert issues[0].code == "no_pages"


def test_failed_page_is_blocking():
    pages = [PageReading(page_id=1, index=0, text="", confidence=0.0, state="error")]
    issues = validator.validate(payload(pages=pages))
    assert has_blocking(issues)
    assert any(issue.code == "page_processing_failed" for issue in issues)


def test_empty_page_is_warning_not_blocking():
    pages = [PageReading(page_id=1, index=0, text="", confidence=0.0, state="read")]
    issues = validator.validate(payload(pages=pages))
    assert not has_blocking(issues)
    assert any(issue.code == "empty_page_extraction" for issue in issues)


def test_low_confidence_is_warning():
    pages = [PageReading(page_id=1, index=0, text="some text", confidence=0.2, state="read")]
    issues = validator.validate(payload(pages=pages))
    assert not has_blocking(issues)
    assert any(issue.code == "low_confidence_reading" for issue in issues)


def test_missing_answer_when_checking():
    issues = validator.validate(payload(check_answers=True, answers=[]))
    missing = [i for i in issues if i.code == "missing_answer"]
    assert len(missing) == 2
    assert not has_blocking(issues)


def test_duplicate_answers_are_blocking():
    answers = [
        ExtractedAnswerView(question_id=1, question_no="Q1", answer_text="56"),
        ExtractedAnswerView(question_id=1, question_no="Q1", answer_text="56 again"),
    ]
    issues = validator.validate(payload(check_answers=True, answers=answers))
    assert has_blocking(issues)
    assert any(issue.code == "duplicate_question_ids" for issue in issues)


def test_duplicate_exam_questions_are_blocking():
    issues = validator.validate(
        payload(expected_questions=[{"id": 1, "q_no": "Q1"}, {"id": 2, "q_no": "Q1"}])
    )
    assert has_blocking(issues)


def test_empty_answer_is_warning():
    answers = [ExtractedAnswerView(question_id=1, question_no="Q1", answer_text="   ")]
    issues = validator.validate(payload(check_answers=True, answers=answers))
    assert any(issue.code == "empty_answer" for issue in issues)
    assert not has_blocking(issues)


def test_run_validators_is_extensible():
    class ContentValidator:
        name = "content"

        def validate(self, payload):
            from app.schemas import ValidationIssueData

            return [
                ValidationIssueData(
                    severity=ValidationSeverity.INFO, code="content_note", message="note"
                )
            ]

    issues = run_validators(payload(), [validator, ContentValidator()])
    assert any(issue.code == "content_note" for issue in issues)
