"""Scoring engine.

Strategies are independent of OCR: they receive question + student answer +
rubric and return per-criterion scores. Rubric weights and knockout rules are
configurable per question in the DB. Feedback is generated only when enabled
by a flag.
"""
from __future__ import annotations

from app.config import get_settings
from app.logging import get_logger
from app.providers.llm.minimax import parse_json_lenient
from app.schemas import (
    CriteriaScore,
    LLMMessage,
    ScoreResult,
)

logger = get_logger("exam_eval.scoring")


def normalize_text(text: str) -> str:
    import re
    import unicodedata

    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class ExactMatchStrategy:
    """Deterministic: full credit when the normalized answer matches the key."""

    name = "exact"

    def __init__(self, settings=None):
        self.settings = settings or get_settings()

    async def score(
        self, question: dict, answer: str, rubric: list[dict] | None = None, **_: object
    ) -> ScoreResult:
        max_score = float(question.get("max_score") or 0)
        key = question.get("answer_key")
        alternatives = [
            part.strip()
            for part in (question.get("answer_key_alternatives") or "").split("|")
            if part.strip()
        ]
        if key:
            alternatives.insert(0, key)
        normalized = normalize_text(answer)
        matched = bool(normalized) and any(
            normalized == normalize_text(candidate) for candidate in alternatives
        )
        awarded = max_score if matched else 0.0
        return ScoreResult(
            question_id=question["id"],
            question_no=question["q_no"],
            total_awarded=awarded,
            total_max=max_score,
            criteria=[
                CriteriaScore(
                    criterion="correctness",
                    awarded=awarded,
                    max_score=max_score,
                    rationale="Exact key match" if matched else "Does not match the answer key",
                    confidence=1.0,
                )
            ],
            strategy=self.name,
            provider="deterministic",
        )


class NumericStrategy:
    """Deterministic numeric comparison with an absolute tolerance."""

    name = "numeric"

    _NUMBER = None

    def __init__(self, settings=None, tolerance_ratio: float = 0.02):
        self.settings = settings or get_settings()
        self.tolerance_ratio = tolerance_ratio

    @classmethod
    def _numbers(cls, text: str) -> list[float]:
        import re

        return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text or "")]

    async def score(
        self, question: dict, answer: str, rubric: list[dict] | None = None, **_: object
    ) -> ScoreResult:
        max_score = float(question.get("max_score") or 0)
        key_numbers = self._numbers(question.get("answer_key") or "")
        given = self._numbers(answer)
        expected = key_numbers[0] if key_numbers else None
        got = given[-1] if given else None
        if expected is None or got is None:
            return ScoreResult(
                question_id=question["id"],
                question_no=question["q_no"],
                total_awarded=0.0,
                total_max=max_score,
                criteria=[
                    CriteriaScore(
                        criterion="correctness",
                        awarded=0.0,
                        max_score=max_score,
                        rationale="Could not parse a numeric answer",
                        confidence=0.3,
                    )
                ],
                strategy=self.name,
                provider="deterministic",
            )
        tolerance = abs(expected) * self.tolerance_ratio or self.tolerance_ratio
        diff = abs(got - expected)
        if diff <= tolerance:
            awarded = max_score
            rationale = f"Correct ({got} == {expected})"
        elif diff <= 4 * tolerance:
            awarded = max_score / 2
            rationale = f"Approximate ({got} vs {expected})"
        else:
            awarded = 0.0
            rationale = f"Incorrect ({got} != {expected})"
        return ScoreResult(
            question_id=question["id"],
            question_no=question["q_no"],
            total_awarded=awarded,
            total_max=max_score,
            criteria=[
                CriteriaScore(
                    criterion="correctness",
                    awarded=awarded,
                    max_score=max_score,
                    rationale=rationale,
                    confidence=1.0,
                )
            ],
            strategy=self.name,
            provider="deterministic",
        )


RUBRIC_SYSTEM_PROMPT = """You are a fair exam evaluator for school answers (Hindi and/or English).
Score the student's answer strictly against the rubric criteria and the reference answer.
Rules:
- Awarded marks must be between 0 and the criterion maximum.
- Do not let grammar, spelling or handwriting affect correctness/completeness criteria.
- If the student's reasoning is wrong, correctness must be low even if the language is fluent.
- Base your judgement ONLY on the answer text provided. Never invent content.
- If the answer is empty, award 0 for everything.
Return JSON only:
{"criteria": [{"criterion": "<name>", "awarded": <number>, "rationale": "<short>"}],
 "confidence": 0.0-1.0, "feedback": "<optional short feedback for the student>"}
"""


def build_rubric_prompt(question: dict, answer: str, rubric: list[dict], enable_feedback: bool) -> str:
    criteria_lines = "\n".join(
        f"- {c['name']} (max {c['max_score']}): {c.get('description') or ''}"
        for c in rubric
    )
    feedback_line = (
        'Include a short "feedback" string.' if enable_feedback else "Do not include feedback; omit the key."
    )
    return (
        f"QUESTION ({question['q_no']}): {question.get('text')}\n"
        f"REFERENCE ANSWER: {question.get('answer_key') or '(none provided)'}\n"
        f"RUBRIC:\n{criteria_lines}\n\n"
        f"STUDENT_ANSWER:\n{answer.strip() or '(empty)'}\n\n"
        f"{feedback_line}\nReturn the JSON object now."
    )


class RubricLLMStrategy:
    """Rubric scoring via the configured LLM provider."""

    name = "rubric_llm"

    def __init__(self, llm, settings=None):
        self.llm = llm
        self.settings = settings or get_settings()

    async def score(
        self,
        question: dict,
        answer: str,
        rubric: list[dict] | None = None,
        *,
        context: dict | None = None,
    ) -> ScoreResult:
        rubric = rubric or [
            {"name": "correctness", "max_score": float(question.get("max_score") or 0), "weight": 1.0}
        ]
        try:
            result = await self.llm.complete(
                [
                    LLMMessage(role="system", content=RUBRIC_SYSTEM_PROMPT),
                    LLMMessage(
                        role="user",
                        content=build_rubric_prompt(
                            question, answer, rubric, self.settings.scoring_enable_feedback
                        ),
                    ),
                ],
                json_mode=True,
                stage="score_rubric",
                context=context,
            )
            data = parse_json_lenient(result.text)
        except Exception as exc:  # noqa: BLE001
            return ScoreResult(
                question_id=question["id"],
                question_no=question["q_no"],
                total_awarded=0.0,
                total_max=float(question.get("max_score") or 0),
                criteria=[],
                strategy=self.name,
                error=str(exc)[:300],
            )

        by_name = {c["name"]: c for c in rubric}
        criteria_scores: list[CriteriaScore] = []
        for item in data.get("criteria", []):
            name = str(item.get("criterion") or "").strip()
            spec = by_name.get(name)
            if spec is None:
                continue
            max_score = float(spec.get("max_score") or 0)
            awarded = _clamp_float(item.get("awarded"), 0.0, max_score)
            criteria_scores.append(
                CriteriaScore(
                    criterion=name,
                    awarded=awarded,
                    max_score=max_score,
                    rationale=str(item.get("rationale") or "")[:500] or None,
                    confidence=_safe_float(data.get("confidence")),
                )
            )
        if not criteria_scores:
            return ScoreResult(
                question_id=question["id"],
                question_no=question["q_no"],
                total_awarded=0.0,
                total_max=float(question.get("max_score") or 0),
                criteria=[],
                strategy=self.name,
                error="model returned no valid criteria",
            )

        total_awarded, knockout_applied = apply_rubric_rules(criteria_scores, rubric)
        if knockout_applied:
            for criterion in criteria_scores:
                if criterion.rationale:
                    criterion.rationale += " [knockout rule applied]"
                    break
        feedback = data.get("feedback") if self.settings.scoring_enable_feedback else None
        if feedback:
            criteria_scores[-1].feedback = str(feedback)[:1000]
        return ScoreResult(
            question_id=question["id"],
            question_no=question["q_no"],
            total_awarded=total_awarded,
            total_max=float(question.get("max_score") or 0),
            criteria=criteria_scores,
            strategy=self.name,
            provider=result.provider,
            model=result.model,
        )


class HybridScoringStrategy:
    """Deterministic when a key exists, LLM rubric otherwise. Per-question override wins."""

    name = "hybrid"

    def __init__(self, llm, settings=None):
        self.settings = settings or get_settings()
        self.exact = ExactMatchStrategy(settings)
        self.numeric = NumericStrategy(settings)
        self.rubric = RubricLLMStrategy(llm, settings)

    async def score(
        self, question: dict, answer: str, rubric: list[dict] | None = None, *, context: dict | None = None
    ) -> ScoreResult:
        mode = (question.get("strategy") or "").strip() or self.settings.scoring_strategy
        key = question.get("answer_key") or ""
        if mode == "exact":
            return await self.exact.score(question, answer, rubric)
        if mode == "numeric":
            return await self.numeric.score(question, answer, rubric)
        if mode == "hybrid" and key:
            if _looks_numeric(key):
                return await self.numeric.score(question, answer, rubric)
            return await self.exact.score(question, answer, rubric)
        return await self.rubric.score(question, answer, rubric, context=context)


def apply_rubric_rules(criteria: list[CriteriaScore], rubric: list[dict]) -> tuple[float, bool]:
    """Weighted total with knockout rules enforced in code, not by the model."""
    weights = {c["name"]: float(c.get("weight") or 1.0) for c in rubric}
    knockouts = {c["name"]: c for c in rubric if c.get("knockout")}
    total = 0.0
    knockout_applied = False
    for criterion in criteria:
        weight = weights.get(criterion.criterion, 1.0)
        total += criterion.awarded * weight
    for criterion in criteria:
        rule = knockouts.get(criterion.criterion)
        if not rule:
            continue
        threshold = float(rule.get("knockout_below_fraction") or 0.0)
        if criterion.max_score > 0 and criterion.awarded <= threshold * criterion.max_score:
            knockout_applied = True
            break
    if knockout_applied:
        total = 0.0
    return round(total, 2), knockout_applied


def _looks_numeric(text: str) -> bool:
    import re

    return bool(re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", text or ""))


def _clamp_float(value, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, number))


def _safe_float(value) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None
