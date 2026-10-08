"""Entry-level rubric scoring.

This is the "whole submission" scoring path that powers entry evaluation. The
client's rubric is configured as a Rubric + RubricCriterion list (re-using the
existing model), and the LLM is asked to score each criterion against the
student's situation + proposed AI solution text. The LLM also returns a short
"strengths" / "gaps" narrative and per-criterion rationales.

Token usage is recorded on `entry_scores` so the final export contains the same
Input/Output/Total Token columns the client sheet expects.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.logging import get_logger
from app.models import Entry, EntryScore, EntryScoreCriterion, Rubric, RubricCriterion
from app.pipeline.cleaning import rule_clean
from app.providers.llm import LLMProvider, parse_json_lenient
from app.schemas import LLMMessage

logger = get_logger("exam_eval.entry_score")


SYSTEM_PROMPT = """You are an expert, fair evaluator of student project idea submissions.
You will receive a student's description of a real-life situation they want to
solve (50-100 words) and their proposed AI-based solution (100-150 words).
You will also receive a rubric with named criteria. For each criterion you must:

  - assign an integer score between 0 and the criterion's max_score (inclusive)
  - give a short rationale (one or two sentences, in the same language as the submission)
  - note a confidence between 0 and 1

After the per-criterion scores, write:
  - strengths: 2-3 short bullets of what the student did well
  - gaps: 1-3 short bullets of what was unclear / missing / weak
  - observations: a short overall paragraph (3-5 sentences)

Do NOT translate the student's text. Score what they wrote. Penalise only when the
submission is missing, off-topic, or shows no understanding of the criterion.
Return JSON only, with this exact shape:

{
  "criteria": [
    {"name": "<criterion name>", "awarded": 0, "max_score": 0, "rationale": "<text>", "confidence": 0.0}
  ],
  "strengths": ["...", "..."],
  "gaps": ["..."],
  "observations": "<paragraph>"
}
"""


@dataclass
class EntryScoreResult:
    score_id: int | None
    total_awarded: float
    total_max: float
    percentage: float
    confidence: float
    strengths: list[str]
    gaps: list[str]
    observations: str
    criteria: list[dict]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: int
    strategy: str = "entry_rubric_llm"
    provider: str | None = None
    model: str | None = None
    error: str | None = None


async def _rubric_spec(session: AsyncSession, rubric_id: int | None) -> list[dict]:
    """Pull a rubric as a list of plain dicts for the prompt. Default rubric is
    a 4-criterion generic one if none is set on the program.
    """
    if rubric_id is None:
        return _default_rubric()
    rows = (
        await session.execute(
            select(RubricCriterion)
            .where(RubricCriterion.rubric_id == rubric_id)
            .order_by(RubricCriterion.id)
        )
    ).scalars().all()
    if not rows:
        return _default_rubric()
    return [
        {
            "name": r.name,
            "weight": float(r.weight),
            "description": r.description or "",
            "max_score": 10.0,  # rubric points default to 10 per criterion; the program
            # can override by editing max_score on the rubric criterion row.
        }
        for r in rows
    ]


def _default_rubric() -> list[dict]:
    return [
        {
            "name": "relevance",
            "weight": 1.0,
            "description": "How relevant the proposal is to the chosen theme and the described situation.",
            "max_score": 10.0,
        },
        {
            "name": "feasibility",
            "weight": 1.0,
            "description": "How realistic it is to build this in the student's context.",
            "max_score": 10.0,
        },
        {
            "name": "originality",
            "weight": 1.0,
            "description": "How original / non-trivially-copied the idea is.",
            "max_score": 10.0,
        },
        {
            "name": "ai_understanding",
            "weight": 1.0,
            "description": "How clearly the student explains how AI is used in the solution.",
            "max_score": 10.0,
        },
        {
            "name": "communication",
            "weight": 1.0,
            "description": "Clarity, structure, and respect for the word-count guidance.",
            "max_score": 10.0,
        },
    ]


async def score_entry(
    session: AsyncSession,
    entry: Entry,
    *,
    llm: LLMProvider,
    settings: Settings,
    rubric_id: int | None = None,
    batch_id: int | None = None,
) -> EntryScoreResult:
    rubric = await _rubric_spec(session, rubric_id)
    situation = entry.situation or ""
    solution = entry.proposed_solution or ""
    cleaned_sit = rule_clean(situation).cleaned_text
    cleaned_sol = rule_clean(solution).cleaned_text
    prompt = (
        f"Theme: {entry.theme or '(unspecified)'}\n"
        f"Language: {entry.language or 'unknown'}\n"
        f"Class: {entry.class_name or 'unspecified'}\n"
        f"AI description (student's own words):\n{entry.ai_description or '(none)'}\n\n"
        f"Situation (situation):\n{cleaned_sit}\n\n"
        f"Proposed AI Solution:\n{cleaned_sol}\n\n"
        f"Rubric:\n{json.dumps(rubric, ensure_ascii=False)}\n"
    )
    started = time.perf_counter()
    try:
        result = await llm.complete(
            messages=[
                LLMMessage(role="system", content=SYSTEM_PROMPT),
                LLMMessage(role="user", content=prompt),
            ],
            json_mode=True,
            max_tokens=settings.minimax_max_output_tokens,
            stage="entry_rubric_llm",
            temperature=0.0,
            context={"entry_id": entry.id, "batch_id": batch_id},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("entry_score_llm_failed", entry_id=entry.id, error=str(exc)[:300])
        # persist an empty error run
        run = EntryScore(
            entry_id=entry.id,
            batch_id=batch_id,
            strategy="entry_rubric_llm",
            provider=settings.llm_provider,
            model=settings.minimax_text_model,
            rubric_id=rubric_id,
            is_current=False,
            error=str(exc)[:500],
        )
        session.add(run)
        await session.commit()
        return EntryScoreResult(
            score_id=None,
            total_awarded=0,
            total_max=sum(c.get("max_score", 0) for c in rubric),
            percentage=0,
            confidence=0,
            strengths=[], gaps=[], observations="",
            criteria=[], input_tokens=0, output_tokens=0, total_tokens=0,
            latency_ms=int((time.perf_counter() - started) * 1000),
            error=str(exc)[:500],
        )

    data = parse_json_lenient(result.text)
    criteria = data.get("criteria") or []
    total_awarded = 0.0
    # total_max always reflects the rubric's max regardless of whether the
    # model returned any criteria (so percentage remains meaningful when
    # the model returns []).
    total_max = sum(float(c.get("max_score") or 0) for c in rubric)
    confidences = []
    normalised_criteria: list[dict] = []
    for spec, scored in zip(rubric, criteria):
        aw = float(scored.get("awarded") or 0)
        mx = float(spec.get("max_score") or 0)
        total_awarded += max(0.0, min(aw, mx))
        if scored.get("confidence") is not None:
            try:
                confidences.append(float(scored["confidence"]))
            except (TypeError, ValueError):
                pass
        normalised_criteria.append(
            {
                "name": spec["name"],
                "awarded": max(0.0, min(aw, mx)),
                "max_score": mx,
                "rationale": (scored.get("rationale") or "")[:1000],
                "confidence": float(scored.get("confidence") or 0.0),
            }
        )

    # supersede previous is_current runs
    previous = (
        await session.execute(
            select(EntryScore).where(
                EntryScore.entry_id == entry.id,
                EntryScore.is_current.is_(True),
            )
        )
    ).scalars().all()
    for prev in previous:
        prev.is_current = False

    overall_conf = sum(confidences) / len(confidences) if confidences else 0.0
    run = EntryScore(
        entry_id=entry.id,
        batch_id=batch_id,
        strategy="entry_rubric_llm",
        provider=settings.llm_provider,
        model=result.model or settings.minimax_text_model,
        rubric_id=rubric_id,
        total_awarded=round(total_awarded, 2),
        total_max=round(total_max, 2),
        percentage=round(100.0 * total_awarded / total_max, 1) if total_max else 0.0,
        confidence=round(overall_conf, 3),
        strengths="\n".join((data.get("strengths") or []))[:4000],
        gaps="\n".join((data.get("gaps") or []))[:4000],
        observations=(data.get("observations") or "")[:4000],
        input_tokens=result.prompt_tokens,
        output_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        is_current=True,
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)
    for nc in normalised_criteria:
        session.add(
            EntryScoreCriterion(
                score_id=run.id,
                criterion=nc["name"],
                awarded=nc["awarded"],
                max_score=nc["max_score"],
                rationale=nc["rationale"],
                confidence=nc["confidence"],
            )
        )
    entry.total_score = round(total_awarded, 2)
    entry.max_score = round(total_max, 2)
    entry.percentage = round(100.0 * total_awarded / total_max, 1) if total_max else 0.0
    entry.confidence = round(overall_conf, 3)
    await session.commit()

    return EntryScoreResult(
        score_id=run.id,
        total_awarded=round(total_awarded, 2),
        total_max=round(total_max, 2),
        percentage=round(100.0 * total_awarded / total_max, 1) if total_max else 0.0,
        confidence=round(overall_conf, 3),
        strengths=data.get("strengths") or [],
        gaps=data.get("gaps") or [],
        observations=(data.get("observations") or ""),
        criteria=normalised_criteria,
        input_tokens=result.prompt_tokens or 0,
        output_tokens=result.completion_tokens or 0,
        total_tokens=result.total_tokens or 0,
        latency_ms=result.latency_ms or int((time.perf_counter() - started) * 1000),
        strategy="entry_rubric_llm",
        provider=settings.llm_provider,
        model=result.model,
    )