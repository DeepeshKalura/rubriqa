"""Scoring tests: exact, numeric, partial, rubric LLM, knockout, invalid input."""
from __future__ import annotations

import json

import pytest

from app.pipeline.scoring import (
    ExactMatchStrategy,
    HybridScoringStrategy,
    NumericStrategy,
    RubricLLMStrategy,
    apply_rubric_rules,
    normalize_text,
)
from app.pipeline.scoring.factory import build_scoring
from app.providers.llm.fake import FakeProvider, FailingProvider


def question(**overrides):
    base = {
        "id": 1,
        "q_no": "Q1",
        "text": "What is 7 x 8?",
        "max_score": 1.0,
        "answer_key": "56",
        "answer_key_alternatives": None,
        "strategy": None,
    }
    base.update(overrides)
    return base


def rubric_spec():
    return [
        {"name": "correctness", "max_score": 0.5, "weight": 1.0, "knockout": True,
         "knockout_below_fraction": 0.0},
        {"name": "completeness", "max_score": 0.3, "weight": 1.0, "knockout": False,
         "knockout_below_fraction": 0.0},
        {"name": "reasoning", "max_score": 0.2, "weight": 1.0, "knockout": False,
         "knockout_below_fraction": 0.0},
    ]


def test_normalize_text_strips_punctuation_and_case():
    assert normalize_text("  Hello, WORLD!! ") == "hello world"


async def test_exact_match_full_credit():
    result = await ExactMatchStrategy().score(question(), "56")
    assert result.total_awarded == 1.0


async def test_exact_match_case_and_punctuation():
    result = await ExactMatchStrategy().score(question(answer_key="New Delhi"), "new delhi.")
    assert result.total_awarded == 1.0


async def test_exact_match_wrong_answer():
    result = await ExactMatchStrategy().score(question(), "54")
    assert result.total_awarded == 0.0


async def test_exact_match_alternatives():
    q = question(answer_key="56", answer_key_alternatives="fifty six|56.0")
    result = await ExactMatchStrategy().score(q, "fifty-six")
    assert result.total_awarded == 1.0


async def test_exact_match_empty_answer():
    result = await ExactMatchStrategy().score(question(), "")
    assert result.total_awarded == 0.0


async def test_numeric_full_credit():
    result = await NumericStrategy().score(question(), "Answer is 56")
    assert result.total_awarded == 1.0


async def test_numeric_partial_credit_within_tolerance():
    result = await NumericStrategy(tolerance_ratio=0.02).score(question(), "57")
    # 57 vs 56: diff 1 > tolerance 1.12? 56*0.02=1.12 so within -> full
    assert result.total_awarded == 1.0
    result = await NumericStrategy(tolerance_ratio=0.005).score(question(), "57")
    assert result.total_awarded == 0.5  # within 4x tolerance


async def test_numeric_wrong():
    result = await NumericStrategy(tolerance_ratio=0.005).score(question(), "100")
    assert result.total_awarded == 0.0


async def test_numeric_unparseable():
    result = await NumericStrategy().score(question(), "fifty six")
    assert result.total_awarded == 0.0
    assert result.criteria[0].confidence <= 0.5


async def test_rubric_llm_with_fake_provider():
    llm = FakeProvider()
    result = await RubricLLMStrategy(llm).score(
        question(max_score=1.0), "An object stays at rest.", rubric_spec()
    )
    assert result.total_awarded > 0
    assert result.total_max == 1.0
    assert result.criteria[0].criterion == "correctness"
    assert llm.calls[0]["stage"] == "score_rubric"


async def test_rubric_llm_handles_invalid_json():
    llm = FakeProvider()
    llm.queue("score_rubric", "this is not json {")
    result = await RubricLLMStrategy(llm).score(question(), "answer", rubric_spec())
    assert result.error is not None
    assert result.total_awarded == 0.0


async def test_rubric_llm_handles_provider_failure():
    llm = FailingProvider()
    result = await RubricLLMStrategy(llm).score(question(), "answer", rubric_spec())
    assert result.error is not None
    assert result.total_awarded == 0.0


async def test_knockout_rule_zeroes_total():
    llm = FakeProvider()
    llm.queue(
        "score_rubric",
        json.dumps(
            {
                "criteria": [
                    {"criterion": "correctness", "awarded": 0, "rationale": "wrong"},
                    {"criterion": "completeness", "awarded": 0.3, "rationale": "complete"},
                    {"criterion": "reasoning", "awarded": 0.2, "rationale": "clear"},
                ],
                "confidence": 0.9,
            }
        ),
    )
    result = await RubricLLMStrategy(llm).score(question(max_score=1.0), "x", rubric_spec())
    assert result.total_awarded == 0.0  # knockout on correctness


def test_apply_rubric_rules_weighting():
    from app.schemas import CriteriaScore

    criteria = [
        CriteriaScore(criterion="correctness", awarded=0.5, max_score=0.5),
        CriteriaScore(criterion="completeness", awarded=0.15, max_score=0.3),
        CriteriaScore(criterion="reasoning", awarded=0.1, max_score=0.2),
    ]
    total, knockout = apply_rubric_rules(criteria, rubric_spec())
    assert total == 0.75
    assert knockout is False


def test_apply_rubric_rules_knockout():
    from app.schemas import CriteriaScore

    criteria = [
        CriteriaScore(criterion="correctness", awarded=0.0, max_score=0.5),
        CriteriaScore(criterion="completeness", awarded=0.3, max_score=0.3),
    ]
    total, knockout = apply_rubric_rules(criteria, rubric_spec())
    assert total == 0.0
    assert knockout is True


async def test_rubric_awarded_clamped_to_max():
    llm = FakeProvider()
    llm.queue(
        "score_rubric",
        json.dumps(
            {
                "criteria": [{"criterion": "correctness", "awarded": 99, "rationale": "x"}],
                "confidence": 0.9,
            }
        ),
    )
    result = await RubricLLMStrategy(llm).score(question(), "x", rubric_spec())
    assert result.criteria[0].awarded <= result.criteria[0].max_score


async def test_hybrid_uses_deterministic_when_key_present():
    strategy = HybridScoringStrategy(FakeProvider())
    result = await strategy.score(question(), "56")
    # numeric-looking key -> numeric strategy (deterministic, no LLM)
    assert result.strategy == "numeric"


async def test_hybrid_uses_rubric_without_key():
    strategy = HybridScoringStrategy(FakeProvider())
    q = question(answer_key=None)
    result = await strategy.score(q, "essay answer", rubric_spec())
    assert result.strategy == "rubric_llm"


async def test_hybrid_numeric_key_detection():
    strategy = HybridScoringStrategy(FakeProvider())
    result = await strategy.score(question(answer_key="56"), "56")
    assert result.strategy == "numeric"


def test_strategy_factory_flags():
    llm = FakeProvider()
    assert build_scoring("exact", llm=llm).name == "exact"
    assert build_scoring("numeric", llm=llm).name == "numeric"
    assert build_scoring("rubric_llm", llm=llm).name == "rubric_llm"
    assert build_scoring("hybrid", llm=llm).name == "hybrid"
    with pytest.raises(ValueError):
        build_scoring("nonsense", llm=llm)
