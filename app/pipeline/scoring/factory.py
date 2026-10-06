"""Scoring strategy factory (feature flag: SCORING_STRATEGY)."""
from __future__ import annotations

from app.config import Settings, get_settings
from app.pipeline.scoring import (
    ExactMatchStrategy,
    HybridScoringStrategy,
    NumericStrategy,
    RubricLLMStrategy,
)


def build_scoring(strategy: str | None = None, *, llm=None, settings: Settings | None = None):
    settings = settings or get_settings()
    name = (strategy or settings.scoring_strategy).lower()
    if name == "exact":
        return ExactMatchStrategy(settings)
    if name == "numeric":
        return NumericStrategy(settings)
    if name == "rubric_llm":
        return RubricLLMStrategy(llm, settings)
    if name == "hybrid":
        return HybridScoringStrategy(llm, settings)
    raise ValueError(f"Unknown SCORING_STRATEGY '{name}'")
