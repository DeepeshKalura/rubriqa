"""Configuration. All secrets come from environment variables only."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- database
    database_url: str = "postgresql+asyncpg://entryeval:entryeval@localhost:5435/entry_eval"
    run_migrations: bool = True
    db_pool_size: int = 5
    db_max_overflow: int = 5
    db_echo: bool = False

    # --- storage
    storage_root: Path = Path("./data")
    max_upload_mb: int = 50

    # --- MiniMax / LLM
    minimax_api_key: str = ""
    llm_api_key: str = ""
    minimax_base_url: str = "https://api.minimax.io/v1"
    minimax_text_model: str = "MiniMax-M3"
    minimax_vision_model: str = "MiniMax-M3"
    minimax_timeout_seconds: int = 120
    minimax_max_retries: int = 3
    minimax_max_output_tokens: int = 4096
    minimax_disable_thinking: bool = True
    llm_provider: Literal["minimax", "gemini", "fake"] = "minimax"

    # --- Google Gemini (alternative OCR provider; free tier, no card needed)
    # Used for handwriting transcription only. Rubric scoring stays on the
    # configured text provider. Pricing verified 2026-10-14:
    # https://ai.google.dev/gemini-api/docs/pricing
    google_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    # Pinned, not "latest": Google retires ids without warning (2.5-flash-lite
    # was already closed to new users on 2026-10-14, the day this was written).
    # 3.1-flash-lite is the cheapest id that still accepts
    # `thinkingConfig.thinkingBudget: 0`; 3.5-flash-lite rejects it with a 400.
    gemini_model: str = "gemini-3.1-flash-lite"
    gemini_timeout_seconds: int = 120
    gemini_max_retries: int = 3
    gemini_max_output_tokens: int = 4096
    # Transcription is a mechanical task; thinking tokens are billed and only
    # eat the output budget. Mirrors MINIMAX_DISABLE_THINKING.
    gemini_disable_thinking: bool = True

    # --- cost guardrails
    # Hard ceiling for one ingestion run. 0 disables the cap. A run that would
    # exceed it stops *before* the next billable call rather than after.
    llm_spend_cap_usd: float = 0.0

    # --- reader feature flags
    reader_mode: Literal[
        "pdf_text", "tesseract", "minimax_vision", "gemini_vision", "hybrid"
    ] = "hybrid"
    hybrid_reader_chain: str = "pdf_text,tesseract,minimax_vision"
    hybrid_accept_confidence: float = 0.75
    hybrid_escalate_below: float = 0.60
    tesseract_langs: str = "eng+hin"
    tesseract_min_word_confidence: int = 45
    vision_max_image_side: int = 2200
    low_confidence_threshold: float = 0.55
    reading_version: str = "v1"

    # --- entry evaluation (the only flow)
    eval_batch_size: int = 250
    eval_batch_size_min: int = 200
    eval_batch_size_max: int = 300
    eval_review_sample_fraction: float = 0.10
    eval_known_themes: str = (
        "ai_education,smart_agriculture,health_wellbeing,disaster_management,"
        "environment_sustainability,rural_innovation,community_safety"
    )
    cleaning_mode: Literal["rules", "llm"] = "rules"
    drive_download_chunk_bytes: int = 65536
    llm_concurrency: int = 3

    # --- google drive (local mirror; the only supported provider in dev phase)
    google_drive_enabled: bool = False
    google_drive_provider: Literal["local"] = "local"
    google_drive_local_root: Path = Path("./data/drive_mirror")

    # --- sandbox (3-way OCR comparison: vision / tesseract / fusion)
    sandbox_default_provider: Literal["vision", "ocr_llm"] = "vision"
    # When the tesseract transcript is suspiciously long compared to the
    # vision transcript, tesseract is hallucinating noise (English script on
    # Devanagari pages, double-length, etc.). Skip the fusion LLM call and use
    # the vision text as the hybrid output. See docs/sandbox/fusion_gate.md
    # for the data this number came from.
    fusion_max_tess_to_vision_ratio: float = 1.5

    # --- app
    app_host: str = "127.0.0.1"
    log_level: str = "INFO"
    log_json: bool = False
    app_version: str = "0.2.0"

    # --- auth (feature flag, default off; see roadmap item 2)
    auth_enabled: bool = False
    auth_admin_user: str = "admin"
    auth_admin_password_hash: str = ""
    session_secret: str = ""

    @property
    def api_key(self) -> str:
        """Accept either MINIMAX_API_KEY or LLM_API_KEY."""
        return (self.minimax_api_key or self.llm_api_key or "").strip()

    @property
    def gemini_key(self) -> str:
        """Google AI Studio key. Never logged; redacted by app.logging."""
        return (self.google_api_key or "").strip()

    @property
    def storage_originals(self) -> Path:
        return self.storage_root / "originals"

    @property
    def storage_derived(self) -> Path:
        return self.storage_root / "derived"

    @property
    def storage_exports(self) -> Path:
        return self.storage_root / "exports"

    @property
    def uploads_dir(self) -> Path:
        return self.storage_root / "uploads"


@lru_cache
def get_settings() -> Settings:
    return Settings()
