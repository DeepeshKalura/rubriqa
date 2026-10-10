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
    llm_provider: Literal["minimax", "fake"] = "minimax"

    # --- reader feature flags
    reader_mode: Literal["pdf_text", "tesseract", "minimax_vision", "hybrid"] = "hybrid"
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

    # --- sandbox (2-way OCR comparison)
    sandbox_default_provider: Literal["vision", "ocr_llm"] = "vision"

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
