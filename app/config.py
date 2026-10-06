"""Configuration. All secrets come from environment variables only."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- database
    database_url: str = "postgresql+asyncpg://exameval:exameval@localhost:5435/exam_eval"
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
    # MiniMax M-series: disable chain-of-thought for transcription/JSON tasks
    # (faster, cheaper, and prevents  thinking blocks eating the token budget)
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
    # Confidence threshold below which a page reading is marked low-confidence
    # (and which may trigger a vision repair pass).
    low_confidence_threshold: float = 0.55
    # Vision legibility threshold for the same purpose.
    low_legibility_threshold: float = 0.6
    repair_enabled: bool = True
    repair_auto_accept_min_legibility: float = 0.72
    reading_version: str = "v1"

    # --- scoring
    scoring_strategy: Literal["hybrid", "rubric_llm", "exact", "numeric"] = "hybrid"
    scoring_enable_feedback: bool = False

    # --- google forms
    google_forms_enabled: bool = False
    google_forms_provider: Literal["google", "mock"] = "mock"
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://127.0.0.1:8090/google/oauth/callback"
    app_secret_key: str = ""

    # --- jobs / worker
    worker_concurrency: int = 2
    ocr_threads: int = 2
    llm_concurrency: int = 3
    job_heartbeat_seconds: int = 15
    job_stuck_minutes: int = 10
    job_max_attempts: int = 3
    job_timeout_seconds: int = 900
    sweeper_interval_seconds: int = 60
    retry_backoff_base_seconds: float = 2.0

    # --- app
    app_host: str = "127.0.0.1"
    log_level: str = "INFO"
    log_json: bool = False
    app_version: str = "0.1.0"

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
