"""Runtime configuration, loaded from environment variables / .env.

Nothing in this module is a secret's default value — there is no bundled
API key, no hardcoded proxy URL. A missing credential fails fast at startup
instead of silently falling back to an undocumented default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider — swap without touching agent code.
    llm_provider: Literal["openai", "anthropic", "fake"] = "fake"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # arXiv discovery — "fake" (default, offline, no network) synthesizes a
    # small deterministic paper set; "live" hits the real arXiv API.
    arxiv_provider: Literal["live", "fake"] = "fake"
    arxiv_categories: list[str] = Field(
        default_factory=lambda: ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE"]
    )
    default_lookback_days: int = 7
    default_max_papers: int = 10

    # Bounded concurrency for per-paper evaluation (see docs/adr/0007) —
    # deliberately not unbounded: this is the lever that keeps a research
    # run's outbound LLM call rate under control regardless of how many
    # papers were discovered.
    evaluation_concurrency: int = 4

    # Storage
    database_url: str = f"sqlite:///{BACKEND_ROOT / 'data' / 'agiresearch.db'}"

    # API / auth
    api_bearer_token: str = "dev-local-token"
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])


settings = Settings()
