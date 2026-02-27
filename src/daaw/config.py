"""Centralized configuration — env vars, defaults, singleton factory."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class LLMProviderConfig:
    provider: str
    model: str
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass(frozen=True)
class AppConfig:
    groq_api_key: str = ""
    gemini_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    artifact_store_dir: str = ".daaw_store"
    max_critic_retries: int = 3
    max_planner_retries: int = 3
    circuit_breaker_threshold: int = 3


_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Singleton factory — reads env vars once, returns frozen config."""
    global _config
    if _config is None:
        _config = AppConfig(
            groq_api_key=os.environ.get("GROQ_API_KEY", ""),
            gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            artifact_store_dir=os.environ.get("DAAW_STORE_DIR", ".daaw_store"),
        )
    return _config
