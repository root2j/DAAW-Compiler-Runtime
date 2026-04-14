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
    # Generic OpenAI-compatible gateway (LiteLLM, Ollama, vLLM, etc.)
    gateway_url: str = ""
    gateway_token: str = ""
    gateway_model: str = "default"
    # Webhook notifications (Discord, Slack, or generic HTTP)
    notify_webhook_url: str = ""
    notify_webhook_type: str = "generic"
    artifact_store_dir: str = ".daaw_store"
    max_critic_retries: int = 3
    max_planner_retries: int = 3
    circuit_breaker_threshold: int = 3
    # Split-provider: use a stronger model for the compiler (planner) while
    # keeping a cheaper/faster model for task execution. When set, these
    # override the CLI/UI provider+model for the compile phase only.
    #   DAAW_COMPILER_PROVIDER=groq
    #   DAAW_COMPILER_MODEL=llama-3.3-70b-versatile
    compiler_provider: str = ""
    compiler_model: str = ""


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
            gateway_url=os.environ.get("GATEWAY_URL", ""),
            gateway_token=os.environ.get("GATEWAY_TOKEN", ""),
            gateway_model=os.environ.get("GATEWAY_MODEL", "default"),
            notify_webhook_url=os.environ.get("NOTIFY_WEBHOOK_URL", ""),
            notify_webhook_type=os.environ.get("NOTIFY_WEBHOOK_TYPE", "generic"),
            artifact_store_dir=os.environ.get("DAAW_STORE_DIR", ".daaw_store"),
            compiler_provider=os.environ.get("DAAW_COMPILER_PROVIDER", ""),
            compiler_model=os.environ.get("DAAW_COMPILER_MODEL", ""),
        )
    return _config


def reset_config() -> None:
    """Clear the cached config so the next get_config() re-reads env vars.

    Useful for long-lived processes (Streamlit) after .env changes.
    """
    global _config
    _config = None
    load_dotenv(override=True)
