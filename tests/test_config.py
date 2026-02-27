"""Tests for config module."""

import os

import pytest

from daaw.config import AppConfig, get_config


class TestAppConfig:
    def test_frozen(self):
        config = AppConfig()
        with pytest.raises(Exception):
            config.groq_api_key = "new_key"

    def test_defaults(self):
        config = AppConfig()
        assert config.artifact_store_dir == ".daaw_store"
        assert config.max_critic_retries == 3
        assert config.max_planner_retries == 3
        assert config.circuit_breaker_threshold == 3

    def test_custom_values(self):
        config = AppConfig(
            groq_api_key="gk",
            gemini_api_key="gem",
            artifact_store_dir="/tmp/custom",
            max_critic_retries=5,
        )
        assert config.groq_api_key == "gk"
        assert config.gemini_api_key == "gem"
        assert config.artifact_store_dir == "/tmp/custom"
        assert config.max_critic_retries == 5


class TestGetConfig:
    def test_returns_config(self):
        config = get_config()
        assert isinstance(config, AppConfig)

    def test_singleton(self):
        """get_config() should return the same instance."""
        c1 = get_config()
        c2 = get_config()
        assert c1 is c2

    def test_reads_gemini_key(self):
        """Gemini key should be populated from .env."""
        config = get_config()
        # User confirmed they have GEMINI_API_KEY in .env
        assert config.gemini_api_key != "", (
            "Expected GEMINI_API_KEY in .env — is .env loaded?"
        )
