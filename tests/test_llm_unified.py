"""Tests for the unified LLM client and provider routing."""

import asyncio
import os

import pytest

from conftest import skip_on_rate_limit
from daaw.config import AppConfig
from daaw.llm.base import LLMMessage
from daaw.llm.unified import UnifiedLLMClient


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestUnifiedLLMClientRouting:
    def test_no_keys_no_providers(self):
        config = AppConfig()  # all keys empty
        client = UnifiedLLMClient(config)
        assert client.available_providers() == []

    def test_only_configured_providers(self):
        config = AppConfig(groq_api_key="fake_key")
        client = UnifiedLLMClient(config)
        assert "groq" in client.available_providers()
        assert "openai" not in client.available_providers()

    def test_unavailable_provider_raises(self):
        config = AppConfig()
        client = UnifiedLLMClient(config)
        with pytest.raises(ValueError, match="not available"):
            run(client.chat("groq", [LLMMessage(role="user", content="hi")]))

    def test_error_message_lists_available(self):
        config = AppConfig(groq_api_key="fake")
        client = UnifiedLLMClient(config)
        try:
            run(client.chat("openai", [LLMMessage(role="user", content="hi")]))
        except ValueError as e:
            assert "groq" in str(e)


class TestUnifiedLLMClientGroq:
    """Integration tests using real Groq API."""

    @pytest.fixture(autouse=True)
    def skip_without_key(self):
        from dotenv import load_dotenv
        load_dotenv()
        if not os.environ.get("GROQ_API_KEY"):
            pytest.skip("GROQ_API_KEY not set")

    @skip_on_rate_limit
    def test_groq_available(self, app_config):
        client = UnifiedLLMClient(app_config)
        assert "groq" in client.available_providers()

    @skip_on_rate_limit
    def test_groq_simple_chat(self, app_config):
        client = UnifiedLLMClient(app_config)
        messages = [LLMMessage(role="user", content="Reply with exactly: HELLO")]
        resp = run(client.chat("groq", messages, max_tokens=50))
        assert resp.content  # non-empty
        assert resp.model  # model name present
        assert "HELLO" in resp.content.upper()

    @skip_on_rate_limit
    def test_groq_with_system_prompt(self, app_config):
        """System prompt should be accepted and influence the response."""
        client = UnifiedLLMClient(app_config)
        messages = [
            LLMMessage(role="system", content="You are a helpful assistant."),
            LLMMessage(role="user", content="Say hi"),
        ]
        resp = run(client.chat("groq", messages, max_tokens=100))
        assert resp.content  # non-empty response means system prompt was accepted

    @skip_on_rate_limit
    def test_groq_json_mode(self, app_config):
        """Test JSON response_format."""
        client = UnifiedLLMClient(app_config)
        messages = [
            LLMMessage(
                role="user",
                content='Return a JSON object with key "status" set to "ok". No other text.',
            ),
        ]
        resp = run(
            client.chat(
                "groq", messages,
                max_tokens=100,
                response_format={"type": "json_object"},
            )
        )
        import json
        data = json.loads(resp.content)
        assert data["status"] == "ok"

    @skip_on_rate_limit
    def test_groq_temperature(self, app_config):
        """Low temperature should give consistent results."""
        client = UnifiedLLMClient(app_config)
        messages = [LLMMessage(role="user", content="What is 2+2? Reply with just the number.")]
        resp = run(client.chat("groq", messages, temperature=0.0, max_tokens=10))
        assert "4" in resp.content

    @skip_on_rate_limit
    def test_groq_usage_stats(self, app_config):
        """Response should include token usage info."""
        client = UnifiedLLMClient(app_config)
        messages = [LLMMessage(role="user", content="Say hi")]
        resp = run(client.chat("groq", messages, max_tokens=10))
        assert resp.usage.get("total_tokens", 0) > 0
