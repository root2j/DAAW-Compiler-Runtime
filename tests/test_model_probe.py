"""Tests for the gateway model-compatibility probe."""

from __future__ import annotations

import asyncio

import pytest

from daaw.llm.model_probe import (
    ProbeResult,
    _classify,
    clear_probe_cache,
    get_cached_probe,
    probe_model,
)


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_probe_cache()
    yield
    clear_probe_cache()


class TestClassifier:
    def test_valid_json(self):
        assert _classify('{"ok": true}') == "VALID-JSON"

    def test_valid_json_with_fence(self):
        assert _classify('```json\n{"ok": true}\n```') == "VALID-JSON"

    def test_json_broken(self):
        assert _classify('{"ok": true,,}') == "JSON-BROKEN"

    def test_token_salad_unused(self):
        assert _classify("<unused42><unused1><pad>") == "TOKEN-SALAD"

    def test_token_salad_tool_marker(self):
        assert _classify("<tool|>hello<tool_response|>") == "TOKEN-SALAD"

    def test_empty(self):
        assert _classify("") == "EMPTY"
        assert _classify("   \n  ") == "EMPTY"

    def test_prose(self):
        assert _classify("Sure, here you go: hello world.") == "PROSE"


class TestProbeModel:
    def _make_client(self, *, status: int = 200, content: str = '{"ok":true,"n":1}'):
        """Factory for a fake httpx.AsyncClient serving a fixed response."""

        class FakeResp:
            def __init__(self, s, c):
                self.status_code = s
                self._content = c
                self.text = str(c)[:500]

            def json(self):
                return {
                    "choices": [{"message": {"content": self._content}}],
                    "usage": {},
                }

        class FakeClient:
            def __init__(self, *a, **kw):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return None
            async def post(self, *a, **kw):
                return FakeResp(status, content)

        return FakeClient

    def test_valid_response_marks_usable(self, monkeypatch):
        import httpx
        monkeypatch.setattr(httpx, "AsyncClient",
                            self._make_client(content='{"ok":true,"n":1}'))
        r = run(probe_model("http://localhost:11434/v1", "any-model"))
        assert r.is_usable
        assert r.classification == "VALID-JSON"
        assert r.badge == "OK"

    def test_token_salad_not_usable(self, monkeypatch):
        import httpx
        monkeypatch.setattr(httpx, "AsyncClient",
                            self._make_client(content="<unused1><pad><tool|>"))
        r = run(probe_model("http://localhost:11434/v1", "gemma4:e4b"))
        assert not r.is_usable
        assert r.classification == "TOKEN-SALAD"
        assert r.badge == "DRIFT"

    def test_http_500_is_error(self, monkeypatch):
        import httpx
        monkeypatch.setattr(
            httpx, "AsyncClient",
            self._make_client(status=500, content="backend crash"),
        )
        r = run(probe_model("http://localhost:11434/v1", "m"))
        assert not r.is_usable
        assert r.classification == "ERROR"

    def test_network_exception_handled(self, monkeypatch):
        import httpx

        class BoomClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return None
            async def post(self, *a, **kw):
                raise httpx.ConnectError("cannot connect")

        monkeypatch.setattr(httpx, "AsyncClient", BoomClient)
        r = run(probe_model("http://unreachable:1/v1", "m"))
        assert r.classification == "ERROR"
        assert "ConnectError" in r.preview

    def test_cache_hit_on_second_call(self, monkeypatch):
        import httpx
        calls = {"n": 0}

        class CountingClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return None
            async def post(self, *a, **kw):
                calls["n"] += 1

                class R:
                    status_code = 200
                    def json(self_inner):
                        return {"choices": [{"message": {"content": '{"ok":true}'}}]}
                return R()

        monkeypatch.setattr(httpx, "AsyncClient", CountingClient)
        run(probe_model("http://x:1/v1", "m"))
        run(probe_model("http://x:1/v1", "m"))
        assert calls["n"] == 1

    def test_force_bypasses_cache(self, monkeypatch):
        import httpx
        calls = {"n": 0}

        class CountingClient:
            def __init__(self, *a, **kw): pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return None
            async def post(self, *a, **kw):
                calls["n"] += 1

                class R:
                    status_code = 200
                    def json(self_inner):
                        return {"choices": [{"message": {"content": '{"ok":true}'}}]}
                return R()

        monkeypatch.setattr(httpx, "AsyncClient", CountingClient)
        run(probe_model("http://x:1/v1", "m"))
        run(probe_model("http://x:1/v1", "m", force=True))
        assert calls["n"] == 2

    def test_get_cached_probe_returns_none_when_empty(self):
        assert get_cached_probe("http://nope/v1", "unknown") is None


class TestProbeResult:
    def test_badge_for_each_class(self):
        r = ProbeResult(model="m", gateway_url="u",
                        classification="VALID-JSON", elapsed_seconds=1.0,
                        preview="", is_usable=True)
        assert r.badge == "OK"
        r.classification = "TOKEN-SALAD"; assert r.badge == "DRIFT"
        r.classification = "EMPTY";        assert r.badge == "EMPTY"
        r.classification = "JSON-BROKEN";  assert r.badge == "FLAKY"
        r.classification = "ERROR";        assert r.badge == "ERROR"
