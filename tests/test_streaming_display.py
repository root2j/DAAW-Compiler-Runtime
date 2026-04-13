"""Tests for the progressive JSON pretty-printer used by the streaming UIs."""

from __future__ import annotations

from daaw.ui._streaming_display import prettify_partial_json


class TestPrettifyPartialJson:
    def test_empty(self):
        text, lang = prettify_partial_json("")
        assert text == ""
        assert lang == "text"

    def test_valid_json_is_indented(self):
        src = '{"name":"plan","tasks":[{"id":"t1","name":"first"}]}'
        text, lang = prettify_partial_json(src)
        assert lang == "json"
        # Indented output contains newlines that the raw didn't have.
        assert "\n" in text
        assert '"name": "plan"' in text

    def test_partial_json_with_unclosed_brackets_gets_repaired(self):
        """A mid-stream partial should still pretty-print via the
        bracket-balancing repair."""
        src = '{"name":"plan","tasks":[{"id":"t1"'
        text, lang = prettify_partial_json(src)
        # Repaired form is parseable.
        assert lang == "json"
        assert "plan" in text

    def test_garbage_falls_back_to_soft_broken_text(self):
        # Genuine malformed stream: stray @@ outside strings makes it
        # unparseable even after bracket-balancing.
        src = '{"a":"x"@@not json,"b":"y","c":"z"'
        text, lang = prettify_partial_json(src)
        assert lang == "text"
        # Soft break before every ," pair — there are two in this input.
        assert text.count("\n") >= 2

    def test_tail_truncation_does_not_break_json(self):
        """Accumulator longer than max_chars should still produce json
        OR soft text (never raise)."""
        body = '{"k":"' + ("x" * 5000) + '"}'
        text, lang = prettify_partial_json(body, max_chars=500)
        assert lang in ("json", "text")
        assert len(text) <= 1000  # tail cap + some indent overhead

    def test_trailing_comma_is_repaired(self):
        src = '{"a": 1, "b": 2,}'
        text, lang = prettify_partial_json(src)
        assert lang == "json"
        assert '"a": 1' in text
