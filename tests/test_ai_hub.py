from unittest import mock

import pytest

import ai_hub
from ai_hub.__main__ import main


class Resp:
    def __init__(self, data, status=200):
        self._data, self.status_code, self.ok = data, status, status < 400
        self.text = str(data)

    def json(self):
        return self._data


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for env, _ in ai_hub.PROVIDERS.values():
        monkeypatch.delenv(env, raising=False)
    for p in ai_hub.PROVIDERS:
        monkeypatch.delenv(f"AI_HUB_{p.upper()}_MODEL", raising=False)


def test_unknown_provider():
    with pytest.raises(ValueError):
        ai_hub.ask("nope", "hi")


def test_missing_key():
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        ai_hub.ask("anthropic", "hi")


def test_anthropic_skips_thinking_blocks(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    data = {"stop_reason": "end_turn", "content": [
        {"type": "thinking", "thinking": ""}, {"type": "text", "text": "hello"}]}
    with mock.patch("requests.post", return_value=Resp(data)) as post:
        assert ai_hub.ask("anthropic", "hi", system="be brief") == "hello"
    body = post.call_args.kwargs["json"]
    assert body["model"] == "claude-opus-5-5" and body["system"] == "be brief"


def test_anthropic_refusal(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    with mock.patch("requests.post", return_value=Resp({"stop_reason": "refusal", "content": []})):
        with pytest.raises(RuntimeError, match="refusal"):
            ai_hub.ask("anthropic", "hi")


def test_openai_and_model_env_override(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("AI_HUB_OPENAI_MODEL", "gpt-x")
    data = {"choices": [{"message": {"content": "yo"}}]}
    with mock.patch("requests.post", return_value=Resp(data)) as post:
        assert ai_hub.ask("openai", "hi") == "yo"
    assert post.call_args.kwargs["json"]["model"] == "gpt-x"


def test_gemini_joins_parts_and_skips_thoughts(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    data = {"candidates": [{"content": {"parts": [
        {"text": "hmm", "thought": True}, {"text": "a"}, {"text": "b"}]}}]}
    with mock.patch("requests.post", return_value=Resp(data)):
        assert ai_hub.ask("gemini", "hi") == "ab"


def test_http_error_includes_body(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    with mock.patch("requests.post", return_value=Resp({"error": "bad model"}, 404)):
        with pytest.raises(RuntimeError, match="HTTP 404.*bad model"):
            ai_hub.ask("gemini", "hi")


def test_ask_all_only_configured_and_isolates_errors(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    def fake(url, **kw):
        if "openai" in url:
            return Resp({"choices": [{"message": {"content": "ok"}}]})
        return Resp({}, 500)

    with mock.patch("requests.post", side_effect=fake):
        out = ai_hub.ask_all("hi")
    assert out["openai"] == "ok"
    assert out["gemini"].startswith("error:")
    assert "anthropic" not in out


def test_cli_no_providers(capsys):
    assert main(["hi"]) == 1
    assert "No providers configured" in capsys.readouterr().err


def test_cli_model_requires_provider():
    with pytest.raises(SystemExit):
        main(["-m", "x", "hi"])


def test_cli_list(capsys, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "anthropic" in out and "configured" in out
