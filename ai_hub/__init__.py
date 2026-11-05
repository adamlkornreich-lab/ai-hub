"""Unified client for OpenAI, Anthropic and Gemini."""

import os

import requests

__all__ = ["PROVIDERS", "ask", "ask_all", "configured", "default_model"]

TIMEOUT = 120

# provider -> (API key env var, default model). Override the model per provider
# with AI_HUB_<PROVIDER>_MODEL, e.g. AI_HUB_ANTHROPIC_MODEL=claude-sonnet-5-5.
PROVIDERS = {
    "openai": ("OPENAI_API_KEY", "gpt-4o-mini"),
    "anthropic": ("ANTHROPIC_API_KEY", "claude-opus-5-5"),
    "gemini": ("GEMINI_API_KEY", "gemini-2.5-flash"),
}


def _check(provider: str) -> None:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}; choose from {list(PROVIDERS)}")


def _key(provider: str) -> str:
    env = PROVIDERS[provider][0]
    key = os.environ.get(env)
    if not key:
        raise RuntimeError(f"Set {env} to use {provider}")
    return key


def default_model(provider: str) -> str:
    """Model used when none is given: env override, else the built-in default."""
    _check(provider)
    return os.environ.get(f"AI_HUB_{provider.upper()}_MODEL") or PROVIDERS[provider][1]


def configured() -> list[str]:
    """Providers that have an API key set."""
    return [p for p, (env, _) in PROVIDERS.items() if os.environ.get(env)]


def _post(url: str, headers: dict, body: dict) -> dict:
    r = requests.post(url, headers=headers, json=body, timeout=TIMEOUT)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
    return r.json()


def _openai(key: str, model: str, prompt: str, system: str | None) -> str:
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    data = _post(
        "https://api.openai.com/v1/chat/completions",
        {"Authorization": f"Bearer {key}"},
        {"model": model, "messages": messages},
    )
    return data["choices"][0]["message"].get("content") or ""


def _anthropic(key: str, model: str, prompt: str, system: str | None) -> str:
    body = {"model": model, "max_tokens": 16000,
            "messages": [{"role": "user", "content": prompt}]}
    if system:
        body["system"] = system
    data = _post(
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": key, "anthropic-version": "2023-06-01"},
        body,
    )
    if data.get("stop_reason") == "refusal":
        raise RuntimeError("request was declined (stop_reason: refusal)")
    # Current models can return thinking blocks before the text; keep only text.
    return "".join(b["text"] for b in data.get("content", []) if b.get("type") == "text")


def _gemini(key: str, model: str, prompt: str, system: str | None) -> str:
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    data = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {"x-goog-api-key": key},
        body,
    )
    candidates = data.get("candidates") or []
    if not candidates:
        reason = data.get("promptFeedback", {}).get("blockReason", "no candidates")
        raise RuntimeError(f"no reply from gemini ({reason})")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


_CALLS = {"openai": _openai, "anthropic": _anthropic, "gemini": _gemini}


def ask(provider: str, prompt: str, model: str | None = None,
        system: str | None = None) -> str:
    """Send a prompt to the given provider and return the text reply."""
    _check(provider)
    return _CALLS[provider](_key(provider), model or default_model(provider), prompt, system)


def ask_all(prompt: str, system: str | None = None) -> dict[str, str]:
    """Query every provider that has an API key configured, in parallel."""
    from concurrent.futures import ThreadPoolExecutor

    providers = configured()
    if not providers:
        return {}
    with ThreadPoolExecutor(max_workers=len(providers)) as pool:
        futures = {p: pool.submit(ask, p, prompt, None, system) for p in providers}
    out = {}
    for p, f in futures.items():
        try:
            out[p] = f.result()
        except Exception as e:  # keep going if one provider fails
            out[p] = f"error: {e}"
    return out
