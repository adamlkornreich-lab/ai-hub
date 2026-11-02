"""Unified client for OpenAI, Anthropic and Gemini."""

import os

import requests

PROVIDERS = {
    "openai": ("OPENAI_API_KEY", "gpt-4o-mini"),
    "anthropic": ("ANTHROPIC_API_KEY", "claude-3-5-haiku-latest"),
    "gemini": ("GEMINI_API_KEY", "gemini-1.5-flash"),
}


def _key(provider: str) -> str:
    key = os.environ.get(PROVIDERS[provider][0])
    if not key:
        raise RuntimeError(f"Set {PROVIDERS[provider][0]} to use {provider}")
    return key


def ask(provider: str, prompt: str, model: str | None = None) -> str:
    """Send a prompt to the given provider and return the text reply."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}; choose from {list(PROVIDERS)}")
    model = model or PROVIDERS[provider][1]
    key = _key(provider)
    if provider == "openai":
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    if provider == "anthropic":
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            json={"model": model, "max_tokens": 1024,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key},
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def ask_all(prompt: str) -> dict[str, str]:
    """Query every provider that has an API key configured."""
    out = {}
    for p, (env, _) in PROVIDERS.items():
        if os.environ.get(env):
            try:
                out[p] = ask(p, prompt)
            except Exception as e:  # keep going if one provider fails
                out[p] = f"error: {e}"
    return out
