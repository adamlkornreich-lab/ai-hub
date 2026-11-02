# ai-hub

One small Python interface to OpenAI, Anthropic (Claude) and Google Gemini.

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=... ANTHROPIC_API_KEY=... GEMINI_API_KEY=...
python -m ai_hub "Explain recursion in one line"            # all configured providers
python -m ai_hub -p anthropic "Explain recursion in one line"
```

Only providers with an API key set are queried. Never commit keys.
