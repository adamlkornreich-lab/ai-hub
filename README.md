# ai-hub

One small Python interface to OpenAI, Anthropic (Claude) and Google Gemini.

```bash
pip install -e .            # or: pip install -r requirements.txt
export OPENAI_API_KEY=... ANTHROPIC_API_KEY=... GEMINI_API_KEY=...

python -m ai_hub "Explain recursion in one line"              # all configured providers, in parallel
python -m ai_hub -p anthropic "Explain recursion in one line"
python -m ai_hub -p gemini -m gemini-2.5-pro -s "Be terse" "What is a monad?"
echo "Summarize: ..." | python -m ai_hub -p openai            # prompt from stdin
python -m ai_hub --list                                       # providers, default models, which are configured
```

After `pip install -e .` the same CLI is available as `ai-hub`.

## Python

```python
from ai_hub import ask, ask_all

ask("anthropic", "Hello", system="Reply in French")
ask_all("Hello")   # {"openai": "...", "anthropic": "...", "gemini": "..."}
```

## Default models

| Provider  | Default            | Override env var         |
|-----------|--------------------|--------------------------|
| openai    | `gpt-4o-mini`      | `AI_HUB_OPENAI_MODEL`    |
| anthropic | `claude-opus-5-5`  | `AI_HUB_ANTHROPIC_MODEL` |
| gemini    | `gemini-2.5-flash` | `AI_HUB_GEMINI_MODEL`    |

Only providers with an API key set are queried. Never commit keys.

## Tests

```bash
pip install -e '.[dev]' && pytest
```
