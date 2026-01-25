# CLAUDE.md — AI Judgment Battery

## Model Policy

**ALWAYS use the latest models. No exceptions.**

| Provider | Default Model | Notes |
|----------|---------------|-------|
| Anthropic | claude-opus-4-5-20251101 | Claude Opus 4.5 |
| OpenAI | gpt-5.1 | GPT-5.1 (use 5.2 when available) |
| Gemini | gemini-2.5-flash | Gemini 2.5 Flash |

If I want a different model, I will explicitly tell you.

## Project Commands

```bash
# Run battery
python harness/run_battery.py --model <model> --workers 5

# Compare models
python harness/compare.py --models claude-opus gpt-5.1 gemini-2.5-flash

# Analyze results
python harness/analyze.py --latest
```

## Key Fixes Applied

- GPT-5.1: Uses `max_completion_tokens=8000` (reasoning tokens issue)
- Gemini: Uses `max_output_tokens=8000` (truncation issue)
