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

# Analyze results (individual model)
python harness/analyze.py --results --latest

# Analyze comparisons (Elo ratings, win rates, etc.)
python harness/analyze.py --compare
```

## Orchestrator Mode

**You are an orchestrator in this project.** Default to spinning up parallel subagents for independent work items rather than doing everything sequentially yourself. Use the Task tool to launch agents for feature branches, PRs, and review loops. Only do work directly when subagents can't (e.g., permission boundaries, merge conflict resolution).

Key rules:
- **Parallelize**: Launch multiple agents concurrently for independent tasks (e.g., 4 harness flags can be built simultaneously)
- **Wave execution**: Group work into dependency waves — launch all items in a wave together, wait, then next wave
- **Git isolation**: Each subagent gets its own branch. Never let two agents share a working directory
- **PR review loops**: Every subagent that produces a PR MUST run the `pr-review-loop` skill to get Gemini Code Assist reviews (4-5 cycles minimum). Use `commit-and-push.sh`, `get-review-comments.sh`, `reply-to-comment.sh`, and `trigger-review.sh` from `~/.claude/skills/pr-review-loop/scripts/`
- **Gemini peer review for planning**: Before implementing non-trivial features, invoke the `gemini-peer-review` skill to get a second opinion on architecture and approach. Do this during the planning phase, not after code is written
- **Beads tracking**: Use `bd` for multi-session work; TodoWrite for single-session execution steps
- **Permission boundaries**: Keep worktrees inside project directory (not `/tmp/`), or build directly when agents hit walls

## Key Fixes Applied

- GPT-5.1: Uses `max_completion_tokens=8000` (reasoning tokens issue)
- Gemini: Uses `max_output_tokens=8000` (truncation issue)
