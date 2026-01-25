#!/usr/bin/env python3
"""
AI Judgment Battery - Shared Utilities

Common functions and configurations used across the harness scripts.
"""

import json
from pathlib import Path
from typing import Optional

# Directory paths
RESULTS_DIR = Path(__file__).parent.parent / "results"
COMPARISONS_DIR = Path(__file__).parent.parent / "comparisons"
HUMAN_EVAL_DIR = Path(__file__).parent.parent / "human_evals"

# Model configurations - centralized mapping
MODEL_CONFIGS = {
    # Anthropic
    "claude-opus": {
        "provider": "anthropic",
        "model_id": "claude-opus-4-5-20251101",
    },
    "claude-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
    },
    "claude-haiku": {
        "provider": "anthropic",
        "model_id": "claude-3-5-haiku-20241022",
    },
    # OpenAI
    "gpt-5.1": {
        "provider": "openai",
        "model_id": "gpt-5.1",
    },
    "gpt-5.2": {
        "provider": "openai",
        "model_id": "gpt-5.2",
    },
    # Gemini
    "gemini-3-pro": {
        "provider": "gemini",
        "model_id": "gemini-3-pro-preview",
    },
    "gemini-3-flash": {
        "provider": "gemini",
        "model_id": "gemini-3-flash-preview",
    },
    "gemini-2.5-pro": {
        "provider": "gemini",
        "model_id": "gemini-2.5-pro",
    },
    "gemini-2.5-flash": {
        "provider": "gemini",
        "model_id": "gemini-2.5-flash",
    },
}

# Default judges for multi-judge comparison
DEFAULT_JUDGES = ["claude-opus", "gpt-5.1", "gemini-3-pro"]


def load_results_file(filepath: Path) -> dict:
    """Load a results JSON file."""
    with open(filepath) as f:
        return json.load(f)


def find_best_results_for_model(model_key: str) -> Optional[Path]:
    """Find the best (most complete) results file for a model."""
    pattern = f"*{model_key}*.json"
    files = list(RESULTS_DIR.glob(pattern))

    if not files:
        return None

    best_file = None
    best_count = 0

    for f in files:
        try:
            data = load_results_file(f)
            responses = data.get("responses", [])
            valid = sum(
                1 for r in responses if r.get("response") and not r.get("error")
            )
            if valid > best_count:
                best_count = valid
                best_file = f
        except Exception:
            continue

    return best_file


def get_response_for_dilemma(results: dict, dilemma_id: str) -> Optional[dict]:
    """Get the response for a specific dilemma from results."""
    for r in results.get("responses", []):
        if r.get("dilemma_id") == dilemma_id:
            if r.get("response") and not r.get("error"):
                return r
    return None


def get_judge_config(judge_model: str) -> tuple[str, str]:
    """Get provider and model_id for a judge model.

    Args:
        judge_model: Either a key like "claude-opus" or a full model ID

    Returns:
        Tuple of (provider, model_id)
    """
    # Check if it's a known key
    if judge_model in MODEL_CONFIGS:
        config = MODEL_CONFIGS[judge_model]
        return config["provider"], config["model_id"]

    # Infer from model name
    if judge_model.startswith("claude"):
        return "anthropic", judge_model
    elif (
        judge_model.startswith("gpt")
        or judge_model.startswith("o1")
        or judge_model.startswith("o3")
    ):
        return "openai", judge_model
    elif judge_model.startswith("gemini"):
        return "gemini", judge_model
    else:
        # Default to anthropic
        return "anthropic", judge_model


def normalize_judge_key(judge_model: str) -> str:
    """Normalize a judge model name to a simple key for comparison.

    Args:
        judge_model: Full model ID like "claude-opus-4-5-20251101"

    Returns:
        Simplified key like "claude-opus"
    """
    judge_lower = judge_model.lower()

    if "opus" in judge_lower:
        return "claude-opus"
    elif "sonnet" in judge_lower:
        return "claude-sonnet"
    elif "haiku" in judge_lower:
        return "claude-haiku"
    elif "gpt-5.1" in judge_lower or judge_lower == "gpt-5.1":
        return "gpt-5.1"
    elif "gpt-5.2" in judge_lower or judge_lower == "gpt-5.2":
        return "gpt-5.2"
    elif "gemini-3-pro" in judge_lower or "gemini-3-pro-preview" in judge_lower:
        return "gemini-3-pro"
    elif "gemini-3-flash" in judge_lower:
        return "gemini-3-flash"
    elif "gemini-2.5-pro" in judge_lower:
        return "gemini-2.5-pro"
    elif "gemini-2.5-flash" in judge_lower:
        return "gemini-2.5-flash"
    else:
        return judge_model
