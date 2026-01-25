#!/usr/bin/env python3
"""
AI Judgment Battery - Human Validation Interface

Presents blinded model responses to human evaluators for ranking.
Compares human rankings to AI judge rankings to validate consistency.
"""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from harness.utils import (
    COMPARISONS_DIR,
    HUMAN_EVAL_DIR,
    find_best_results_for_model,
    get_response_for_dilemma,
    load_results_file,
)


def load_all_ai_rankings() -> dict[str, dict[str, list]]:
    """Load all AI rankings from comparison files into memory.

    Returns:
        Dict mapping dilemma_id -> {judge_model -> [ranked_model_list]}
    """
    all_rankings = {}  # did -> {judge -> [models]}

    for f in COMPARISONS_DIR.glob("compare_*.json"):
        try:
            data = load_results_file(f)
            judge = data.get("judge_model", "unknown")

            for comp in data.get("comparisons", []):
                did = comp.get("dilemma_id")
                if did:
                    rankings = comp.get("rankings", [])
                    sorted_models = [
                        r["model"] for r in sorted(rankings, key=lambda x: x["rank"])
                    ]
                    if did not in all_rankings:
                        all_rankings[did] = {}
                    all_rankings[did][judge] = sorted_models
        except (json.JSONDecodeError, KeyError, TypeError):
            # Skip malformed comparison files
            continue

    return all_rankings


def load_ai_rankings_for_dilemma(dilemma_id: str) -> dict[str, list]:
    """Load AI rankings for a specific dilemma from all comparison files.

    Note: For batch operations, use load_all_ai_rankings() instead to avoid
    repeated disk I/O.
    """
    all_rankings = load_all_ai_rankings()
    return all_rankings.get(dilemma_id, {})


def load_and_prepare_responses(
    dilemma_id: str,
    model_keys: list[str],
    max_response_chars: int = 3000,
) -> Optional[dict]:
    """Load and blind responses for a dilemma."""
    model_results = {}

    for model_key in model_keys:
        filepath = find_best_results_for_model(model_key)
        if filepath:
            model_results[model_key] = load_results_file(filepath)

    if len(model_results) < 2:
        return None

    responses = {}
    first_response = None

    for model_key, results in model_results.items():
        resp = get_response_for_dilemma(results, dilemma_id)
        if resp:
            responses[model_key] = resp
            if first_response is None:
                first_response = resp

    if len(responses) < 2 or first_response is None:
        return None

    # Randomize order for blinding
    model_names = list(responses.keys())
    random.shuffle(model_names)

    blind_labels = [chr(65 + i) for i in range(len(model_names))]  # A, B, C...
    blind_to_real = {}
    real_to_blind = {}

    for i, model_name in enumerate(model_names):
        blind_label = blind_labels[i]
        blind_to_real[blind_label] = model_name
        real_to_blind[model_name] = blind_label

    # Prepare blinded responses
    blinded_responses = {}
    for label in blind_labels:
        model = blind_to_real[label]
        text = responses[model].get("response", "(No response)")
        if len(text) > max_response_chars:
            text = text[:max_response_chars] + "\n\n[Response truncated for evaluation]"
        blinded_responses[label] = text

    return {
        "dilemma_id": dilemma_id,
        "title": first_response.get("dilemma_title", dilemma_id),
        "category": first_response.get("category", "?"),
        "scenario": first_response.get("scenario", "(Scenario not available)"),
        "question": first_response.get("question", "(Question not available)"),
        "responses": blinded_responses,
        "blind_mapping": blind_to_real,
        "real_to_blind": real_to_blind,
    }


def display_dilemma(data: dict, response_labels: list[str]):
    """Display the dilemma and responses for evaluation."""
    print("\n" + "=" * 70)
    print(f"DILEMMA: {data['title']}")
    print(f"Category: {data['category']} | ID: {data['dilemma_id']}")
    print("=" * 70)

    print("\nSCENARIO:")
    print("-" * 70)
    print(data["scenario"])

    print("\nQUESTION:")
    print("-" * 70)
    print(data["question"])

    print("\n" + "=" * 70)
    print("MODEL RESPONSES (blinded)")
    print("=" * 70)

    for label in response_labels:
        print(f"\n{'='*35} Response {label} {'='*35}")
        print(data["responses"][label])
        print()


def get_human_ranking(labels: list[str]) -> tuple[Optional[list[str]], bool]:
    """Get ranking input from human evaluator.

    Returns:
        Tuple of (ranking, should_quit) where ranking is None if skipped
    """
    print("-" * 70)
    print("RANK THE RESPONSES")
    print("-" * 70)
    print(f"Enter your ranking from best to worst (e.g., '{' '.join(labels)}')")
    print("Or enter 'skip' to skip this dilemma, 'quit' to exit")
    print()

    while True:
        try:
            user_input = input("Your ranking: ").strip().upper()

            if user_input.lower() == "skip":
                return None, False

            if user_input.lower() == "quit":
                return None, True

            # Parse ranking
            ranking = user_input.split()

            # Validate
            if set(ranking) != set(labels):
                print(
                    f"Please rank all responses exactly once. Use: {' '.join(labels)}"
                )
                continue

            if len(ranking) != len(labels):
                print(f"Please rank all {len(labels)} responses")
                continue

            return ranking, False

        except (KeyboardInterrupt, EOFError):
            print("\n")
            return None, True


def calculate_correlation(
    human_ranking: list[str], ai_rankings: dict[str, list], mapping: dict
) -> dict:
    """Calculate correlation between human and AI rankings."""
    results = {}

    # Convert human ranking to real model names
    human_models = [mapping[label] for label in human_ranking]

    for judge, ai_models in ai_rankings.items():
        if not ai_models:
            continue

        # Spearman rank correlation
        n = len(human_models)
        if set(human_models) != set(ai_models):
            print(
                f"    Warning: Skipping correlation for '{judge}' (mismatched model sets)"
            )
            continue

        # Get ranks for each model
        human_ranks = {model: i for i, model in enumerate(human_models)}
        ai_ranks = {model: i for i, model in enumerate(ai_models)}

        # Calculate d^2 sum
        d_squared_sum = 0
        for model in human_models:
            if model in ai_ranks:
                d = human_ranks[model] - ai_ranks[model]
                d_squared_sum += d**2

        # Spearman correlation: 1 - (6 * sum(d^2)) / (n * (n^2 - 1))
        if n > 1:
            correlation = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
        else:
            correlation = 1.0

        # Exact match
        exact_match = human_models == ai_models
        winner_match = (
            human_models[0] == ai_models[0] if human_models and ai_models else False
        )

        results[judge] = {
            "correlation": round(correlation, 3),
            "exact_match": exact_match,
            "winner_match": winner_match,
            "human_ranking": human_models,
            "ai_ranking": ai_models,
        }

    return results


def run_human_eval(
    model_keys: list[str],
    num_samples: int = 10,
    specific_dilemmas: Optional[list[str]] = None,
    max_response_chars: int = 3000,
    resume_file: Optional[str] = None,
):
    """Run human evaluation session.

    Args:
        model_keys: Models to evaluate
        num_samples: Number of random dilemmas to evaluate
        specific_dilemmas: Specific dilemma IDs to evaluate
        max_response_chars: Maximum characters to display per response
        resume_file: Path to existing human eval file to resume from
    """
    print("\n" + "=" * 70)
    print("AI JUDGMENT BATTERY - HUMAN VALIDATION")
    print("=" * 70)
    print(f"Models: {', '.join(model_keys)}")
    print(f"Samples: {num_samples}")

    # Handle resume from existing file
    completed_dilemma_ids: set[str] = set()
    evaluations: list[dict] = []
    output_file = None

    if resume_file:
        resume_path = HUMAN_EVAL_DIR / resume_file
        if not resume_path.exists():
            print(f"ERROR: Resume file not found: {resume_path}")
            sys.exit(1)

        with open(resume_path) as f:
            resume_data = json.load(f)

        evaluations = resume_data.get("evaluations", [])
        completed_dilemma_ids = {e["dilemma_id"] for e in evaluations}
        output_file = resume_path
        print(f"Resuming from: {resume_file}")
        print(f"Already completed: {len(completed_dilemma_ids)} dilemmas")

    print()

    # Find all common dilemmas
    model_results = {}
    for model_key in model_keys:
        filepath = find_best_results_for_model(model_key)
        if filepath:
            model_results[model_key] = load_results_file(filepath)

    if len(model_results) < 2:
        print("ERROR: Need at least 2 models with results")
        sys.exit(1)

    # Find common dilemmas
    all_dilemma_ids: Optional[set] = None
    for results in model_results.values():
        ids = {
            r["dilemma_id"]
            for r in results.get("responses", [])
            if r.get("response") and not r.get("error")
        }
        if all_dilemma_ids is None:
            all_dilemma_ids = ids
        else:
            all_dilemma_ids &= ids

    dilemma_ids = sorted(all_dilemma_ids) if all_dilemma_ids else []

    # Remove already completed dilemmas when resuming
    if completed_dilemma_ids:
        dilemma_ids = [d for d in dilemma_ids if d not in completed_dilemma_ids]

    if specific_dilemmas:
        dilemma_ids = [d for d in dilemma_ids if d in specific_dilemmas]
    else:
        # Random sample
        if len(dilemma_ids) > num_samples:
            dilemma_ids = random.sample(dilemma_ids, num_samples)

    print(f"Dilemmas to evaluate: {len(dilemma_ids)}")
    print("-" * 70)
    print()

    # Set up output file
    HUMAN_EVAL_DIR.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc)
    if output_file is None:
        output_file = (
            HUMAN_EVAL_DIR / f"human_eval_{now.strftime('%Y%m%d_%H%M%S')}.json"
        )

    # Pre-load all AI rankings to avoid repeated disk I/O
    print("Loading AI rankings...")
    all_ai_rankings = load_all_ai_rankings()
    print(f"Loaded rankings for {len(all_ai_rankings)} dilemmas")
    print()

    skipped = 0

    for i, did in enumerate(dilemma_ids):
        print(f"\n[{i+1}/{len(dilemma_ids)}] Loading {did}...")

        data = load_and_prepare_responses(did, model_keys, max_response_chars)
        if data is None:
            print(f"  Skipping {did}: insufficient responses")
            skipped += 1
            continue

        labels = sorted(data["responses"].keys())
        display_dilemma(data, labels)

        ranking, should_quit = get_human_ranking(labels)

        if should_quit:
            print("\nExiting evaluation...")
            break

        if ranking is None:
            print("  Skipped")
            skipped += 1
            continue

        # At this point ranking is guaranteed to be list[str]
        human_ranking_blinded: list[str] = ranking

        # Get AI rankings from pre-loaded data
        ai_rankings = all_ai_rankings.get(did, {})
        correlations = calculate_correlation(
            human_ranking_blinded, ai_rankings, data["blind_mapping"]
        )

        # Convert ranking to real model names
        human_models = [data["blind_mapping"][label] for label in human_ranking_blinded]

        evaluation = {
            "dilemma_id": did,
            "title": data["title"],
            "category": data["category"],
            "human_ranking": human_models,
            "human_ranking_blinded": human_ranking_blinded,
            "blind_mapping": data["blind_mapping"],
            "ai_correlations": correlations,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        evaluations.append(evaluation)

        # Show immediate feedback
        print("\n" + "-" * 70)
        print("RESULTS:")
        print(f"  Your ranking: {' > '.join(human_models)}")
        if correlations:
            for judge, corr in correlations.items():
                judge_short = judge.split("-")[0] if "-" in judge else judge[:15]
                match = (
                    "EXACT"
                    if corr["exact_match"]
                    else ("Winner" if corr["winner_match"] else "Diff")
                )
                print(f"  vs {judge_short}: r={corr['correlation']:.2f} ({match})")

        # Save incrementally
        output_data = {
            "run_id": now.strftime("%Y%m%d_%H%M%S"),
            "models_evaluated": model_keys,
            "timestamp": now.isoformat(),
            "total_planned": len(dilemma_ids),
            "completed": len(evaluations),
            "skipped": skipped,
            "evaluations": evaluations,
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

    # Final summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Completed: {len(evaluations)}/{len(dilemma_ids)}")
    print(f"Skipped: {skipped}")
    print(f"Results: {output_file}")

    if evaluations:
        # Aggregate statistics
        print("\n" + "-" * 70)
        print("AGGREGATE STATISTICS")
        print("-" * 70)

        # Human winner counts
        human_wins = {}
        for eval_data in evaluations:
            winner = (
                eval_data["human_ranking"][0] if eval_data["human_ranking"] else None
            )
            if winner:
                human_wins[winner] = human_wins.get(winner, 0) + 1

        print("\nHuman rankings (1st place counts):")
        for model, count in sorted(human_wins.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(evaluations)
            print(f"  {model:<20} {count:>4} ({pct:5.1f}%)")

        # Correlation with each AI judge
        judge_correlations = {}
        judge_winner_matches = {}
        judge_exact_matches = {}

        for eval_data in evaluations:
            for judge, corr in eval_data.get("ai_correlations", {}).items():
                if judge not in judge_correlations:
                    judge_correlations[judge] = []
                    judge_winner_matches[judge] = 0
                    judge_exact_matches[judge] = 0

                judge_correlations[judge].append(corr["correlation"])
                if corr["winner_match"]:
                    judge_winner_matches[judge] += 1
                if corr["exact_match"]:
                    judge_exact_matches[judge] += 1

        print("\nCorrelation with AI judges:")
        for judge in sorted(judge_correlations.keys()):
            corrs = judge_correlations[judge]
            avg_corr = sum(corrs) / len(corrs) if corrs else 0
            winner_rate = judge_winner_matches[judge] / len(corrs) if corrs else 0
            exact_rate = judge_exact_matches[judge] / len(corrs) if corrs else 0
            judge_short = (
                judge.split("-preview")[0] if "-preview" in judge else judge[:25]
            )
            print(
                f"  {judge_short:<25} r={avg_corr:.2f}  Winner={winner_rate*100:.0f}%  Exact={exact_rate*100:.0f}%"
            )

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Human validation of AI judgment comparisons"
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        default=["claude-opus", "gpt-5.1", "gemini-3-pro"],
        help="Models to evaluate (default: claude-opus gpt-5.1 gemini-3-pro)",
    )
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=10,
        help="Number of dilemmas to evaluate (default: 10)",
    )
    parser.add_argument(
        "--dilemmas",
        "-d",
        nargs="+",
        help="Specific dilemma IDs to evaluate",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=3000,
        help="Max response characters to display (default: 3000)",
    )
    parser.add_argument(
        "--resume",
        "-r",
        help="Resume from existing human eval file",
    )

    args = parser.parse_args()

    run_human_eval(
        model_keys=args.models,
        num_samples=args.samples,
        specific_dilemmas=args.dilemmas,
        max_response_chars=args.max_chars,
        resume_file=args.resume,
    )


if __name__ == "__main__":
    main()
