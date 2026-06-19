#!/usr/bin/env python3
"""
Evaluation Harness (Step 4 — §7 of SOLUTION_SPEC).

Runs the full pipeline against all 20 rows of dataset/sample_claims.csv
using two strategies:
  - Strategy A: Single-pass Zero-Shot
  - Strategy B: Chain-of-Thought (step-by-step reasoning)

Compares results against labeled data and reports accuracy metrics.

Usage:
    cd code/
    python evaluation/main.py
"""

import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv

# Add code/ to path
_script_dir = Path(__file__).resolve().parent
_code_dir = _script_dir.parent
sys.path.insert(0, str(_code_dir))

# Load env
load_dotenv(_code_dir / ".env", override=True)
load_dotenv(_code_dir.parent / ".env", override=True)

from pipeline.loader import (
    load_csv,
    load_user_history,
    load_evidence_requirements,
    validate_and_prepare_images,
)
from pipeline.model_call import call_vlm
from pipeline.post_processor import post_process

# Paths
REPO_ROOT = _code_dir.parent
DATASET_DIR = REPO_ROOT / "dataset"
CACHE_DIR = _code_dir / ".cache"

# Fields to evaluate accuracy on
ACCURACY_FIELDS = ["claim_status", "issue_type", "object_part"]

# All fields to compare (for detailed report)
ALL_FIELDS = [
    "valid_image", "evidence_standard_met", "risk_flags",
    "issue_type", "object_part", "claim_status", "severity",
    "supporting_image_ids",
]


def format_output_value(val):
    """Convert a processed output value to string for comparison."""
    if isinstance(val, list):
        return ";".join(str(v) for v in val)
    if isinstance(val, bool):
        return str(val).lower()
    return str(val)


def run_strategy(
    strategy_name: str,
    strategy_key: str,
    claims: list[dict],
    user_history: dict,
    evidence_reqs: list[dict],
) -> list[dict]:
    """
    Run a strategy against all claims and return post-processed results.
    """
    print(f"\n{'='*80}")
    print(f"  STRATEGY: {strategy_name} ({strategy_key})")
    print(f"{'='*80}")

    results = []
    total_tokens = 0
    api_calls = 0
    cache_hits = 0
    errors = 0

    for i, row in enumerate(claims):
        user_id = row["user_id"]
        claim_object = row["claim_object"]
        image_paths = row["image_paths"]

        print(f"\n  [{i+1:2d}/{len(claims)}] {user_id} | {claim_object}", end="")

        # Validate images
        image_data = validate_and_prepare_images(
            image_paths, str(DATASET_DIR)
        )
        valid_image_ids = [img["image_id"] for img in image_data if img["is_valid"]]

        # Get user history
        history = user_history.get(user_id)

        try:
            # Call VLM with strategy
            raw_result = call_vlm(
                claim_row=row,
                image_data_list=image_data,
                user_history=history,
                evidence_requirements=evidence_reqs,
                cache_dir=str(CACHE_DIR),
                model="gpt-4o",
                strategy=strategy_key,
            )

            # Track usage
            metadata = raw_result.get("_metadata", {})
            usage = metadata.get("usage", {})
            tokens = usage.get("total_tokens", 0)
            total_tokens += tokens
            if "cache_key" in metadata:
                # Check if this was an API call or cache hit by checking
                # if usage tokens are present
                if tokens > 0:
                    api_calls += 1
                else:
                    cache_hits += 1

            # Post-process
            processed = post_process(
                raw_output=dict(raw_result),
                claim_object=claim_object,
                user_history=history,
                valid_image_ids=valid_image_ids,
            )

            results.append(processed)

        except Exception as e:
            print(f" ❌ ERROR: {e}")
            errors += 1
            # Add a placeholder result for failed rows
            results.append({
                "valid_image": "false",
                "evidence_standard_met": "false",
                "evidence_standard_met_reason": f"Processing error: {e}",
                "risk_flags": "none",
                "issue_type": "unknown",
                "object_part": "unknown",
                "claim_status": "not_enough_information",
                "claim_status_justification": f"Error: {e}",
                "supporting_image_ids": "none",
                "severity": "unknown",
            })

    print(f"\n\n  📊 {strategy_name} Summary:")
    print(f"     API calls: {api_calls} | Cache hits: {cache_hits} | "
          f"Errors: {errors}")
    print(f"     Total tokens: {total_tokens:,}")

    return results


def compute_accuracy(
    results: list[dict],
    expected: list[dict],
    fields: list[str],
) -> dict:
    """
    Compute per-field accuracy between results and expected values.
    Returns {field: {correct, total, accuracy, mismatches}}.
    """
    metrics = {}
    for field in fields:
        correct = 0
        total = len(results)
        mismatches = []

        for i, (res, exp) in enumerate(zip(results, expected)):
            res_val = str(res.get(field, "")).strip()
            exp_val = str(exp.get(field, "")).strip()

            if res_val == exp_val:
                correct += 1
            else:
                mismatches.append({
                    "row": i + 1,
                    "user_id": exp.get("user_id", "?"),
                    "expected": exp_val,
                    "got": res_val,
                })

        metrics[field] = {
            "correct": correct,
            "total": total,
            "accuracy": (correct / total * 100) if total > 0 else 0,
            "mismatches": mismatches,
        }

    return metrics


def print_metrics(
    strategy_name: str,
    metrics: dict,
    fields: list[str],
    show_mismatches: bool = True,
):
    """Print accuracy metrics in a formatted table."""
    print(f"\n  📈 {strategy_name} — Accuracy:")
    print(f"  {'Field':<25} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print(f"  {'-'*52}")

    for field in fields:
        m = metrics[field]
        bar = "█" * int(m["accuracy"] / 5) + "░" * (20 - int(m["accuracy"] / 5))
        print(f"  {field:<25} {m['correct']:>8} {m['total']:>6} "
              f"{m['accuracy']:>9.1f}%  {bar}")

    # Overall (average of the 3 key fields)
    key_fields = ["claim_status", "issue_type", "object_part"]
    avg = sum(metrics[f]["accuracy"] for f in key_fields if f in metrics) / len(key_fields)
    print(f"  {'-'*52}")
    print(f"  {'AVERAGE (key 3)':<25} {'':>8} {'':>6} {avg:>9.1f}%")

    if show_mismatches:
        for field in fields:
            m = metrics[field]
            if m["mismatches"]:
                print(f"\n  ⚠️  {field} mismatches ({len(m['mismatches'])}):")
                for mm in m["mismatches"][:10]:  # Show first 10
                    print(f"     Row {mm['row']:2d} ({mm['user_id']}): "
                          f"expected='{mm['expected']}' got='{mm['got']}'")
                if len(m["mismatches"]) > 10:
                    print(f"     ... and {len(m['mismatches']) - 10} more")


def main():
    start_time = time.time()

    print("=" * 80)
    print("STEP 4: Full Evaluation Harness — Two-Strategy Comparison")
    print("=" * 80)

    # Verify API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n❌ ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    # Load data
    print("\n📂 Loading data...")
    claims = load_csv(str(DATASET_DIR / "sample_claims.csv"))
    user_history = load_user_history(str(DATASET_DIR / "user_history.csv"))
    evidence_reqs = load_evidence_requirements(
        str(DATASET_DIR / "evidence_requirements.csv")
    )
    print(f"  Claims: {len(claims)} | Users: {len(user_history)} | "
          f"Requirements: {len(evidence_reqs)}")

    # ---- Strategy A: Zero-Shot ----
    results_zs = run_strategy(
        "Strategy A: Zero-Shot",
        "zero_shot",
        claims, user_history, evidence_reqs,
    )

    # ---- Strategy B: Chain-of-Thought ----
    results_cot = run_strategy(
        "Strategy B: Chain-of-Thought",
        "cot",
        claims, user_history, evidence_reqs,
    )

    # ---- Compute Metrics ----
    print("\n" + "=" * 80)
    print("  EVALUATION RESULTS")
    print("=" * 80)

    metrics_zs = compute_accuracy(results_zs, claims, ALL_FIELDS)
    metrics_cot = compute_accuracy(results_cot, claims, ALL_FIELDS)

    print_metrics("Strategy A: Zero-Shot", metrics_zs, ALL_FIELDS)
    print_metrics("Strategy B: Chain-of-Thought", metrics_cot, ALL_FIELDS)

    # ---- Head-to-Head Comparison ----
    print(f"\n{'='*80}")
    print("  HEAD-TO-HEAD COMPARISON (Key 3 Fields)")
    print(f"{'='*80}")
    print(f"\n  {'Field':<25} {'Zero-Shot':>12} {'CoT':>12} {'Winner':>12}")
    print(f"  {'-'*64}")

    for field in ACCURACY_FIELDS:
        zs_acc = metrics_zs[field]["accuracy"]
        cot_acc = metrics_cot[field]["accuracy"]
        if zs_acc > cot_acc:
            winner = "Zero-Shot"
        elif cot_acc > zs_acc:
            winner = "CoT"
        else:
            winner = "Tie"
        print(f"  {field:<25} {zs_acc:>11.1f}% {cot_acc:>11.1f}% {winner:>12}")

    zs_avg = sum(metrics_zs[f]["accuracy"] for f in ACCURACY_FIELDS) / len(ACCURACY_FIELDS)
    cot_avg = sum(metrics_cot[f]["accuracy"] for f in ACCURACY_FIELDS) / len(ACCURACY_FIELDS)
    overall_winner = "Zero-Shot" if zs_avg > cot_avg else ("CoT" if cot_avg > zs_avg else "Tie")
    print(f"  {'-'*64}")
    print(f"  {'AVERAGE':<25} {zs_avg:>11.1f}% {cot_avg:>11.1f}% {overall_winner:>12}")

    # ---- Timing ----
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total evaluation time: {elapsed:.1f}s")
    print(f"   Recommended strategy: {overall_winner}")

    print(f"\n{'='*80}")
    print("Step 4 complete. Review the metrics above before proceeding to Step 5.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
