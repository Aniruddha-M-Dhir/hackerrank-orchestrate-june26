#!/usr/bin/env python3
"""
Multi-Modal Evidence Review Pipeline — Main Entry Point

Reads dataset/claims.csv and produces output.csv with structured predictions
for each damage claim using GPT-4o with Chain-of-Thought prompting.

Usage:
    cd code/
    python main.py

Requires:
    OPENAI_API_KEY set in environment or code/.env
"""

import csv
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATASET_DIR = REPO_ROOT / "dataset"
CACHE_DIR = SCRIPT_DIR / ".cache"
OUTPUT_FILE = REPO_ROOT / "output.csv"

# Load environment variables
load_dotenv(SCRIPT_DIR / ".env", override=True)
load_dotenv(REPO_ROOT / ".env", override=True)

from pipeline.loader import (
    load_csv,
    load_user_history,
    load_evidence_requirements,
    validate_and_prepare_images,
)
from pipeline.model_call import call_vlm
from pipeline.post_processor import post_process

# ---- Configuration ----
MODEL = "gpt-4o"
STRATEGY = "cot"  # Chain-of-Thought — selected based on §7 evaluation results

# Required output columns in exact order (from problem_statement.md)
OUTPUT_COLUMNS = [
    "user_id", "image_paths", "user_claim", "claim_object",
    "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part", "claim_status",
    "claim_status_justification", "supporting_image_ids",
    "valid_image", "severity",
]


def process_claim(
    row: dict,
    user_history_map: dict,
    evidence_reqs: list[dict],
) -> dict:
    """Process a single claim row through the full pipeline."""
    user_id = row["user_id"]
    claim_object = row["claim_object"]
    image_paths = row["image_paths"]

    # Step 1: Validate and prepare images (§2.3 format sniffing)
    image_data = validate_and_prepare_images(image_paths, str(DATASET_DIR))
    valid_image_ids = [img["image_id"] for img in image_data if img["is_valid"]]

    # Check if ANY image failed format sniffing → possible_manipulation
    any_invalid_format = any(
        img.get("rejection_reason") == "invalid_format" for img in image_data
    )

    # Get user history
    history = user_history_map.get(user_id)

    # Step 2: Call VLM with CoT strategy (§4 structured output)
    raw_result = call_vlm(
        claim_row=row,
        image_data_list=image_data,
        user_history=history,
        evidence_requirements=evidence_reqs,
        cache_dir=str(CACHE_DIR),
        model=MODEL,
        strategy=STRATEGY,
    )

    # Step 3: Deterministic post-processing (§5)
    processed = post_process(
        raw_output=dict(raw_result),
        claim_object=claim_object,
        user_history=history,
        valid_image_ids=valid_image_ids,
    )

    # If format sniffing caught a fake file, inject possible_manipulation
    if any_invalid_format:
        flags = processed.get("risk_flags", "none")
        if "possible_manipulation" not in flags:
            if flags == "none":
                processed["risk_flags"] = "possible_manipulation"
            else:
                processed["risk_flags"] = flags + ";possible_manipulation"

    # Assemble the full output row (input columns + generated columns)
    output_row = {
        "user_id": user_id,
        "image_paths": image_paths,
        "user_claim": row["user_claim"],
        "claim_object": claim_object,
    }
    output_row.update(processed)

    return output_row


def write_output_csv(rows: list[dict], output_path: str):
    """Write the output CSV with exact column order."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=OUTPUT_COLUMNS,
            quoting=csv.QUOTE_ALL,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    start_time = time.time()

    print("=" * 70)
    print("Multi-Modal Evidence Review Pipeline")
    print(f"Model: {MODEL} | Strategy: {STRATEGY}")
    print("=" * 70)

    # Verify API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n❌ ERROR: OPENAI_API_KEY not set in environment or .env file")
        sys.exit(1)

    # Load data
    print("\n📂 Loading data...")
    claims = load_csv(str(DATASET_DIR / "claims.csv"))
    user_history = load_user_history(str(DATASET_DIR / "user_history.csv"))
    evidence_reqs = load_evidence_requirements(
        str(DATASET_DIR / "evidence_requirements.csv")
    )
    print(f"  Claims to process: {len(claims)}")
    print(f"  User history entries: {len(user_history)}")
    print(f"  Evidence requirements: {len(evidence_reqs)}")

    # Process all claims
    print(f"\n🔬 Processing {len(claims)} claims...")
    print("-" * 70)

    output_rows = []
    total_tokens = 0
    errors = 0

    for i, row in enumerate(claims):
        user_id = row["user_id"]
        claim_object = row["claim_object"]
        print(f"  [{i+1:2d}/{len(claims)}] {user_id} | {claim_object}", end="")

        try:
            result = process_claim(row, user_history, evidence_reqs)
            output_rows.append(result)
        except Exception as e:
            print(f" ❌ ERROR: {e}")
            errors += 1
            # Write a safe fallback row
            output_rows.append({
                "user_id": user_id,
                "image_paths": row["image_paths"],
                "user_claim": row["user_claim"],
                "claim_object": claim_object,
                "evidence_standard_met": "false",
                "evidence_standard_met_reason": f"Processing error: {e}",
                "risk_flags": "manual_review_required",
                "issue_type": "unknown",
                "object_part": "unknown",
                "claim_status": "not_enough_information",
                "claim_status_justification": f"Automated processing failed: {e}",
                "supporting_image_ids": "none",
                "valid_image": "false",
                "severity": "unknown",
            })

    # Write output
    print(f"\n\n📝 Writing output to {OUTPUT_FILE}...")
    write_output_csv(output_rows, str(OUTPUT_FILE))

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✅ Complete!")
    print(f"   Rows processed: {len(output_rows)}")
    print(f"   Errors: {errors}")
    print(f"   Output: {OUTPUT_FILE}")
    print(f"   Runtime: {elapsed:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
