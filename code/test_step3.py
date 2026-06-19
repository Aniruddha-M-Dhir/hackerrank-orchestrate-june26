#!/usr/bin/env python3
"""
Step 3 Test Script: Apply the deterministic post-processor to the 3 cached
responses from Step 2, and show Before vs After diffs.

Usage:
    cd code/
    python test_step3.py
"""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add code/ to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.loader import (
    load_csv,
    load_user_history,
    validate_and_prepare_images,
    extract_image_id,
)
from pipeline.post_processor import post_process
from pipeline.model_call import _compute_cache_key, _load_from_cache

# Load env vars
_script_dir = Path(__file__).resolve().parent
load_dotenv(_script_dir / ".env", override=True)

# Paths
REPO_ROOT = Path(__file__).parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
CACHE_DIR = Path(__file__).parent / ".cache"

# Fields to compare
COMPARE_FIELDS = [
    "valid_image", "evidence_standard_met", "evidence_standard_met_reason",
    "risk_flags", "issue_type", "object_part", "claim_status",
    "claim_status_justification", "supporting_image_ids", "severity"
]


def format_value(val):
    """Format a value for display."""
    if isinstance(val, list):
        return ";".join(str(v) for v in val)
    if isinstance(val, bool):
        return str(val).lower()
    return str(val)


def main():
    print("=" * 80)
    print("STEP 3: Post-Processor Verification — Before vs After")
    print("=" * 80)

    # Load data
    claims = load_csv(str(DATASET_DIR / "sample_claims.csv"))
    user_history = load_user_history(str(DATASET_DIR / "user_history.csv"))

    test_rows = claims[:3]

    for i, row in enumerate(test_rows):
        user_id = row["user_id"]
        claim_object = row["claim_object"]
        image_paths = row["image_paths"]

        # Validate images (same as Step 2)
        image_data = validate_and_prepare_images(
            image_paths, str(DATASET_DIR)
        )
        valid_image_ids = [img["image_id"] for img in image_data if img["is_valid"]]

        # Compute cache key to load the cached response
        image_hashes = [img.get("file_hash", "") for img in image_data]
        history = user_history.get(user_id)
        history_str = json.dumps(history, sort_keys=True) if history else ""
        cache_key = _compute_cache_key(
            claim_object, row["user_claim"], image_hashes, history_str,
            strategy="zero_shot"
        )

        # Load cached response
        cached = _load_from_cache(str(CACHE_DIR), cache_key)
        if cached is None:
            print(f"\n❌ ROW {i+1} ({user_id}): No cached response found!")
            print(f"   Cache key: {cache_key}")
            continue

        # Extract raw values (BEFORE post-processing)
        raw_before = {}
        for field in COMPARE_FIELDS:
            raw_before[field] = format_value(cached.get(field))

        # Apply post-processing
        processed = post_process(
            raw_output=dict(cached),
            claim_object=claim_object,
            user_history=history,
            valid_image_ids=valid_image_ids,
        )

        # Extract processed values (AFTER post-processing)
        raw_after = {}
        for field in COMPARE_FIELDS:
            raw_after[field] = str(processed.get(field, ""))

        # Get expected values from sample CSV
        expected = {}
        for field in COMPARE_FIELDS:
            expected[field] = row.get(field, "N/A")

        # Print comparison
        print(f"\n{'='*80}")
        print(f"ROW {i+1}: {user_id} | {claim_object}")
        print(f"{'='*80}")

        has_changes = False
        for field in COMPARE_FIELDS:
            before = raw_before[field]
            after = raw_after[field]
            exp = expected[field]

            changed = before != after
            matches_expected = after == exp

            if changed:
                has_changes = True
                status = "✅ FIXED" if matches_expected else "🔧 CHANGED"
                print(f"\n  {status} {field}:")
                print(f"    BEFORE (raw):       {before}")
                print(f"    AFTER (processed):  {after}")
                print(f"    EXPECTED:           {exp}")
            else:
                match_icon = "✅" if matches_expected else "⚠️"
                print(f"  {match_icon} {field}: {after} (expected: {exp})")

        if not has_changes:
            print(f"\n  ℹ️  No changes from post-processing for this row.")

        # Summary
        match_count = sum(
            1 for f in COMPARE_FIELDS if raw_after[f] == expected[f]
        )
        before_match_count = sum(
            1 for f in COMPARE_FIELDS if raw_before[f] == expected[f]
        )
        print(f"\n  📊 Score: {before_match_count}/{len(COMPARE_FIELDS)} → "
              f"{match_count}/{len(COMPARE_FIELDS)}")

    print(f"\n{'='*80}")
    print("Step 3 complete.")
    print("Post-processing rules applied:")
    print("  1. evidence_standard_met=false → claim_status=not_enough_information")
    print("     + add manual_review_required to risk_flags")
    print("  2. claim_status=not_enough_information → severity=unknown")
    print("  3. §5.1 Invalid Image Override")
    print("  4. §5.2 Contradiction Override")
    print("  5. §5.3 Null State Fallbacks")
    print("  6. §2.2 Risk flags history-merge with escalation trigger")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
