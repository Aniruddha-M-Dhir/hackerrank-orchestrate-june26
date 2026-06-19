#!/usr/bin/env python3
"""
Step 2 Test Script: Run the model call against the first 3 rows of
dataset/sample_claims.csv, print raw JSON outputs, and cache responses.

Usage:
    cd code/
    python test_step2.py

Requires OPENAI_API_KEY in environment or .env file.
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
    load_evidence_requirements,
    validate_and_prepare_images,
)
from pipeline.model_call import call_vlm

# Load env vars — override=True ensures .env values take precedence
_script_dir = Path(__file__).resolve().parent
load_dotenv(_script_dir / ".env", override=True)
load_dotenv(_script_dir.parent / ".env", override=True)

# Paths
REPO_ROOT = Path(__file__).parent.parent
DATASET_DIR = REPO_ROOT / "dataset"
CACHE_DIR = Path(__file__).parent / ".cache"


def main():
    print("=" * 70)
    print("STEP 2: Model Call Test — First 3 rows of sample_claims.csv")
    print("=" * 70)

    # Verify API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n❌ ERROR: OPENAI_API_KEY not set in environment or .env file")
        print("  Set it with: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    # Load data
    print("\n📂 Loading data...")
    claims = load_csv(str(DATASET_DIR / "sample_claims.csv"))
    user_history = load_user_history(str(DATASET_DIR / "user_history.csv"))
    evidence_reqs = load_evidence_requirements(
        str(DATASET_DIR / "evidence_requirements.csv")
    )

    print(f"  Total sample claims: {len(claims)}")
    print(f"  User history entries: {len(user_history)}")
    print(f"  Evidence requirements: {len(evidence_reqs)}")

    # Process first 3 rows only
    test_rows = claims[:3]

    print(f"\n🔬 Processing {len(test_rows)} rows...")
    print("-" * 70)

    for i, row in enumerate(test_rows):
        user_id = row["user_id"]
        claim_object = row["claim_object"]
        image_paths = row["image_paths"]

        print(f"\n{'='*70}")
        print(f"ROW {i+1}: {user_id} | {claim_object}")
        print(f"Images: {image_paths}")
        print(f"Claim: {row['user_claim'][:120]}...")
        print(f"{'='*70}")

        # Step 1: Validate images
        image_data = validate_and_prepare_images(
            image_paths, str(DATASET_DIR)
        )
        valid_count = sum(1 for img in image_data if img["is_valid"])
        print(f"\n  📷 Images: {len(image_data)} total, {valid_count} valid")
        for img in image_data:
            status = "✅" if img["is_valid"] else "❌"
            reason = f" ({img['rejection_reason']})" if img.get("rejection_reason") else ""
            print(f"    {status} {img['image_id']}: {img.get('mime_type', 'N/A')}{reason}")

        # Step 2: Call VLM
        print(f"\n  🤖 Calling model...")
        history = user_history.get(user_id)

        try:
            raw_result = call_vlm(
                claim_row=row,
                image_data_list=image_data,
                user_history=history,
                evidence_requirements=evidence_reqs,
                cache_dir=str(CACHE_DIR),
                model="gpt-4o",
            )

            # Print raw JSON (excluding base64 data in metadata)
            print_result = {k: v for k, v in raw_result.items()}
            print(f"\n  📋 RAW MODEL OUTPUT:")
            print(json.dumps(print_result, indent=2))

            # Show expected vs actual for key fields (from sample_claims.csv)
            print(f"\n  📊 COMPARISON WITH EXPECTED:")
            expected_fields = [
                "evidence_standard_met", "risk_flags", "issue_type",
                "object_part", "claim_status", "valid_image", "severity"
            ]
            for field in expected_fields:
                expected = row.get(field, "N/A")
                actual = raw_result.get(field)
                if isinstance(actual, list):
                    actual = ";".join(str(a) for a in actual)
                elif isinstance(actual, bool):
                    actual = str(actual).lower()
                match = "✅" if str(expected).strip() == str(actual).strip() else "⚠️"
                print(f"    {match} {field}: expected='{expected}' got='{actual}'")

        except Exception as e:
            print(f"\n  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*70}")
    print("Step 2 complete. Cache saved to: code/.cache/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
