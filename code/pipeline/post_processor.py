"""
Deterministic post-processor (§5 of SOLUTION_SPEC).

Takes the raw LLM output and enforces logical consistency rules
before writing to output.csv. The LLM output is treated as a suggestion.

Rules:
1. Invalid Image Override: valid_image=false → evidence_standard_met=false, claim_status=not_enough_information
2. Contradiction Override: issue_type=none + part visible + user claimed damage → claim_status=contradicted, severity=none
3. Null State Fallbacks: no supporting images → supporting_image_ids=none
4. Risk flags history-merge (§2.2): union of LLM visual flags + user_history flags + escalation trigger
5. Enum validation: all values must be from the allowed lists
"""

from typing import Optional


# Allowed enum values from problem_statement.md
VALID_CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}
VALID_ISSUE_TYPES = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown"
}
VALID_SEVERITY = {"none", "low", "medium", "high", "unknown"}
VALID_RISK_FLAGS = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required"
}
VALID_OBJECT_PARTS = {
    "car": {
        "front_bumper", "rear_bumper", "door", "hood", "windshield",
        "side_mirror", "headlight", "taillight", "fender", "quarter_panel",
        "body", "unknown"
    },
    "laptop": {
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner",
        "port", "base", "body", "unknown"
    },
    "package": {
        "box", "package_corner", "package_side", "seal", "label",
        "contents", "item", "unknown"
    },
}


def merge_risk_flags(
    llm_flags: list[str],
    user_history: Optional[dict],
) -> list[str]:
    """
    §2.2 Risk Flags History-Merge Rule.

    1. Start with LLM-detected visual flags (excluding user_history_risk and manual_review_required)
    2. Inject history_flags from user_history.csv
    3. Escalation trigger: if user has user_history_risk AND LLM detected ANY visual flag → add manual_review_required
    4. Deduplicate
    """
    # Clean LLM flags — remove "none" and reserved flags the LLM shouldn't set
    visual_flags = set()
    for f in llm_flags:
        f = f.strip().lower()
        if f and f != "none" and f in VALID_RISK_FLAGS:
            # LLM should not set these reserved flags
            if f not in ("user_history_risk", "manual_review_required"):
                visual_flags.add(f)

    # Inject history flags
    history_flags = set()
    if user_history:
        raw_history = user_history.get("history_flags", "none")
        for hf in raw_history.split(";"):
            hf = hf.strip().lower()
            if hf and hf != "none" and hf in VALID_RISK_FLAGS:
                history_flags.add(hf)

    # Merge
    all_flags = visual_flags | history_flags

    # §2.2 Escalation trigger
    has_history_risk = "user_history_risk" in history_flags
    has_visual_risk = len(visual_flags) > 0
    if has_history_risk and has_visual_risk:
        all_flags.add("manual_review_required")

    # If no flags at all, return "none"
    if not all_flags:
        return ["none"]

    # Sort for determinism
    return sorted(all_flags)


def post_process(
    raw_output: dict,
    claim_object: str,
    user_history: Optional[dict],
    valid_image_ids: list[str],
) -> dict:
    """
    Apply all deterministic post-processing rules (§5) to the raw LLM output.

    Args:
        raw_output: Raw JSON from the LLM
        claim_object: "car", "laptop", or "package"
        user_history: User's history dict or None
        valid_image_ids: List of valid image IDs for this claim

    Returns:
        Post-processed output dict ready for CSV
    """
    result = dict(raw_output)

    # Remove metadata before processing
    result.pop("_metadata", None)

    # --- Enum validation ---
    if result.get("claim_status") not in VALID_CLAIM_STATUS:
        result["claim_status"] = "not_enough_information"

    if result.get("issue_type") not in VALID_ISSUE_TYPES:
        result["issue_type"] = "unknown"

    if result.get("severity") not in VALID_SEVERITY:
        result["severity"] = "unknown"

    allowed_parts = VALID_OBJECT_PARTS.get(claim_object, set())
    if result.get("object_part") not in allowed_parts:
        result["object_part"] = "unknown"

    # --- §5.1 Invalid Image Override ---
    if result.get("valid_image") is False:
        result["evidence_standard_met"] = False
        result["claim_status"] = "not_enough_information"

    # --- §5.2 Contradiction Override ---
    if (result.get("issue_type") == "none"
            and result.get("claim_status") != "not_enough_information"):
        result["claim_status"] = "contradicted"
        result["severity"] = "none"

    # --- §5.3 Null State Fallbacks ---
    supporting = result.get("supporting_image_ids", [])
    if isinstance(supporting, list):
        # Filter to only valid image IDs
        filtered = [s for s in supporting if s and s != "none" and s in valid_image_ids]
        if not filtered:
            result["supporting_image_ids"] = ["none"]
        else:
            result["supporting_image_ids"] = filtered
    else:
        result["supporting_image_ids"] = ["none"]

    # --- §2.2 Risk flags history merge ---
    llm_flags = result.get("risk_flags", ["none"])
    if isinstance(llm_flags, str):
        llm_flags = [f.strip() for f in llm_flags.split(";")]
    result["risk_flags"] = merge_risk_flags(llm_flags, user_history)

    # --- Validate risk flags ---
    validated_flags = [f for f in result["risk_flags"] if f in VALID_RISK_FLAGS]
    result["risk_flags"] = validated_flags if validated_flags else ["none"]

    # --- Convert booleans to strings for CSV ---
    result["valid_image"] = str(result.get("valid_image", True)).lower()
    result["evidence_standard_met"] = str(result.get("evidence_standard_met", True)).lower()

    # --- Convert lists to semicolon-separated strings ---
    result["risk_flags"] = ";".join(result["risk_flags"])
    result["supporting_image_ids"] = ";".join(result["supporting_image_ids"])

    return result
