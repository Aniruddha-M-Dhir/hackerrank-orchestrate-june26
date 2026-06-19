"""
Data loader and image validation module (Step 1).

Handles:
- CSV ingestion (claims, user_history, evidence_requirements)
- Image format sniffing (§2.3) — validates actual file headers, not just extensions
- Image base64 encoding for VLM calls
"""

import csv
import base64
import hashlib
import os
from pathlib import Path
from typing import Optional

# Mapping of magic bytes -> MIME type for image format sniffing (§2.3)
_MAGIC_SIGNATURES = [
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),  # WebP starts with RIFF....WEBP
    (b"BM", "image/bmp"),
]


def sniff_image_format(file_path: str) -> Optional[str]:
    """
    Read the file header to determine the actual image format.
    Returns the MIME type if it's a valid image, None otherwise.
    This implements §2.3: never trust file extensions.
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(32)
    except (OSError, IOError):
        return None

    if len(header) < 4:
        return None

    for magic, mime_type in _MAGIC_SIGNATURES:
        if header.startswith(magic):
            # Extra check for WebP: must have WEBP after RIFF header
            if mime_type == "image/webp":
                if len(header) >= 12 and header[8:12] == b"WEBP":
                    return mime_type
                continue
            return mime_type

    return None


def encode_image_base64(file_path: str) -> Optional[str]:
    """Read an image file and return its base64 encoding."""
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except (OSError, IOError):
        return None


def hash_image_file(file_path: str) -> str:
    """Create a SHA-256 hash of an image file for caching purposes."""
    sha = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()
    except (OSError, IOError):
        return ""


def extract_image_id(image_path: str) -> str:
    """Extract image ID from path: filename without extension."""
    return Path(image_path).stem


def load_csv(file_path: str) -> list[dict]:
    """Load a CSV file into a list of dicts."""
    rows = []
    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from keys and values
            cleaned = {k.strip(): v.strip() if v else v for k, v in row.items()}
            rows.append(cleaned)
    return rows


def load_user_history(file_path: str) -> dict:
    """Load user_history.csv into a dict keyed by user_id."""
    rows = load_csv(file_path)
    return {row["user_id"]: row for row in rows}


def load_evidence_requirements(file_path: str) -> list[dict]:
    """Load evidence_requirements.csv."""
    return load_csv(file_path)


def validate_and_prepare_images(
    image_paths_str: str, dataset_root: str
) -> list[dict]:
    """
    For a semicolon-separated image_paths string, validate each image
    and prepare it for the VLM call.

    Returns a list of dicts with:
    - path: original path string
    - image_id: extracted ID
    - mime_type: sniffed MIME type or None
    - is_valid: whether the image passed format sniffing
    - base64: base64-encoded image data or None
    - file_hash: SHA-256 hash for caching
    - rejection_reason: why the image was rejected, if any
    """
    results = []
    paths = [p.strip() for p in image_paths_str.split(";") if p.strip()]

    for rel_path in paths:
        abs_path = os.path.join(dataset_root, rel_path)
        image_id = extract_image_id(rel_path)

        entry = {
            "path": rel_path,
            "abs_path": abs_path,
            "image_id": image_id,
            "mime_type": None,
            "is_valid": False,
            "base64": None,
            "file_hash": "",
            "rejection_reason": None,
        }

        # Check file exists
        if not os.path.isfile(abs_path):
            entry["rejection_reason"] = "file_not_found"
            results.append(entry)
            continue

        # Sniff actual format (§2.3)
        mime_type = sniff_image_format(abs_path)
        if mime_type is None:
            entry["rejection_reason"] = "invalid_format"
            results.append(entry)
            continue

        entry["mime_type"] = mime_type
        entry["is_valid"] = True
        entry["file_hash"] = hash_image_file(abs_path)
        entry["base64"] = encode_image_base64(abs_path)

        if entry["base64"] is None:
            entry["is_valid"] = False
            entry["rejection_reason"] = "encoding_failed"

        results.append(entry)

    return results
