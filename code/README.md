# Multi-Modal Evidence Review Pipeline

This repository contains the solution for the HackerRank Orchestrate challenge: **Multi-Modal Evidence Review**. It is a scalable, deterministic pipeline that leverages Vision-Language Models (VLM) combined with strict rule-based post-processing to evaluate damage claims based on user conversations, historical risk data, and image evidence.

## Architecture

The system operates in three main steps per claim:

1.  **Image Validation & Preparation (`pipeline/loader.py`)**
    *   Reads `claims.csv`, `user_history.csv`, and `evidence_requirements.csv`.
    *   Sniffs file magic bytes to verify true MIME types (preventing extension-spoofing).
    *   Converts valid images to base64 for the VLM.

2.  **VLM Structured Output (`pipeline/model_call.py`)**
    *   Uses OpenAI's GPT-4o with `response_format={"type": "json_schema"}` to guarantee exact adherence to the required JSON schema and Enum constraints.
    *   Utilizes a **Chain-of-Thought (CoT)** prompting strategy (selected via rigorous evaluation) instructing the model to reason step-by-step before emitting the final JSON.
    *   Defends against prompt-injection (text embedded in images or user claims) via strict XML tagging and explicit system instructions.
    *   Implements caching (SHA-256 hashes of inputs + images) and exponential backoff for rate limits.

3.  **Deterministic Post-Processor (`pipeline/post_processor.py`)**
    *   Applies strict logical overrides (§5 of the spec). E.g., if `valid_image=false`, it forces `claim_status=not_enough_information`.
    *   Implements History-Merge (§2.2): unions visual risk flags with user historical flags, applying the escalation trigger (`manual_review_required`) if a user has historical risk *and* a new visual risk flag is detected.

## Requirements

*   Python 3.10+
*   An OpenAI API Key (`OPENAI_API_KEY`)

## Setup Instructions

1.  Navigate to the `code/` directory:
    ```bash
    cd code/
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Set your OpenAI API Key. You can either export it in your terminal or create a `.env` file in the `code/` directory:
    ```bash
    cp .env.example .env
    # Edit .env and add your key: OPENAI_API_KEY=sk-proj-...
    ```

## Running the Pipeline

To run the pipeline against the full dataset (`dataset/claims.csv`) and generate `output.csv` in the repository root:

```bash
cd code/
python main.py
```

*Note: The pipeline automatically caches VLM responses in `code/.cache/`. If you interrupt and restart the script, it will pick up exactly where it left off without duplicate API calls.*

## Running the Evaluation Harness

To run the two-strategy evaluation (§7 of the spec) against the `dataset/sample_claims.csv` dataset:

```bash
cd code/
python evaluation/main.py
```

This will run the pipeline using both the Zero-Shot and Chain-of-Thought strategies, compare the results against the provided labels, and print a detailed accuracy breakdown to the console.

## Project Structure

```text
.
├── code/
│   ├── main.py                  # Main entry point (generates output.csv)
│   ├── pipeline/                # Core logic modules
│   │   ├── loader.py            # Data loading and image validation
│   │   ├── model_call.py        # VLM interaction, JSON schema, and CoT prompt
│   │   └── post_processor.py    # Deterministic overrides and history merge
│   ├── evaluation/
│   │   ├── main.py              # Evaluation harness (Zero-Shot vs CoT)
│   │   └── evaluation_report.md # Final metrics and strategy justification
│   ├── requirements.txt         # Dependencies
│   └── .env                     # (Ignored) Environment variables
├── dataset/                     # Input datasets and images
├── README.md                    # Root project info
├── problem_statement.md         # Challenge description
└── output.csv                   # (Generated) Final predictions
```
