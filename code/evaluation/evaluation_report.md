# Evaluation Report: Multi-Modal Evidence Review

## 1. Executive Summary

This report evaluates two pipeline strategies for the HackerRank Orchestrate challenge (Multi-Modal Evidence Review). The system evaluates damage claims by processing user inputs, historical risk data, and image evidence against specific evidence requirements using a Vision-Language Model (GPT-4o) and a deterministic post-processor.

We evaluated two strategies against a 20-row labeled dataset (`dataset/sample_claims.csv`):
*   **Strategy A (Zero-Shot):** The model is given a standard prompt with context and schema constraints, directly generating the required JSON structure.
*   **Strategy B (Chain-of-Thought):** The model is instructed to internally reason step-by-step (assessing images, extracting the claim, matching evidence, checking consistency, verifying damage, assessing risk, and finalizing the decision) before outputting the final JSON.

**Conclusion:** **Strategy B (Chain-of-Thought)** was selected as the final production strategy because it demonstrated higher overall accuracy across key fields, particularly in complex cases requiring cross-image correlation.

## 2. Two-Strategy Comparison Results

The evaluation was run across all 20 rows of `sample_claims.csv`. Accuracy is measured against the ground-truth labels provided in the dataset.

### Head-to-Head Comparison (Key 3 Fields)

| Field | Strategy A (Zero-Shot) | Strategy B (Chain-of-Thought) | Winner |
| :--- | :--- | :--- | :--- |
| `claim_status` | 65.0% | **70.0%** | **CoT** |
| `issue_type` | 55.0% | 55.0% | Tie |
| `object_part` | 80.0% | **85.0%** | **CoT** |
| **AVERAGE** | 66.7% | **70.0%** | **CoT** |

### Detailed Field-by-Field Breakdown

| Field | Zero-Shot Accuracy | Chain-of-Thought Accuracy | Delta |
| :--- | :--- | :--- | :--- |
| `valid_image` | 90.0% | **95.0%** | +5.0% |
| `evidence_standard_met` | 75.0% | **80.0%** | +5.0% |
| `risk_flags` | 55.0% | 55.0% | 0.0% |
| `issue_type` | 55.0% | 55.0% | 0.0% |
| `object_part` | 80.0% | **85.0%** | +5.0% |
| `claim_status` | 65.0% | **70.0%** | +5.0% |
| `severity` | 45.0% | **55.0%** | +10.0% |
| `supporting_image_ids` | 70.0% | 70.0% | 0.0% |

## 3. Strategy Justification

**Strategy B (Chain-of-Thought)** is superior and selected for the final run for the following reasons:

1.  **Higher Overall Accuracy:** CoT outperformed Zero-Shot on almost all metrics, achieving a 70.0% average across the 3 key fields (vs 66.7%). It improved `claim_status` by 5%, `object_part` by 5%, and `severity` by 10%.
2.  **Better Logical Consistency in Complex Cases:** CoT was better able to handle cases with multiple images where one image contradicted another (e.g., `user_002` where two different cars were shown). The step-by-step reasoning phase forces the model to evaluate "CROSS-IMAGE CONSISTENCY" before jumping to a conclusion, fixing a critical failure mode present in the Zero-Shot run.
3.  **Cost-Benefit Tradeoff:** The performance gains came at a very acceptable operational cost. The token overhead was minimal (only ~11% increase), which is a worthwhile trade-off for higher classification accuracy on critical fields like `claim_status`.

## 4. Operational Analysis

The following metrics summarize the resource usage of the two strategies on the 20-row sample dataset using the `gpt-4o` model.

| Metric | Strategy A (Zero-Shot) | Strategy B (Chain-of-Thought) |
| :--- | :--- | :--- |
| Total API Calls | 20 | 20 |
| Total Tokens | 48,846 | 54,252 (+11%) |
| Runtime (approx.) | ~90s | ~92s |
| Estimated Cost ($2.50/1M in, $10/1M out) | ~$0.15 | ~$0.17 |

*Note: Cost assumes the majority of tokens are prompt (input) tokens due to the base64 encoded images. The JSON structured output is consistently short (usually around 100-150 tokens).*

## 5. Identified Failure Modes and Future Improvements

Despite the improvements from CoT and the deterministic post-processor, some persistent failure modes remain:

1.  **Issue Type Confusion:** `issue_type` accuracy remained tied at 55%. The model frequently confuses closely related categories (e.g., classifying a `broken_part` as a `crack` or `scratch`, or `stain` as `water_damage`). Future improvements could include providing visual examples (few-shot prompting) in the system prompt for each category.
2.  **Severity Overestimation:** The model exhibits a bias towards overestimating severity (often predicting `high` when the label is `medium`). Severity is highly subjective. Tighter definitional constraints in the prompt could help calibrate this.
3.  **Risk Flag Sensitivity:** The model tends to under-detect subtle visual risk flags like `claim_mismatch` or `damage_not_visible` (accuracy at 55%). While the history-merge logic catches user-level risks, the visual detection of these edge cases could be improved by breaking the VLM call into a multi-agent workflow (e.g., one agent specifically hunting for inconsistencies).

## 6. Deterministic Post-Processing Impact

The evaluation results *include* the deterministic post-processor (§5 of the spec). The post-processor proved essential, implementing the following rules which improved the baseline VLM output:
*   **Invalid Image Override:** Forcing `evidence_standard_met=false` and `claim_status=not_enough_information` when `valid_image=false`.
*   **Evidence Insufficiency Override:** Forcing `claim_status=not_enough_information` and adding the `manual_review_required` flag when evidence requirements are not met.
*   **Contradiction Override:** Setting `claim_status=contradicted` if no issue is found (`issue_type=none`) but the object part is visible.
*   **NEI Severity Override:** Forcing `severity=unknown` when the claim status is `not_enough_information`.
*   **Risk Flags History-Merge:** Reliably combining VLM-detected visual flags with historical flags from `user_history.csv` and applying escalation triggers.
