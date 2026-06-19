I'm working on the HackerRank Orchestrate challenge in this repo. Before doing anything
else, read AGENTS.md, README.md, and problem_statement.md in full, then read
code/SOLUTION_SPEC.md in full — it's a supplementary spec I've grounded in the actual
dataset, and it resolves several ambiguities you'd otherwise have to guess at. Follow
the onboarding flow in AGENTS.md §3 if it hasn't run yet for this repo root.

SOLUTION_SPEC.md is the implementation contract. In particular, don't skip or
soften: the valid_image vs evidence_standard_met distinction (§2.1), the risk_flags
history-merge rule (§2.2), the image-format-sniffing requirement (§2.3), the
prompt-injection handling (§2.4), the model call's forced structured-tool-output
approach (§4), and the deterministic post-processing rules (§5) — these aren't
optional style preferences, they're the difference between matching the labeled
behavior in sample_claims.csv and not.

Build the full solution under code/. Start by telling me your planned file layout
under code/ (don't write code yet). Then build incrementally and show me output after
each stage:

1. Loader + image validation/conversion. Sanity-check by running it against every row
   of sample_claims.csv and printing which images failed to decode or convert, before
   any model call exists.
2. The model call (§4) against a handful of sample_claims.csv rows only, so we can
   eyeball outputs against the known labels before running the full set.
3. The deterministic post-processor (§5). Re-run against the same handful of rows and
   show me the diff in outputs before vs after post-processing.
4. The full evaluation harness (§7) against all of sample_claims.csv, with the metrics
   and the two-strategy comparison. Show me the metrics before writing
   evaluation_report.md.
5. The final run against dataset/claims.csv producing output.csv, plus code/README.md
   and the saved prompts/configs.

Append §9 of SOLUTION_SPEC.md (open questions / assumptions log) as you go, any time
you have to make a judgment call that isn't already pinned down by the spec.

Do not special-case any user_id, case folder, or filename anywhere in the code —
sample_claims.csv is for evaluation only, never for hardcoding answers. Read all
secrets from environment variables only.