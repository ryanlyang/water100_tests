# Runtime outputs

Runtime artifacts are written beneath the external `paths.output_root` from the
study configuration. The pipeline will create these subdirectories:

```text
candidate_models/
split_manifests/
patch_masks/
token_banks/
fcv_scores/
control_scores/
selection_results/
plots/
run_logs/
```

Large checkpoints, token banks, score files, and plots are intentionally not
tracked in the source tree.

During Step 4, `candidate_models/` contains one directory per sweep run. Each
directory has 20 float32 epoch checkpoints, `metrics.csv`, `run_summary.json`,
and a resumable optimizer/scheduler state. Keep every epoch checkpoint until
Step 12 rank/scatter analysis has completed successfully.

During Step 7, `fcv_scores/` contains one shared cached donor-index plan plus a
per-image CSV and provenance summary for every candidate. The aggregate CSV
contains only biased-validation and FCV metrics; Oracle and test metrics are
not available to this stage.

During Step 8, `control_scores/` contains one shared control plan, four
per-image control CSVs per candidate, and one per-candidate summary. The pool
aggregate remains validation-only and has no Oracle/test access.

During Step 9, `selection_results/oracle_scores/` contains one analysis-only
original-validation summary per candidate. `candidate_oracle_scores.csv`
strictly indexes the full Oracle pool, `candidate_selector_scores.csv` stores
all validation-only selector inputs/formulas, and `selection_table.csv` records
one deterministically selected checkpoint per selector. Test metrics are not
written until Step 10.

During Step 10, `selection_results/final_test_scores/` contains one resumable
test summary and one hashed per-example logits/predictions CSV per unique
selected checkpoint. `final_test_results.csv` expands
those metrics back to all frozen selector rows without changing their order.
Its summary records both Step 9 hashes and explicitly states that test metrics
did not affect selection.

During Step 11, `selection_results/candidate_pool_test_scores/` contains one
resumable post-hoc test summary and one hashed per-example record file for
every candidate. The strict
`candidate_pool_test_scores.csv` index is complete only at 540 rows and marks
the pool as ineligible for selection. `gap_closure_summary.csv` reports the raw
FCV-to-Oracle gap fraction, selected candidate identities, and the unfair
candidate-pool upper bound. Its JSON sidecar binds the result to the frozen
selection, Step 10 results, pool index, and test-manifest hashes.

During Step 12, `candidate_rank_analysis.csv` stores the one-to-one candidate
join, canonical validation/test metrics, oriented selector scores, and ranks.
`rank_correlation_results.csv` contains one row per locked selector with
descriptive Spearman/Kendall tau-b coefficients, run-cluster bootstrap
intervals, selection regret, and top-k overlap results.
`selector_scatter_plots/` contains six individual plots, one combined grid,
and the biased-validation-versus-test-WGA plot colored by FCV score.
The summary hashes every input and output and records that test metrics did not
affect selection.
