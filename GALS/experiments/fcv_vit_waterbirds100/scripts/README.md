# Scripts

`inspect_config.py` validates the locked study protocol and prints the derived
candidate-pool size.

`prepare_metadata.py` implements the deterministic Step 2 split. It emits
public candidate-training and FCV-validation manifests without context/group
columns, plus clearly marked analysis-only oracle-validation and test
manifests. It also writes `manifest_bundle.json`, which authenticates all four
manifest hashes against the original metadata hash, deterministic split-index
artifact, split summary, and locked holdout configuration. Every downstream
stage validates this bundle and the role-specific `source_split`.

`prepare_patch_masks.py` implements Step 3. It categorically decodes the VOC
RGB R4RR maps, applies validation resize/crop geometry, and emits 14x14 patch
scores plus disjoint evidence/background/ambiguous index sets. It also writes
CSV/JSON eligibility diagnostics and mandatory alignment overlays before any
acceptance failure is raised.

`cache_pretrained_model.py` validates and caches the locked timm checkpoint
before GPUs are requested. `train_candidate.py` trains or resumes one indexed
LR/weight-decay/seed run and saves all 20 epoch candidates.
`submit_step4_candidate_pool.sh` submits the cache job followed by indices
0--26 as a Tigris GH200 Slurm array. `aggregate_candidate_metrics.py` validates
and combines the independent metric files without concurrent CSV writes.

`verify_vit_patch_forward.py` implements the Step 5 acceptance check. It
extracts raw patches before positional embeddings, resumes through the model's
own timm token/position and classifier path, and exits nonzero unless the
reconstructed logits have maximum absolute error below `1e-5`.

`build_background_token_banks.py` implements Step 6. With `--checkpoint` it
builds one candidate's land/water banks; with `--run-index` it processes all 20
epoch checkpoints belonging to one Step 4 sweep run while reusing the same
validated public validation loader and patch-mask artifact.
`submit_step6_token_banks.sh` first persists mandatory pretrained and
real-candidate reconstruction reports, then submits run indices 0--26 as a
Tigris GH200 array. Steps 6--8 require and record those report hashes.
`aggregate_token_banks.py` verifies that every candidate has both context banks
and writes the compact pool index used by Step 7.

`prepare_opposite_donor_plan.py` implements the candidate-independent random
draw cache for Step 7. `score_fcv_candidates.py` applies those draws to one
candidate or all 20 checkpoints in a sweep run and writes auditable per-image
scores. `submit_step7_fcv_scoring.sh` chains donor-plan preparation, a 27-task
Tigris GH200 array, and diagnostic aggregation. `aggregate_fcv_scores.py`
strictly validates the full 540-candidate FCV metric table unless
`--allow-incomplete` is requested for failure diagnosis.

`prepare_control_plan.py` caches the Step 8 same-context donors, matched random
positions, shuffled teacher masks, and opposite-class evidence donors.
`score_fcv_controls.py` scores all four controls with one candidate model load.
`submit_step8_fcv_controls.sh` runs plan preparation, a 27-task Tigris GH200
array, and diagnostic aggregation. `aggregate_fcv_controls.py` validates the
full candidate/control matrix.

`score_oracle_candidates.py` evaluates one checkpoint or one 20-epoch sweep
run on the analysis-only original mixed validation split and writes a hashed
per-example file with labels, groups, logits/probabilities, predictions,
correctness, and loss. `aggregate_oracle_metrics.py` recomputes all selector
metrics from those raw files and verifies the complete 540-candidate Oracle
index. `build_selection_table.py` joins the strict Step 4, 7, 8, and Oracle
indexes and applies every locked Step 9 selector with deterministic ties.
`submit_step9_selectors.sh` submits Oracle scoring, separate partial
diagnostics, and the final strict table build on Tigris. None of these scripts
loads the test manifest; final test evaluation remains Step 10.

`evaluate_selected_checkpoints.py` implements Step 10. Its
`--validate-selection-only` mode verifies and freezes Step 9 without opening
test data. Normal mode loads only the analysis-marked test manifest, evaluates
each unique selected checkpoint once, writes hashed per-example logits and
predictions, recomputes aggregate metrics from them, and writes
`final_test_results.csv` in the original selector order.
`submit_step10_final_test.sh` validates selection
on the submit node and launches the resumable Tigris GH200 evaluation job.

`score_pool_test_candidates.py` implements the Step 11 post-hoc test pass over
one checkpoint or one complete 20-epoch sweep run. Each candidate path and
SHA-256 must match the complete mapping frozen by Step 9 before test access.
Each candidate also emits hashed per-example logits/predictions from which the
pool aggregate recomputes its metrics.
`aggregate_pool_test_scores.py` verifies all 540 provenance-bound summaries;
`--allow-incomplete` is diagnostic only. `compute_gap_closure.py` revalidates
the frozen Step 9 and Step 10 artifacts, computes the raw gap-closure fraction,
and reports the deterministic full-pool upper bound. The upper-bound scores are
explicitly analysis-only and never become selector inputs.
`submit_step11_gap_closure.sh` launches the 27-task GH200 pool array, a partial
diagnostic aggregate, and the strict final gap job on Tigris.

`analyze_rank_quality.py` implements Step 12. It strictly joins the frozen
Step 9 selector matrix to the complete Step 11 candidate outcomes by candidate
ID, path, and checkpoint SHA-256, reproduces
all frozen choices, computes oriented Spearman/Kendall correlations, regret,
and fixed top-k overlap with run-cluster bootstrap intervals, then writes
individual/combined plots and the biased-validation-versus-test plot colored by
FCV score. Naive epoch-independent correlation p-values are not reported.
`submit_step12_rank_analysis.sh` revalidates Step 11 and submits the CPU-only
`fcv_vit_rank` Tigris job.
