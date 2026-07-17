# Runtime outputs

Runtime artifacts are written beneath `paths.output_root`, outside the source
tree. The online protocol uses:

```text
split_manifests/
patch_masks/
preflight/
online_runs/
online_scores/fcv/
online_scores/controls/
online_scores/oracle/
online_test_analysis_only/
selection_results/
plots/
run_logs/
```

`preflight/online_campaign_provenance.json` is the immutable campaign trust
root. Every resume state, per-candidate summary, completed run, selection
receipt, and post-hoc summary is required to bind to its exact SHA-256. The
two real online smoke receipts in `preflight/` prove an epoch-1 interruption
and epoch-2 resume before the full array starts. They record full online-epoch
wall time and separate durable FCV/control/Oracle/test evidence from retained
checkpoints, resume states, and plans. Their projections must fit both the
eight-writer concurrency reserve and the complete 540-candidate 35 GiB launch
guard, while projected per-run time must fit the seven-day limit. The aggregate
`online_smoke_gate_receipt.json` binds both receipts to the current campaign
and makes a repeated full-launch command idempotent after run 0 advances.

`online_runs/<run_id>/` stores `validation_metrics.csv`, the separate
`test_metrics_analysis_only.csv`, deterministic intervention plans, a
`retention_state.json`, and at most three steady-state retained checkpoint
files. Replacing a winner may briefly stage one fourth file until the atomic
resume commit; an interrupted transaction is pruned on restart. During an
active run the directory also contains one atomic `resume_state.pt`; that file
is deleted when the run completes.

The checkpoint for the current epoch and its model-specific token banks are
created under node-local scratch. They are deleted after FCV, controls,
Oracle, test analysis, and the atomic resume commit complete. Thus the output
tree never accumulates 540 model files or token-bank pairs.

`selection_results/online_unprivileged_freeze_receipt.json` binds the
train-holdout selector matrix and choices before Oracle evidence is opened.
`online_control_diagnostics.csv` and its summary surface control warnings and
donor/token-distribution diagnostics for every candidate. The subsequent
`online_selection_summary.json` proves all validation selections were frozen
without opening test artifacts. The dependent post-hoc phase then writes
selected test results, full-pool test results, gap closure, rank analysis, and
scatter plots. Every compact row is revalidated against its hashed per-image
evidence before reporting.

Selection freeze also writes `online_checkpoint_cleanup_plan.json` before any
deletion and `online_checkpoint_cleanup_receipt.json` afterward. Non-global
per-run winners are removed at this boundary, leaving at most three unique
globally selected primary checkpoints. The post-hoc analysis refuses to run if
the receipt, surviving hashes, or recorded deletions do not validate.
