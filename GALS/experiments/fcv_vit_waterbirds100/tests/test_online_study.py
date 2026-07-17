from __future__ import annotations

import hashlib
import json
import copy
import random
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))

from fcv.online_schema import (
    ONLINE_TEST_COLUMNS,
    ONLINE_VALIDATION_COLUMNS,
    RETAINED_SELECTOR_SPECS,
)
from fcv.campaign_provenance import (  # noqa: E402
    CampaignProvenanceError,
    create_campaign_provenance_receipt,
    load_campaign_provenance_receipt,
)
from fcv.candidate_training import (  # noqa: E402
    SweepRun,
    _rng_state,
    pretrained_provenance_path,
    state_dict_sha256,
)
from fcv.online_analysis import (  # noqa: E402
    analyze_online_test_results,
    _cleanup_non_global_retained_checkpoints,
    _concurrent_storage_projection,
    _runtime_projection,
    _storage_high_water_projection,
    ensure_online_smoke_gate,
    freeze_online_validation_selection,
    OnlineAnalysisError,
    validate_reusable_online_smoke_receipt,
)
from fcv.online_study import (
    OnlineStudyError,
    _append_test_index_row,
    _completed_run,
    _invalidate_completed_run_for_bounded_repair,
    _isolated_training_rng,
    _local_winners,
    _prepare_online_intervention_plans,
    _prune_retention,
    _restore_rng_state,
    _run_epoch,
    _stage_retention,
    _test_index_prefix,
    _validation_row,
    train_and_score_online_run,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(
    candidate_id: str,
    checkpoint_sha256: str,
    biased: float,
    fcv: float,
    oracle: float,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "checkpoint_sha256": checkpoint_sha256,
        "biased_val_accuracy": biased,
        "primary_selector_score": fcv,
        "oracle_validation_balanced_group_accuracy": oracle,
    }


class OnlineStudyRetentionTests(unittest.TestCase):
    def test_resume_commit_reproduces_all_next_epoch_tensors_and_metrics(self) -> None:
        random.seed(23)
        np.random.seed(23)
        torch.manual_seed(23)
        features = torch.randn(12, 4)
        labels = torch.tensor([0, 1] * 6)
        dataset = TensorDataset(features, labels, torch.arange(len(labels)))
        generator = torch.Generator(device="cpu").manual_seed(23)
        loader = DataLoader(
            dataset, batch_size=3, shuffle=True, generator=generator
        )
        model = nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-2)
        criterion = nn.CrossEntropyLoss()
        _run_epoch(
            model,
            loader,
            criterion,
            torch.device("cpu"),
            "float32",
            optimizer=optimizer,
        )
        committed_model = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        committed_optimizer = copy.deepcopy(optimizer.state_dict())
        committed_rng = _rng_state(generator)

        reference_metrics = _run_epoch(
            model,
            loader,
            criterion,
            torch.device("cpu"),
            "float32",
            optimizer=optimizer,
        )
        reference_state = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }

        resumed_model = nn.Linear(4, 2)
        resumed_model.load_state_dict(committed_model, strict=True)
        resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1.0e-2)
        resumed_optimizer.load_state_dict(committed_optimizer)
        resumed_generator = torch.Generator(device="cpu")
        resumed_loader = DataLoader(
            dataset, batch_size=3, shuffle=True, generator=resumed_generator
        )
        _restore_rng_state(committed_rng, resumed_generator)
        resumed_metrics = _run_epoch(
            resumed_model,
            resumed_loader,
            criterion,
            torch.device("cpu"),
            "float32",
            optimizer=resumed_optimizer,
        )
        self.assertEqual(reference_metrics, resumed_metrics)
        for key, reference in reference_state.items():
            self.assertTrue(
                torch.equal(reference, resumed_model.state_dict()[key]),
                msg=f"Resume changed model tensor {key}",
            )

    def test_campaign_receipt_rejects_changed_shared_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = {
                "paths": {"output_root": temporary},
                "outputs": {
                    "split_manifests": "split_manifests",
                    "patch_masks": "patch_masks",
                },
            }
            original = {
                "training_fingerprint": "a" * 64,
                "pretrained": {"sha256": "b" * 64},
                "manifests": {"biased_validation": {"sha256": "c" * 64}},
            }
            with mock.patch(
                "fcv.campaign_provenance._current_bindings",
                return_value=original,
            ), mock.patch(
                "fcv.campaign_provenance.verify_non_test_campaign_inputs"
            ), mock.patch(
                "fcv.test_evaluation.prepare_final_test_source"
            ):
                created = create_campaign_provenance_receipt(
                    config,
                    pretrained_path=Path(temporary) / "pretrained.json",
                    verify_all_image_bytes=True,
                )
                loaded = load_campaign_provenance_receipt(
                    config,
                    pretrained_path=Path(temporary) / "pretrained.json",
                )
            self.assertEqual(created["bindings_sha256"], loaded["bindings_sha256"])

            changed = dict(original)
            changed["pretrained"] = {"sha256": "d" * 64}
            with mock.patch(
                "fcv.campaign_provenance._current_bindings",
                return_value=changed,
            ), self.assertRaises(CampaignProvenanceError):
                load_campaign_provenance_receipt(
                    config,
                    pretrained_path=Path(temporary) / "pretrained.json",
                )

    def test_smoke_storage_projection_scales_all_writers_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary).resolve()
            run_dir = output_root / "online_runs" / "run"
            retained = run_dir / "retained_checkpoints"
            retained.mkdir(parents=True)
            (retained / "winner.pt").write_bytes(b"x" * 100)
            (run_dir / "resume_state.pt").write_bytes(b"r" * 100)
            plans = run_dir / "plans"
            plans.mkdir()
            (plans / "plan.pt").write_bytes(b"p" * 100)
            candidate_id = "candidate"
            for path in (
                output_root / "online_scores" / "fcv",
                output_root / "online_scores" / "controls",
                output_root / "online_scores" / "oracle",
                output_root / "online_test_analysis_only",
            ):
                path.mkdir(parents=True, exist_ok=True)
                (path / f"{candidate_id}_evidence.csv").write_bytes(b"e" * 100)
            baseline_path = output_root / "baseline.json"
            baseline_path.write_text(
                json.dumps(
                    {
                        "output_root": str(output_root),
                        "allocated_bytes": 0,
                        "storage_breakdown_bytes": {
                            "categorized_total": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = {
                "storage": {
                    "max_concurrent_streaming_runs": 8,
                    "worst_case_concurrent_growth_gib": 1.0,
                    "full_campaign_projection_safety_factor": 2.0,
                    "launch_guard_gib": 1.0,
                },
                "candidate_pool": {
                    "expected_candidate_checkpoints": 1,
                    "expected_training_runs": 1,
                    "max_retained_checkpoints_per_run": 3,
                }
            }
            observed = _storage_high_water_projection(
                config,
                output_root,
                run_dir,
                baseline_path,
                candidate_id,
                1,
            )
            self.assertGreater(observed["observed_single_epoch_growth_bytes"], 0)
            self.assertGreater(observed["candidate_evidence_bytes"], 0)
            self.assertGreater(observed["projected_full_campaign_bytes"], 0)
            self.assertTrue(observed["within_full_campaign_launch_guard"])

            config["storage"]["worst_case_concurrent_growth_gib"] = 1.0e-7
            with self.assertRaises(OnlineAnalysisError):
                _storage_high_water_projection(
                    config,
                    output_root,
                    run_dir,
                    baseline_path,
                    candidate_id,
                    1,
                )

    def test_concurrent_projection_accepts_realistic_vit_s_state_sizes(self) -> None:
        gib = 1024 ** 3
        resume_bytes = 260_202_496
        checkpoint_bytes = 86_715_237
        observed_committed_growth = resume_bytes + checkpoint_bytes
        projected = _concurrent_storage_projection(
            observed_committed_growth_bytes=observed_committed_growth,
            transient_checkpoint_bytes_per_writer=checkpoint_bytes,
            writers=8,
            reserve_bytes=5 * gib,
        )
        expected = (observed_committed_growth + checkpoint_bytes) * 8
        self.assertEqual(projected["projected_concurrent_growth_bytes"], expected)
        self.assertLess(expected, 5 * gib)
        self.assertGreater(
            (observed_committed_growth + checkpoint_bytes) * 8 * 2,
            5 * gib,
        )
        with self.assertRaises(OnlineAnalysisError):
            _concurrent_storage_projection(
                observed_committed_growth_bytes=observed_committed_growth,
                transient_checkpoint_bytes_per_writer=checkpoint_bytes,
                writers=8,
                reserve_bytes=3 * gib,
            )

    def test_runtime_projection_enforces_seven_day_limit(self) -> None:
        config = {
            "training": {"epochs": 20},
            "cluster": {
                "runtime_projection_safety_factor": 1.5,
                "online_run_time_limit_hours": 168,
            },
        }
        frame = pd.DataFrame({"epoch_online_total_seconds": [60.0, 90.0]})
        projection = _runtime_projection(config, frame)
        self.assertEqual(projection["projected_run_seconds"], 90.0 * 20 * 1.5)
        self.assertTrue(projection["within_run_time_limit"])

        frame["epoch_online_total_seconds"] = [25000.0, 25000.0]
        with self.assertRaises(OnlineAnalysisError):
            _runtime_projection(config, frame)

    def test_smoke_gate_is_campaign_bound_and_restart_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary).resolve()
            preflight = output_root / "preflight"
            preflight.mkdir(parents=True)
            config = {"paths": {"output_root": str(output_root)}}
            campaign = {
                "artifact_path": str(preflight / "campaign.json"),
                "artifact_sha256": "a" * 64,
                "bindings_sha256": "b" * 64,
            }
            for epoch in (1, 2):
                receipt = {
                    "artifact_type": "fcv_vit_online_real_smoke_receipt",
                    "status": "passed",
                    "run_index": 0,
                    "validated_epoch_prefix": epoch,
                    "required_resumed_from_epoch": None if epoch == 1 else 1,
                    "campaign_provenance_path": campaign["artifact_path"],
                    "campaign_provenance_sha256": campaign["artifact_sha256"],
                    "campaign_bindings_sha256": campaign["bindings_sha256"],
                    "checkpoint_retention_expanded": False,
                    "storage_high_water_projection": {
                        "within_configured_reserve": True,
                        "within_full_campaign_launch_guard": True,
                    },
                    "runtime_projection": {"within_run_time_limit": True},
                }
                receipt_path = (
                    preflight
                    / f"online_smoke_run_000_epoch_{epoch:03d}.json"
                )
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

            with mock.patch(
                "fcv.online_analysis.load_campaign_provenance_receipt",
                return_value=campaign,
            ):
                epoch_one = validate_reusable_online_smoke_receipt(
                    config, output_root, expected_epoch=1
                )
                created = ensure_online_smoke_gate(config, output_root)
                reused = ensure_online_smoke_gate(config, output_root)
            self.assertEqual(epoch_one["receipt_status"], "reused")
            self.assertEqual(created["gate_status"], "created")
            self.assertEqual(reused["gate_status"], "reused")
            self.assertEqual(created["gate_sha256"], reused["gate_sha256"])

            epoch_two_path = preflight / "online_smoke_run_000_epoch_002.json"
            epoch_two = json.loads(epoch_two_path.read_text(encoding="utf-8"))
            epoch_two["runtime_projection"]["within_run_time_limit"] = False
            epoch_two_path.write_text(json.dumps(epoch_two), encoding="utf-8")
            with mock.patch(
                "fcv.online_analysis.load_campaign_provenance_receipt",
                return_value=campaign,
            ), self.assertRaises(OnlineAnalysisError):
                ensure_online_smoke_gate(config, output_root)

    def test_cache_and_online_runner_share_one_pretrained_provenance_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = {"paths": {"output_root": temporary}}
            self.assertEqual(
                pretrained_provenance_path(config),
                Path(temporary).resolve()
                / "preflight"
                / "pretrained_model_summary.json",
            )

    def test_validation_and_test_schemas_are_disjoint(self) -> None:
        self.assertFalse(
            any(column.startswith("test_") for column in ONLINE_VALIDATION_COLUMNS)
        )
        self.assertTrue(any(column.startswith("test_") for column in ONLINE_TEST_COLUMNS))
        self.assertEqual(
            list(RETAINED_SELECTOR_SPECS),
            [
                "biased_validation_accuracy",
                "equal_weight_original_and_opposite_fcv_accuracy",
                "oracle_validation_balanced_group_accuracy",
            ],
        )

    def test_canonical_validation_loss_uses_per_example_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            score_path = root / "scores.csv"
            pd.DataFrame(
                {
                    "fcv_eligible": [True, True],
                    "p_y_counterfactual_mean": [0.4, 0.6],
                    "p_y_original": [0.5, 0.75],
                }
            ).to_csv(score_path, index=False)
            artifacts = []
            for name in ("fcv.json", "controls.json", "oracle.json"):
                path = root / name
                path.write_text("{}\n", encoding="utf-8")
                artifacts.append(path)
            canonical = 0.5000000113
            batch_reduced = 0.5
            row = _validation_row(
                {
                    "model": {"name": "test"},
                    "fcv": {
                        "selector_analysis": {
                            "probability_ratio_epsilon": 1.0e-8,
                            "control_normalized_lambda": 1.0,
                        }
                    },
                },
                SweepRun(0, 1.0e-5, 0.01, 0),
                1,
                {"loss": 0.2, "accuracy": 0.9},
                {"loss": batch_reduced, "accuracy": 0.8},
                1.0e-5,
                9.0e-6,
                "a" * 64,
                1.0,
                2.0,
                {
                    "score_csv_path": str(score_path),
                    "biased_validation_loss_recomputed": canonical,
                    "opposite_context_counterfactual_accuracy": 0.7,
                    "opposite_context_counterfactual_majority_accuracy": 0.7,
                    "opposite_context_true_class_probability": 0.6,
                    "mean_counterfactual_confidence_drop": 0.1,
                    "primary_selector_score": 0.75,
                    "token_distribution_diagnostics": {
                        "global_means": {
                            "target_donor_cosine_similarity_mean": 0.1,
                            "target_nearest_donor_cosine_mean": 0.2,
                            "donor_unique_source_images": 3.0,
                            "donor_max_source_fraction": 0.4,
                        }
                    },
                    "real_swap_integrity_diagnostics": {
                        "replaced_token_changed_fraction": 1.0,
                        "replacement_delta_mean": 0.3,
                        "replacement_delta_max": 0.8,
                        "foreground_token_max_abs_error": 0.0,
                        "donor_reconstruction_max_abs_error": 0.0,
                    },
                },
                {
                    "controls": {
                        "same_context": {
                            "counterfactual_accuracy": 0.75,
                            "mean_confidence_drop": 0.02,
                        },
                        "random_mask": {"counterfactual_accuracy": 0.71},
                        "shuffled_mask": {"counterfactual_accuracy": 0.72},
                        "evidence_swap": {"counterfactual_accuracy": 0.73},
                    },
                    "diagnostic_warning_count": 0,
                    "diagnostic_status": "passed",
                },
                {
                    "metrics": {
                        "loss": 0.4,
                        "accuracy": 0.8,
                        "balanced_group_accuracy": 0.7,
                        "worst_group_accuracy": 0.6,
                        **{f"group_{group}_accuracy": 0.6 for group in range(4)},
                    }
                },
                artifacts[0],
                artifacts[1],
                artifacts[2],
            )
            self.assertEqual(row["biased_val_loss"], canonical)
            self.assertEqual(
                row["biased_val_loss_batch_reduced_diagnostic"], batch_reduced
            )
            self.assertGreater(abs(canonical - batch_reduced), 1.0e-9)
            self.assertLess(abs(canonical - batch_reduced), 1.0e-6)

    def test_synthetic_online_producer_freeze_and_posthoc_pipeline(self) -> None:
        """Exercise the complete online API chain with a tiny real train epoch."""

        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary).resolve()
            run = SweepRun(0, 1.0e-3, 1.0e-4, 0)
            config = {
                "study": {"name": "synthetic-online-integration"},
                "paths": {"output_root": str(output_root)},
                "outputs": {"plots": "plots"},
                "model": {"name": "tiny", "pretrained": True},
                "training": {
                    "epochs": 1,
                    "seeds": [0],
                    "learning_rates": [run.learning_rate],
                    "weight_decays": [run.weight_decay],
                    "precision": "float32",
                    "num_workers": 0,
                    "augmentation": {"label_smoothing": 0.0},
                },
                "candidate_pool": {
                    "expected_training_runs": 1,
                    "expected_candidate_checkpoints": 1,
                    "max_final_retained_checkpoints": 3,
                },
                "reproducibility": {
                    "deterministic_algorithms": False,
                    "cudnn_benchmark": False,
                },
                "storage": {
                    "max_concurrent_streaming_runs": 1,
                    "worst_case_concurrent_growth_gib": 1.0,
                },
                "execution": {
                    "fcv_counterfactual_forward_batch_size": 2,
                    "control_target_batch_size": 2,
                    "control_counterfactual_forward_batch_size": 2,
                },
                "fcv": {
                    "primary_selector": {
                        "name": "equal_weight_original_and_opposite_fcv_accuracy"
                    },
                    "selector_analysis": {
                        "probability_ratio_epsilon": 1.0e-8,
                        "control_normalized_lambda": 1.0,
                        "fcv_accuracy_lambdas": [],
                    },
                },
                "evaluation": {
                    "final_test": {"precision": "float32"},
                    "gap_closure": {
                        "biased_selector": "biased_validation_accuracy",
                        "fcv_selector": (
                            "equal_weight_original_and_opposite_fcv_accuracy"
                        ),
                        "oracle_selector": (
                            "oracle_validation_balanced_group_accuracy"
                        ),
                        "denominator_epsilon": 1.0e-12,
                    },
                    "rank_analysis": {
                        "selectors": [],
                        "create_scatter_plots": False,
                    },
                },
            }

            def tiny_model(*args, **kwargs) -> nn.Module:
                model = nn.Linear(2, 2)
                with torch.no_grad():
                    model.weight.copy_(
                        torch.tensor([[0.2, -0.1], [-0.3, 0.4]])
                    )
                    model.bias.copy_(torch.tensor([0.05, -0.05]))
                return model

            initial_sha = state_dict_sha256(tiny_model().state_dict())
            campaign = {
                "artifact_path": str(output_root / "campaign.json"),
                "artifact_sha256": "a" * 64,
                "bindings_sha256": "b" * 64,
                "bindings": {
                    "training_fingerprint": "fingerprint",
                    "software_versions": {"runtime": "test"},
                    "software_fingerprint": "software",
                    "source_tree": {"source_tree_sha256": "source"},
                    "pretrained": {
                        "path": str(output_root / "pretrained.json"),
                        "sha256": "c" * 64,
                        "backbone_sha256": "d" * 64,
                    },
                    "initialization": {
                        "initial_model_state_sha256_by_seed": {"0": initial_sha}
                    },
                    "manifests": {
                        role: {
                            "sha256": f"{index + 1}" * 64,
                            "bundle_sha256": f"{index + 5}" * 64,
                        }
                        for index, role in enumerate(
                            (
                                "candidate_train",
                                "biased_validation",
                                "oracle_validation",
                                "test",
                            )
                        )
                    },
                    "patch_masks": {
                        "sha256": "e" * 64,
                        "summary_sha256": "f" * 64,
                        "preprocessing_config_sha256": "0" * 64,
                        "teacher_maps_sha256": "1" * 64,
                    },
                },
            }
            pretrained = {
                "artifact_path": campaign["bindings"]["pretrained"]["path"],
                "artifact_sha256": campaign["bindings"]["pretrained"]["sha256"],
                "pretrained_backbone_sha256": campaign["bindings"]["pretrained"][
                    "backbone_sha256"
                ],
            }
            manifest_hashes = {
                "candidate_train": campaign["bindings"]["manifests"][
                    "candidate_train"
                ]["sha256"],
                "biased_validation": campaign["bindings"]["manifests"][
                    "biased_validation"
                ]["sha256"],
            }
            features = torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]]
            )
            labels = torch.tensor([0, 1, 0, 1])
            dataset = TensorDataset(features, labels, torch.arange(4))
            train_loader = DataLoader(dataset, batch_size=2, shuffle=False)
            val_loader = DataLoader(dataset, batch_size=2, shuffle=False)

            candidate_id = run.candidate_id(1)

            def fake_fcv(*args, **kwargs):
                fcv_dir = Path(args[5])
                fcv_dir.mkdir(parents=True, exist_ok=True)
                score_path = fcv_dir / f"{candidate_id}_scores.csv"
                pd.DataFrame(
                    {
                        "fcv_eligible": [True, True],
                        "p_y_counterfactual_mean": [0.7, 0.6],
                        "p_y_original": [0.8, 0.75],
                    }
                ).to_csv(score_path, index=False)
                summary = {
                    "score_csv_path": str(score_path),
                    "biased_validation_loss_recomputed": 0.5,
                    "opposite_context_counterfactual_accuracy": 0.7,
                    "opposite_context_counterfactual_majority_accuracy": 0.7,
                    "opposite_context_true_class_probability": 0.65,
                    "mean_counterfactual_confidence_drop": 0.1,
                    "primary_selector_score": 0.75,
                    "token_distribution_diagnostics": {
                        "global_means": {
                            "target_donor_cosine_similarity_mean": 0.1,
                            "target_nearest_donor_cosine_mean": 0.2,
                            "donor_unique_source_images": 2.0,
                            "donor_max_source_fraction": 0.5,
                        }
                    },
                    "real_swap_integrity_diagnostics": {
                        "replaced_token_changed_fraction": 1.0,
                        "replacement_delta_mean": 0.2,
                        "replacement_delta_max": 0.4,
                        "foreground_token_max_abs_error": 0.0,
                        "donor_reconstruction_max_abs_error": 0.0,
                    },
                }
                (fcv_dir / f"{candidate_id}_summary.json").write_text(
                    json.dumps(summary), encoding="utf-8"
                )
                return summary

            def fake_controls(*args, **kwargs):
                control_dir = Path(args[7])
                control_dir.mkdir(parents=True, exist_ok=True)
                summary = {
                    "controls": {
                        "same_context": {
                            "counterfactual_accuracy": 0.76,
                            "mean_confidence_drop": 0.02,
                        },
                        "random_mask": {"counterfactual_accuracy": 0.71},
                        "shuffled_mask": {"counterfactual_accuracy": 0.72},
                        "evidence_swap": {"counterfactual_accuracy": 0.73},
                    },
                    "diagnostic_warning_count": 0,
                    "diagnostic_status": "passed",
                }
                (control_dir / f"{candidate_id}_controls_summary.json").write_text(
                    json.dumps(summary), encoding="utf-8"
                )
                return summary

            def fake_oracle(*args, **kwargs):
                oracle_dir = Path(args[3])
                oracle_dir.mkdir(parents=True, exist_ok=True)
                summary = {
                    "metrics": {
                        "loss": 0.4,
                        "accuracy": 0.8,
                        "balanced_group_accuracy": 0.7,
                        "worst_group_accuracy": 0.6,
                        **{f"group_{group}_accuracy": 0.6 for group in range(4)},
                    }
                }
                (oracle_dir / f"{candidate_id}_oracle_summary.json").write_text(
                    json.dumps(summary), encoding="utf-8"
                )
                return summary

            test_metrics = {
                "loss": 0.3,
                "accuracy": 0.82,
                "balanced_group_accuracy": 0.72,
                "worst_group_accuracy": 0.62,
                **{f"group_{group}_accuracy": 0.62 for group in range(4)},
                **{f"group_{group}_count": 1 for group in range(4)},
            }
            test_source = SimpleNamespace(
                manifest_path=output_root / "test.csv",
                manifest_sha256=campaign["bindings"]["manifests"]["test"][
                    "sha256"
                ],
                manifest_bundle_sha256=campaign["bindings"]["manifests"]["test"][
                    "bundle_sha256"
                ],
                batch_size=2,
                num_workers=0,
            )

            patches = [
                mock.patch(
                    "fcv.online_study._validate_online_config"
                ),
                mock.patch(
                    "fcv.online_study.load_campaign_provenance_receipt",
                    return_value=campaign,
                ),
                mock.patch(
                    "fcv.online_study._manifest_hashes", return_value=manifest_hashes
                ),
                mock.patch(
                    "fcv.online_study.software_versions",
                    return_value=campaign["bindings"]["software_versions"],
                ),
                mock.patch(
                    "fcv.online_study.software_fingerprint", return_value="software"
                ),
                mock.patch(
                    "fcv.online_study.source_tree_provenance",
                    return_value=campaign["bindings"]["source_tree"],
                ),
                mock.patch(
                    "fcv.online_study.candidate_training_fingerprint",
                    return_value="fingerprint",
                ),
                mock.patch(
                    "fcv.online_study.load_pretrained_cache_provenance",
                    return_value=pretrained,
                ),
                mock.patch("fcv.online_study.PublicManifestDataset"),
                mock.patch(
                    "fcv.online_study.prepare_token_bank_source",
                    return_value=SimpleNamespace(),
                ),
                mock.patch(
                    "fcv.online_study.prepare_oracle_validation_source",
                    return_value=SimpleNamespace(),
                ),
                mock.patch(
                    "fcv.online_study.build_dataloaders",
                    return_value=(
                        {"train": train_loader, "biased_val": val_loader},
                        {},
                    ),
                ),
                mock.patch("fcv.online_study.build_model", side_effect=tiny_model),
                mock.patch(
                    "fcv.online_study.pretrained_backbone_sha256",
                    return_value=pretrained["pretrained_backbone_sha256"],
                ),
                mock.patch(
                    "fcv.online_study.build_scheduler",
                    side_effect=lambda optimizer, *_: (
                        torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
                    ),
                ),
                mock.patch("fcv.online_study.assert_storage_budget"),
                mock.patch("fcv.online_study.build_background_token_banks"),
                mock.patch("fcv.online_study._prepare_online_intervention_plans"),
                mock.patch(
                    "fcv.online_study.score_candidate_fcv", side_effect=fake_fcv
                ),
                mock.patch(
                    "fcv.online_study.score_candidate_controls",
                    side_effect=fake_controls,
                ),
                mock.patch(
                    "fcv.online_study.evaluate_candidate_oracle",
                    side_effect=fake_oracle,
                ),
                mock.patch(
                    "fcv.online_study.prepare_final_test_source",
                    return_value=test_source,
                ),
                mock.patch(
                    "fcv.online_study.evaluate_checkpoint_test_metrics",
                    return_value=(
                        test_metrics,
                        "fingerprint",
                        pd.DataFrame({"sample_id": ["one"]}),
                    ),
                ),
                mock.patch(
                    "fcv.online_study.recompute_test_metrics_from_frame",
                    return_value=test_metrics,
                ),
                mock.patch(
                    "fcv.online_analysis.load_campaign_provenance_receipt",
                    return_value=campaign,
                ),
                mock.patch("fcv.online_analysis.verify_non_test_campaign_inputs"),
                mock.patch(
                    "fcv.online_analysis.candidate_training_fingerprint",
                    return_value="fingerprint",
                ),
                mock.patch(
                    "fcv.online_analysis._validate_unprivileged_evidence"
                ),
                mock.patch("fcv.online_analysis._validate_oracle_evidence"),
            ]
            with ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                produced = train_and_score_online_run(
                    config,
                    run,
                    output_root / "train.csv",
                    output_root / "validation.csv",
                    output_root / "patch_masks.pt",
                    output_root / "oracle.csv",
                    output_root / "test.csv",
                    output_root,
                    device_name="cpu",
                    pretrained_provenance_path=pretrained["artifact_path"],
                )
                self.assertEqual(produced["status"], "complete")
                frozen = freeze_online_validation_selection(config, output_root)
                self.assertEqual(frozen["status"], "complete")

                test_pool = pd.read_csv(
                    output_root
                    / "online_runs"
                    / run.run_id
                    / "test_metrics_analysis_only.csv"
                )
                with mock.patch(
                    "fcv.online_analysis._load_test_pool", return_value=test_pool
                ):
                    analyzed = analyze_online_test_results(config, output_root)
                self.assertEqual(analyzed["status"], "complete")
                self.assertEqual(analyzed["candidate_count"], 1)
                self.assertTrue(
                    (
                        output_root
                        / "selection_results"
                        / "online_gap_closure_summary.json"
                    ).is_file()
                )

    def test_exact_ties_use_candidate_id_ascending(self) -> None:
        rows = [
            _row("candidate_b", "b" * 64, 0.5, 0.5, 0.5),
            _row("candidate_a", "a" * 64, 0.5, 0.5, 0.5),
        ]
        winners = _local_winners(rows)
        self.assertEqual(set(winners.values()), {"candidate_a"})

    def test_retention_never_keeps_more_than_three_unique_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            retained = root / "retained"
            rows = []

            checkpoint_a = root / "a.pt"
            checkpoint_a.write_bytes(b"checkpoint-a")
            rows.append(_row("candidate_a", _sha256(checkpoint_a), 0.5, 0.7, 0.4))
            state = _stage_retention(rows, checkpoint_a, retained)
            self.assertEqual(state["unique_checkpoint_count"], 1)

            checkpoint_b = root / "b.pt"
            checkpoint_b.write_bytes(b"checkpoint-b")
            rows.append(_row("candidate_b", _sha256(checkpoint_b), 0.8, 0.6, 0.9))
            state = _stage_retention(rows, checkpoint_b, retained)
            self.assertEqual(state["unique_checkpoint_count"], 2)

            checkpoint_c = root / "c.pt"
            checkpoint_c.write_bytes(b"checkpoint-c")
            rows.append(_row("candidate_c", _sha256(checkpoint_c), 0.7, 0.95, 0.8))
            state = _stage_retention(rows, checkpoint_c, retained)
            self.assertLessEqual(state["unique_checkpoint_count"], 3)
            self.assertEqual(
                set(state["checkpoints"]), {"candidate_b", "candidate_c"}
            )

            # Stale winners are deleted only after the caller has committed its
            # atomic resume state.
            self.assertTrue((retained / "candidate_a.pt").is_file())
            _prune_retention(state, retained)
            self.assertEqual(
                {path.stem for path in retained.glob("*.pt")},
                {"candidate_b", "candidate_c"},
            )

    def test_partial_plan_commits_recover_at_either_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank_dir = root / "banks"
            bank_dir.mkdir()
            donor_path = root / "donor.pt"
            control_path = root / "control.pt"
            source = mock.Mock()
            donor_plan = {"plan": "donor"}

            # Crash after donor commit but before control commit.
            donor_path.write_bytes(b"committed donor")
            with mock.patch(
                "fcv.online_study.load_background_bank",
                side_effect=[{"label": 0}, {"label": 1}],
            ) as load_bank, mock.patch(
                "fcv.online_study.prepare_opposite_donor_plan",
                return_value=donor_plan,
            ) as prepare_donor, mock.patch(
                "fcv.online_study.prepare_control_plan"
            ) as prepare_control:
                _prepare_online_intervention_plans(
                    {},
                    source,
                    bank_dir,
                    "candidate",
                    "a" * 64,
                    donor_path,
                    control_path,
                )
            self.assertEqual(load_bank.call_count, 2)
            prepare_donor.assert_called_once()
            prepare_control.assert_called_once()
            self.assertFalse(prepare_control.call_args.kwargs["overwrite"])

            # If the donor is absent but a dependent control survives, rebuild
            # the deterministic pair and replace the orphaned control.
            donor_path.unlink()
            control_path.write_bytes(b"orphaned control")
            with mock.patch(
                "fcv.online_study.load_background_bank",
                side_effect=[{"label": 0}, {"label": 1}],
            ), mock.patch(
                "fcv.online_study.prepare_opposite_donor_plan",
                return_value=donor_plan,
            ), mock.patch(
                "fcv.online_study.prepare_control_plan"
            ) as prepare_control:
                _prepare_online_intervention_plans(
                    {},
                    source,
                    bank_dir,
                    "candidate",
                    "a" * 64,
                    donor_path,
                    control_path,
                )
            self.assertTrue(prepare_control.call_args.kwargs["overwrite"])

    def test_analysis_only_test_evaluation_cannot_advance_training_rng(self) -> None:
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        generator = torch.Generator(device="cpu").manual_seed(17)
        before = _rng_state(generator)
        with _isolated_training_rng(generator) as committed:
            random.random()
            np.random.random()
            torch.rand(4)
            torch.rand(4, generator=generator)
        after = _rng_state(generator)

        self.assertEqual(before["python"], committed["python"])
        self.assertEqual(before["python"], after["python"])
        self.assertEqual(before["numpy"][0], after["numpy"][0])
        np.testing.assert_array_equal(before["numpy"][1], after["numpy"][1])
        self.assertEqual(before["numpy"][2:], after["numpy"][2:])
        self.assertTrue(torch.equal(before["torch_cpu"], after["torch_cpu"]))
        self.assertTrue(
            torch.equal(before["train_generator"], after["train_generator"])
        )

    def test_test_index_recovers_one_uncommitted_row_without_resume_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test_metrics_analysis_only.csv"
            run = SweepRun(0, 1.0e-5, 0.01, 0)

            def test_row(epoch: int) -> dict[str, object]:
                row = {column: 0 for column in ONLINE_TEST_COLUMNS}
                row.update(
                    {
                        "run_index": run.run_index,
                        "candidate_id": run.candidate_id(epoch),
                        "epoch": epoch,
                        "seed": run.seed,
                        "learning_rate": run.learning_rate,
                        "weight_decay": run.weight_decay,
                        "checkpoint_sha256": str(epoch) * 64,
                        "per_image_path": f"per-image-{epoch}.csv",
                        "per_image_sha256": "a" * 64,
                        "summary_path": f"summary-{epoch}.json",
                        "summary_sha256": "b" * 64,
                    }
                )
                return row

            _append_test_index_row(path, run, 0, test_row(1))
            # Simulate a crash after writing the next analysis-only row but
            # before committing the optimizer-bearing resume state.
            interrupted = pd.concat(
                [pd.read_csv(path), pd.DataFrame([test_row(2)])], ignore_index=True
            )
            interrupted.to_csv(path, index=False)
            prefix = _test_index_prefix(
                path, run, 1, allow_one_uncommitted_row=True
            )
            self.assertEqual(prefix["candidate_id"].tolist(), [run.candidate_id(1)])
            prefix.to_csv(path, index=False)
            _append_test_index_row(path, run, 1, test_row(2))
            recovered = pd.read_csv(path)
            self.assertEqual(
                recovered["candidate_id"].tolist(),
                [run.candidate_id(1), run.candidate_id(2)],
            )

            with self.assertRaises(OnlineStudyError):
                _test_index_prefix(path, run, 0, allow_one_uncommitted_row=True)

    def test_bounded_repair_invalidates_state_without_adding_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            retained = run_dir / "retained_checkpoints"
            retained.mkdir(parents=True)
            for name in ("a.pt", "b.pt", "c.pt"):
                (retained / name).write_bytes(name.encode())
            for name in (
                "run_summary.json",
                "validation_metrics.csv",
                "test_metrics_analysis_only.csv",
                "retention_state.json",
                "resume_state.pt",
            ):
                (run_dir / name).write_bytes(b"stale")

            _invalidate_completed_run_for_bounded_repair(run_dir, "missing detail")
            self.assertFalse(any(retained.glob("*.pt")))
            receipt = json.loads(
                (run_dir / "bounded_repair_receipt.json").read_text(encoding="utf-8")
            )
            self.assertFalse(receipt["checkpoint_retention_expanded"])
            self.assertEqual(
                receipt["repair_strategy"],
                "rerun_twenty_epochs_with_existing_bounded_retention",
            )

    def test_completed_run_with_missing_detail_enters_bounded_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary).resolve()
            run = SweepRun(0, 1.0e-5, 0.01, 0)
            run_dir = output_root / "online_runs" / run.run_id
            retained = run_dir / "retained_checkpoints"
            retained.mkdir(parents=True)
            (retained / f"{run.candidate_id(1)}.pt").write_bytes(b"winner")

            validation_path = run_dir / "validation_metrics.csv"
            test_path = run_dir / "test_metrics_analysis_only.csv"
            retention_path = run_dir / "retention_state.json"
            summary_path = run_dir / "run_summary.json"
            pd.DataFrame(
                [
                    {
                        "run_index": run.run_index,
                        "candidate_id": run.candidate_id(1),
                        "epoch": 1,
                        "checkpoint_sha256": "c" * 64,
                        "fcv_summary_path": str(run_dir / "missing_fcv.json"),
                        "fcv_summary_sha256": "d" * 64,
                        "controls_summary_path": str(run_dir / "missing_controls.json"),
                        "controls_summary_sha256": "e" * 64,
                        "oracle_summary_path": str(run_dir / "missing_oracle.json"),
                        "oracle_summary_sha256": "f" * 64,
                    }
                ]
            ).to_csv(validation_path, index=False)
            pd.DataFrame(
                [
                    {
                        "run_index": run.run_index,
                        "candidate_id": run.candidate_id(1),
                        "epoch": 1,
                        "checkpoint_sha256": "c" * 64,
                        "per_image_path": str(run_dir / "missing_test.csv"),
                        "per_image_sha256": "1" * 64,
                        "summary_path": str(run_dir / "missing_test.json"),
                        "summary_sha256": "2" * 64,
                    }
                ]
            ).to_csv(test_path, index=False)
            retention_path.write_text("{}\n", encoding="utf-8")
            summary = {
                "artifact_type": "fcv_vit_online_run_summary",
                "status": "complete",
                "run": {
                    "run_index": run.run_index,
                    "learning_rate": run.learning_rate,
                    "weight_decay": run.weight_decay,
                    "seed": run.seed,
                },
                "training_fingerprint": "fingerprint",
                "software_versions": {"package": "version"},
                "software_fingerprint": "software",
                "source_tree_sha256": "source",
                "campaign_provenance_path": "/campaign.json",
                "campaign_provenance_sha256": "a" * 64,
                "campaign_bindings_sha256": "b" * 64,
                "manifest_sha256": {
                    "candidate_train": "c" * 64,
                    "biased_validation": "d" * 64,
                },
                "pretrained_provenance_path": "/pretrained.json",
                "pretrained_provenance_sha256": "e" * 64,
                "pretrained_backbone_sha256": "f" * 64,
                "initial_model_state_sha256": "1" * 64,
                "candidate_count": 1,
                "validation_metrics_sha256": _sha256(validation_path),
                "test_metrics_sha256": _sha256(test_path),
                "retention_state_sha256": _sha256(retention_path),
            }
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            config = {"training": {"epochs": 1}}
            campaign = {
                "artifact_path": "/campaign.json",
                "artifact_sha256": "a" * 64,
                "bindings_sha256": "b" * 64,
                "bindings": {
                    "initialization": {
                        "initial_model_state_sha256_by_seed": {"0": "1" * 64}
                    }
                },
            }
            manifest_hashes = {
                "candidate_train": "c" * 64,
                "biased_validation": "d" * 64,
            }
            pretrained = {
                "artifact_path": "/pretrained.json",
                "artifact_sha256": "e" * 64,
                "pretrained_backbone_sha256": "f" * 64,
            }
            with mock.patch(
                "fcv.online_study.candidate_training_fingerprint",
                return_value="fingerprint",
            ), mock.patch(
                "fcv.online_study.software_versions",
                return_value={"package": "version"},
            ), mock.patch(
                "fcv.online_study.software_fingerprint",
                return_value="software",
            ), mock.patch(
                "fcv.online_study.source_tree_provenance",
                return_value={"source_tree_sha256": "source"},
            ):
                result = _completed_run(
                    config,
                    run,
                    run_dir,
                    output_root,
                    campaign,
                    manifest_hashes,
                    pretrained,
                )
            self.assertIsNone(result)
            self.assertFalse(summary_path.exists())
            self.assertFalse(any(retained.glob("*.pt")))
            self.assertTrue((run_dir / "bounded_repair_receipt.json").is_file())

    def test_post_freeze_cleanup_keeps_only_global_primary_winners(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary).resolve()
            result_dir = output_root / "selection_results"
            result_dir.mkdir(parents=True)
            runs = [
                SweepRun(0, 1.0e-5, 0.01, 0),
                SweepRun(1, 3.0e-5, 0.05, 1),
            ]
            local_winners = [
                {
                    "biased_validation_accuracy": runs[0].candidate_id(1),
                    "equal_weight_original_and_opposite_fcv_accuracy": runs[
                        0
                    ].candidate_id(2),
                    "oracle_validation_balanced_group_accuracy": runs[0].candidate_id(
                        2
                    ),
                },
                {
                    "biased_validation_accuracy": runs[1].candidate_id(1),
                    "equal_weight_original_and_opposite_fcv_accuracy": runs[
                        1
                    ].candidate_id(1),
                    "oracle_validation_balanced_group_accuracy": runs[1].candidate_id(
                        2
                    ),
                },
            ]
            bindings = {}
            for run, selectors in zip(runs, local_winners):
                retained_dir = (
                    output_root / "online_runs" / run.run_id / "retained_checkpoints"
                )
                retained_dir.mkdir(parents=True)
                checkpoints = {}
                for candidate_id in sorted(set(selectors.values())):
                    path = retained_dir / f"{candidate_id}.pt"
                    path.write_bytes(f"weights:{candidate_id}".encode())
                    selected_by = sorted(
                        name for name, selected in selectors.items() if selected == candidate_id
                    )
                    checkpoints[candidate_id] = {
                        "path": str(path.resolve()),
                        "sha256": _sha256(path),
                        "selectors": selected_by,
                    }
                    bindings[candidate_id] = checkpoints[candidate_id]
                retention = {
                    "schema_version": 1,
                    "artifact_type": "fcv_vit_online_local_retention",
                    "status": "complete",
                    "selectors": selectors,
                    "unique_checkpoint_count": len(checkpoints),
                    "checkpoints": checkpoints,
                }
                (retained_dir.parent / "retention_state.json").write_text(
                    json.dumps(retention), encoding="utf-8"
                )

            selected = {
                "biased_validation_accuracy": runs[0].candidate_id(1),
                "equal_weight_original_and_opposite_fcv_accuracy": runs[1].candidate_id(
                    1
                ),
                "oracle_validation_balanced_group_accuracy": runs[1].candidate_id(2),
            }
            rows = []
            for selector, candidate_id in selected.items():
                details = bindings[candidate_id]
                rows.append(
                    {
                        "selector_name": selector,
                        "selected_checkpoint_id": candidate_id,
                        "retained_checkpoint_path": details["path"],
                        "checkpoint_sha256": details["sha256"],
                    }
                )
            selections = pd.DataFrame(rows)
            table_path = result_dir / "selections.csv"
            matrix_path = result_dir / "matrix.csv"
            selections.to_csv(table_path, index=False)
            pd.DataFrame({"candidate_id": sorted(bindings)}).to_csv(
                matrix_path, index=False
            )

            cleanup = _cleanup_non_global_retained_checkpoints(
                output_root,
                result_dir,
                selections,
                runs=runs,
                training_fingerprint="test-fingerprint",
                selection_table_path=table_path,
                candidate_matrix_path=matrix_path,
            )
            self.assertEqual(cleanup["retained_checkpoint_count_before_cleanup"], 4)
            self.assertEqual(cleanup["deleted_checkpoint_count"], 1)
            self.assertEqual(cleanup["retained_checkpoint_count_after_cleanup"], 3)
            for candidate_id, details in bindings.items():
                self.assertEqual(
                    Path(details["path"]).is_file(), candidate_id in set(selected.values())
                )

            # Simulate an interrupted cleanup that left the planned deletion
            # on disk. Re-entry must verify and remove it from the durable
            # plan without requiring all original local winners to reappear.
            deleted_candidate = runs[0].candidate_id(2)
            deleted_path = Path(bindings[deleted_candidate]["path"])
            deleted_path.write_bytes(f"weights:{deleted_candidate}".encode())
            repeated = _cleanup_non_global_retained_checkpoints(
                output_root,
                result_dir,
                selections,
                runs=runs,
                training_fingerprint="test-fingerprint",
                selection_table_path=table_path,
                candidate_matrix_path=matrix_path,
            )
            self.assertEqual(repeated, cleanup)
            self.assertFalse(deleted_path.exists())

            # A fully completed invocation remains idempotent as well.
            completed = _cleanup_non_global_retained_checkpoints(
                output_root,
                result_dir,
                selections,
                runs=runs,
                training_fingerprint="test-fingerprint",
                selection_table_path=table_path,
                candidate_matrix_path=matrix_path,
            )
            self.assertEqual(completed, cleanup)


if __name__ == "__main__":
    unittest.main()
