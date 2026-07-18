"""Online Step-8 FCV scoring and warning-only controls for DecoyMNIST."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from decoy_candidate_training import autocast_context
from decoy_full_config import sha256_file
from decoy_teacher_masks import (
    CATEGORY_BACKGROUND,
    CATEGORY_EVIDENCE,
    exact_source_masks,
    partition_patch_scores,
    pool_patch_occupancy,
)
from decoy_vit_intervention import (
    RECONSTRUCTION_TOLERANCE,
    ViTInterventionError,
    extract_raw_patch_tokens,
    forward_from_raw_patch_tokens,
    mutually_safe_positions,
    replace_spatially_aligned_tokens,
    require_positions_present,
    verify_identity_forward,
)


CONTROL_NAMES = (
    "same_context",
    "random_mask",
    "shuffled_teacher_mask",
    "evidence_swap",
    "exact_synthetic_mask_analysis_only",
)


class FCVScoringError(ValueError):
    """Raised when primary online FCV scoring violates the frozen protocol."""


@dataclass
class OnlineTokenBank:
    """Candidate-specific, memory-only validation features."""

    sample_ids: tuple[str, ...]
    labels: np.ndarray
    raw_patch_tokens: Any
    original_logits: Any
    identity_forward: Mapping[str, Any]

    def index_by_id(self) -> Dict[str, int]:
        return {sample_id: index for index, sample_id in enumerate(self.sample_ids)}

    @property
    def sample_count(self) -> int:
        return len(self.sample_ids)


def harmonic_fcv_score(
    original_accuracy: float,
    counterfactual_accuracy: float,
    *,
    epsilon: float = 1.0e-12,
) -> float:
    a = float(original_accuracy)
    b = float(counterfactual_accuracy)
    if not 0.0 <= a <= 1.0 or not 0.0 <= b <= 1.0:
        raise FCVScoringError("FCV harmonic inputs must be accuracies in [0,1].")
    if float(epsilon) != 1.0e-12:
        raise FCVScoringError("The primary harmonic epsilon is locked to 1e-12.")
    return float((2.0 * a * b) / (a + b + float(epsilon)))


def majority_vote(predictions: Sequence[int], num_classes: int = 10) -> int:
    values = np.asarray(predictions, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise FCVScoringError("Majority vote requires nonempty predictions.")
    if ((values < 0) | (values >= int(num_classes))).any():
        raise FCVScoringError("Majority-vote prediction is outside the class range.")
    return int(np.bincount(values, minlength=int(num_classes)).argmax())


def _stable_rng(seed: int, *parts: str) -> np.random.Generator:
    digest = hashlib.sha256()
    digest.update(str(int(seed)).encode("utf-8"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    derived = int.from_bytes(digest.digest()[:8], "little", signed=False)
    return np.random.default_rng(derived)


def build_online_token_bank(
    model: Any,
    validation_loader: Any,
    device: Any,
    *,
    precision: str,
) -> OnlineTokenBank:
    """Extract one ephemeral raw-token bank; nothing is written to disk."""

    import torch

    model.eval()
    sample_ids: List[str] = []
    labels: List[int] = []
    token_chunks = []
    logit_chunks = []
    identity: Mapping[str, Any] | None = None
    with torch.inference_mode():
        for images, batch_labels, batch_ids in validation_loader:
            images = images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            if identity is None:
                identity = verify_identity_forward(
                    model, images[: min(4, len(images))], tolerance=RECONSTRUCTION_TOLERANCE
                )
            with autocast_context(device, precision):
                raw_tokens = extract_raw_patch_tokens(model, images)
                logits = forward_from_raw_patch_tokens(model, raw_tokens)
            token_chunks.append(raw_tokens.detach().to(device="cpu"))
            logit_chunks.append(logits.detach().float().to(device="cpu"))
            labels.extend(int(value) for value in batch_labels.cpu().tolist())
            sample_ids.extend(str(value) for value in batch_ids)
    if not sample_ids or identity is None:
        raise FCVScoringError("Biased-validation loader produced no examples.")
    if len(set(sample_ids)) != len(sample_ids):
        raise FCVScoringError("Token-bank sample IDs are not unique.")
    bank = OnlineTokenBank(
        sample_ids=tuple(sample_ids),
        labels=np.asarray(labels, dtype=np.int64),
        raw_patch_tokens=torch.cat(token_chunks, dim=0),
        original_logits=torch.cat(logit_chunks, dim=0),
        identity_forward=identity,
    )
    if len(bank.labels) != bank.sample_count:
        raise FCVScoringError("Token-bank labels and IDs are misaligned.")
    return bank


def prepare_exact_control_masks(
    config: Mapping[str, Any], manifest: pd.DataFrame
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Create analysis-only exact digit categories and required decoy cells."""

    data_root = Path(config["paths"]["data_root"]).expanduser().resolve()
    image_size = int(config["model"]["image_size"])
    patch_size = int(config["model"]["patch_size"])
    background_threshold = float(config["fcv"]["background_patch_threshold"])
    evidence_threshold = float(config["fcv"]["evidence_patch_threshold"])
    categories: Dict[str, np.ndarray] = {}
    decoy_positions: Dict[str, np.ndarray] = {}
    for row in manifest.itertuples(index=False):
        sample_id = str(row.sample_id)
        path = (data_root / str(row.image_rel_path)).resolve()
        if not path.is_relative_to(data_root):
            raise FCVScoringError(f"Exact-mask path escapes data root: {row.image_rel_path}")
        if sha256_file(path) != str(row.image_sha256):
            raise FCVScoringError(f"Exact-mask source image changed: {sample_id}.")
        exact = exact_source_masks(path, int(row.label), image_size=image_size)
        digit_scores = pool_patch_occupancy(exact["digit"], patch_size=patch_size)
        categories[sample_id] = partition_patch_scores(
            digit_scores,
            background_threshold=background_threshold,
            evidence_threshold=evidence_threshold,
        )
        decoy_scores = pool_patch_occupancy(exact["decoy"], patch_size=patch_size)
        decoy_positions[sample_id] = np.flatnonzero(decoy_scores > 0.0).astype(np.int64)
    return categories, decoy_positions


def build_control_assignments(
    donor_plan: Mapping[str, Any], *, seed: int, donors_per_target: int
) -> Dict[str, Dict[str, Any]]:
    """Build deterministic in-memory same-context and shuffled-mask assignments."""

    records = list(donor_plan.get("records", []))
    if not records:
        raise FCVScoringError("Donor plan has no eligible targets.")
    by_context: Dict[Tuple[int, str], List[str]] = {}
    for record in records:
        by_context.setdefault(
            (int(record["target_label"]), str(record["corner"])), []
        ).append(str(record["target_sample_id"]))
    for values in by_context.values():
        values.sort()

    ordered_ids = sorted(str(record["target_sample_id"]) for record in records)
    if len(ordered_ids) > 1:
        shuffled_sources = ordered_ids[1:] + ordered_ids[:1]
    else:
        shuffled_sources = ordered_ids[:]
    shuffled_by_target = dict(zip(ordered_ids, shuffled_sources))
    assignments: Dict[str, Dict[str, Any]] = {}
    for record in records:
        target_id = str(record["target_sample_id"])
        pool = [
            sample_id
            for sample_id in by_context[
                (int(record["target_label"]), str(record["corner"]))
            ]
            if sample_id != target_id
        ]
        same_context: List[str] = []
        if len(pool) >= int(donors_per_target):
            rng = _stable_rng(seed, "same_context", target_id)
            chosen = rng.choice(
                np.asarray(pool, dtype=str),
                size=int(donors_per_target),
                replace=False,
            )
            same_context = [str(value) for value in chosen.tolist()]
        assignments[target_id] = {
            "same_context_donor_ids": same_context,
            "shuffled_mask_source_sample_id": shuffled_by_target[target_id],
        }
    return assignments


class _MetricAccumulator:
    def __init__(self, name: str) -> None:
        self.name = name
        self.target_count = 0
        self.draw_count = 0
        self.correct_draws = 0
        self.correct_majorities = 0
        self.true_probability_sum = 0.0
        self.confidence_drop_sum = 0.0
        self.position_count_sum = 0
        self.preserved_error_max = 0.0
        self.donor_error_max = 0.0
        self.changed_token_count = 0
        self.replaced_token_count = 0

    def add(
        self,
        *,
        label: int,
        original_true_probability: float,
        logits: Any,
        position_counts: Sequence[int],
        audits: Sequence[Mapping[str, Any]],
    ) -> None:
        import torch

        if logits.ndim != 2 or len(logits) == 0:
            raise FCVScoringError(f"{self.name} produced invalid logits.")
        probabilities = torch.softmax(logits.float(), dim=1)
        predictions = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
        true_probabilities = (
            probabilities[:, int(label)].detach().cpu().numpy().astype(np.float64)
        )
        draw_count = len(predictions)
        if draw_count != len(position_counts) or draw_count != len(audits):
            raise FCVScoringError(f"{self.name} draw metadata is misaligned.")
        self.target_count += 1
        self.draw_count += draw_count
        self.correct_draws += int((predictions == int(label)).sum())
        self.correct_majorities += int(majority_vote(predictions) == int(label))
        mean_probability = float(true_probabilities.mean())
        self.true_probability_sum += mean_probability
        self.confidence_drop_sum += float(original_true_probability) - mean_probability
        self.position_count_sum += int(sum(int(value) for value in position_counts))
        for audit in audits:
            self.preserved_error_max = max(
                self.preserved_error_max,
                float(audit["preserved_token_max_abs_error"]),
            )
            self.donor_error_max = max(
                self.donor_error_max, float(audit["donor_token_max_abs_error"])
            )
            self.changed_token_count += int(audit["changed_token_count"])
            self.replaced_token_count += int(audit["replaced_patch_count"])

    def summary(self, total_samples: int) -> Dict[str, Any]:
        if self.target_count == 0 or self.draw_count == 0:
            return {
                "status": "no_eligible_targets",
                "eligible_target_count": 0,
                "eligible_target_fraction": 0.0,
                "donor_draw_count": 0,
                "counterfactual_accuracy": None,
                "counterfactual_majority_accuracy": None,
                "mean_true_class_counterfactual_probability": None,
                "mean_original_to_counterfactual_confidence_drop": None,
                "mean_replaced_patch_count": None,
            }
        return {
            "status": "complete",
            "eligible_target_count": self.target_count,
            "eligible_target_fraction": float(self.target_count / int(total_samples)),
            "donor_draw_count": self.draw_count,
            "counterfactual_accuracy": float(self.correct_draws / self.draw_count),
            "counterfactual_majority_accuracy": float(
                self.correct_majorities / self.target_count
            ),
            "mean_true_class_counterfactual_probability": float(
                self.true_probability_sum / self.target_count
            ),
            "mean_original_to_counterfactual_confidence_drop": float(
                self.confidence_drop_sum / self.target_count
            ),
            "mean_replaced_patch_count": float(
                self.position_count_sum / self.draw_count
            ),
            "preserved_token_max_abs_error": self.preserved_error_max,
            "donor_token_max_abs_error": self.donor_error_max,
            "changed_replacement_fraction": float(
                self.changed_token_count / max(self.replaced_token_count, 1)
            ),
        }


def _forward_chunks(
    model: Any,
    token_draws: Sequence[Any],
    device: Any,
    precision: str,
    batch_size: int,
):
    import torch

    if not token_draws:
        raise FCVScoringError("Counterfactual forward received no token draws.")
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(token_draws), int(batch_size)):
            tokens = torch.stack(token_draws[start : start + int(batch_size)]).to(
                device, non_blocking=True
            )
            with autocast_context(device, precision):
                logits = forward_from_raw_patch_tokens(model, tokens)
            chunks.append(logits.detach().float().cpu())
    return torch.cat(chunks, dim=0)


def _original_metrics(bank: OnlineTokenBank) -> Tuple[float, np.ndarray, np.ndarray]:
    import torch

    logits = bank.original_logits.float()
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()
    predictions = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
    accuracy = float((predictions == bank.labels).mean())
    true_probabilities = probabilities[np.arange(bank.sample_count), bank.labels]
    return accuracy, predictions, true_probabilities


def token_bank_classification_metrics(bank: OnlineTokenBank) -> Dict[str, Any]:
    """Reduce original validation logits without another model forward."""

    import torch

    logits = bank.original_logits.float()
    labels = torch.as_tensor(bank.labels, dtype=torch.long)
    loss = float(torch.nn.functional.cross_entropy(logits, labels).item())
    predictions = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
    class_total = np.bincount(bank.labels, minlength=logits.shape[1])
    class_correct = np.bincount(
        bank.labels[predictions == bank.labels], minlength=logits.shape[1]
    )
    if np.any(class_total == 0):
        raise FCVScoringError("Biased validation is missing at least one class.")
    per_class = class_correct.astype(np.float64) / class_total
    return {
        "count": bank.sample_count,
        "loss": loss,
        "accuracy": float(class_correct.sum() / bank.sample_count),
        "balanced_class_accuracy": float(per_class.mean()),
        "worst_class_accuracy": float(per_class.min()),
        "per_class_accuracy": [float(value) for value in per_class],
    }


def score_online_fcv_and_controls(
    config: Mapping[str, Any],
    model: Any,
    token_bank: OnlineTokenBank,
    projected_masks: Mapping[str, np.ndarray],
    donor_plan: Mapping[str, Any],
    exact_categories_by_id: Mapping[str, np.ndarray],
    decoy_positions_by_id: Mapping[str, np.ndarray],
    device: Any,
) -> Dict[str, Any]:
    """Score primary FCV and controls while the candidate remains in memory."""

    model.eval()
    ids = [str(value) for value in projected_masks["sample_ids"].astype(str).tolist()]
    labels = projected_masks["labels"].astype(int)
    categories = projected_masks["patch_categories"].astype(np.uint8)
    eligible = projected_masks["fcv_eligible"].astype(bool)
    if ids != list(token_bank.sample_ids) or not np.array_equal(labels, token_bank.labels):
        raise FCVScoringError("Token bank and projected masks are not row-aligned.")
    if categories.shape[0] != token_bank.sample_count:
        raise FCVScoringError("Projected masks have the wrong sample count.")
    if set(exact_categories_by_id) != set(ids) or set(decoy_positions_by_id) != set(ids):
        raise FCVScoringError("Exact control masks must cover biased validation exactly.")
    eligible_ids = {sample_id for sample_id, value in zip(ids, eligible) if value}
    records = list(donor_plan.get("records", []))
    if {str(record["target_sample_id"]) for record in records} != eligible_ids:
        raise FCVScoringError("Donor-plan targets differ from eligible mask targets.")

    id_to_index = token_bank.index_by_id()
    category_by_id = {sample_id: categories[index] for index, sample_id in enumerate(ids)}
    original_accuracy, _original_predictions, original_true_probabilities = _original_metrics(
        token_bank
    )
    donor_count = int(config["fcv"]["donor_samples_per_target"])
    seed = int(config["fcv"]["donor_plan_seed"])
    assignments = build_control_assignments(
        donor_plan, seed=seed, donors_per_target=donor_count
    )
    accumulators = {"primary": _MetricAccumulator("primary")}
    accumulators.update({name: _MetricAccumulator(name) for name in CONTROL_NAMES})
    warning_count = 0
    warning_reasons: Dict[str, int] = {}

    precision = str(config["training"]["precision"])
    forward_batch_size = int(config["execution"]["fcv_counterfactual_batch_size"])
    target_batch_size = int(config["execution"]["control_target_batch_size"])
    for chunk_start in range(0, len(records), target_batch_size):
        pending: Dict[str, Dict[str, List[Any]]] = {
            name: {"draws": [], "groups": []}
            for name in ("primary",) + CONTROL_NAMES
        }
        for record in records[chunk_start : chunk_start + target_batch_size]:
            target_id = str(record["target_sample_id"])
            target_index = id_to_index[target_id]
            target_label = int(record["target_label"])
            if int(token_bank.labels[target_index]) != target_label:
                raise FCVScoringError(f"Donor-plan target label is stale for {target_id}.")
            target_tokens = token_bank.raw_patch_tokens[target_index].to(device)
            original_probability = float(original_true_probabilities[target_index])
            primary_draws = []
            primary_positions: List[np.ndarray] = []
            primary_audits = []
            opposite_donors = list(record["donors"])
            if len(opposite_donors) != donor_count:
                raise FCVScoringError(f"Wrong primary donor count for {target_id}.")
            for donor in opposite_donors:
                donor_id = str(donor["sample_id"])
                if donor_id not in id_to_index or donor_id not in eligible_ids:
                    raise FCVScoringError(
                        f"Primary donor is outside eligible biased validation: {donor_id}."
                    )
                if int(token_bank.labels[id_to_index[donor_id]]) != int(donor["label"]):
                    raise FCVScoringError(f"Donor-plan label is stale for {donor_id}.")
                if int(donor["label"]) == target_label:
                    raise FCVScoringError(f"Primary donor is not class-conflicting: {donor_id}.")
                donor_tokens = token_bank.raw_patch_tokens[id_to_index[donor_id]].to(
                    device
                )
                positions = mutually_safe_positions(
                    category_by_id[target_id], category_by_id[donor_id]
                )
                require_positions_present(positions, decoy_positions_by_id[target_id])
                replaced, audit = replace_spatially_aligned_tokens(
                    target_tokens, donor_tokens, positions
                )
                primary_draws.append(replaced)
                primary_positions.append(positions)
                primary_audits.append(audit.as_dict())
            pending["primary"]["draws"].extend(primary_draws)
            pending["primary"]["groups"].append(
                {
                    "label": target_label,
                    "original_probability": original_probability,
                    "draw_count": len(primary_draws),
                    "position_counts": [len(value) for value in primary_positions],
                    "audits": primary_audits,
                }
            )

            control_specs: Dict[str, List[Tuple[str, np.ndarray]]] = {
                "same_context": [],
                "random_mask": [],
                "shuffled_teacher_mask": [],
                "evidence_swap": [],
                "exact_synthetic_mask_analysis_only": [],
            }
            for donor_id in assignments[target_id]["same_context_donor_ids"]:
                positions = mutually_safe_positions(
                    category_by_id[target_id], category_by_id[donor_id]
                )
                control_specs["same_context"].append((donor_id, positions))
            shuffled_source = str(
                assignments[target_id]["shuffled_mask_source_sample_id"]
            )
            for draw_index, donor in enumerate(opposite_donors):
                donor_id = str(donor["sample_id"])
                matched_count = len(primary_positions[draw_index])
                random_rng = _stable_rng(seed, "random_mask", target_id, donor_id)
                random_positions = np.sort(
                    random_rng.choice(
                        np.arange(categories.shape[1], dtype=np.int64),
                        size=matched_count,
                        replace=False,
                    )
                )
                control_specs["random_mask"].append((donor_id, random_positions))
                shuffled_positions = mutually_safe_positions(
                    category_by_id[shuffled_source], category_by_id[donor_id]
                )
                control_specs["shuffled_teacher_mask"].append(
                    (donor_id, shuffled_positions)
                )
                evidence_positions = mutually_safe_positions(
                    category_by_id[target_id],
                    category_by_id[donor_id],
                    category=CATEGORY_EVIDENCE,
                )
                control_specs["evidence_swap"].append((donor_id, evidence_positions))
                exact_positions = mutually_safe_positions(
                    exact_categories_by_id[target_id], exact_categories_by_id[donor_id]
                )
                control_specs["exact_synthetic_mask_analysis_only"].append(
                    (donor_id, exact_positions)
                )

            for control_name, specifications in control_specs.items():
                try:
                    if len(specifications) != donor_count:
                        raise ViTInterventionError(
                            f"requires {donor_count} valid donor draws"
                        )
                    draws = []
                    audits = []
                    counts = []
                    for donor_id, positions in specifications:
                        if len(positions) == 0:
                            raise ViTInterventionError("has an empty replacement set")
                        donor_tokens = token_bank.raw_patch_tokens[
                            id_to_index[donor_id]
                        ].to(device)
                        replaced, audit = replace_spatially_aligned_tokens(
                            target_tokens, donor_tokens, positions
                        )
                        draws.append(replaced)
                        audits.append(audit.as_dict())
                        counts.append(len(positions))
                    pending[control_name]["draws"].extend(draws)
                    pending[control_name]["groups"].append(
                        {
                            "label": target_label,
                            "original_probability": original_probability,
                            "draw_count": len(draws),
                            "position_counts": counts,
                            "audits": audits,
                        }
                    )
                except (ViTInterventionError, FCVScoringError) as exc:
                    warning_count += 1
                    reason = f"{control_name}:{exc}"
                    warning_reasons[reason] = warning_reasons.get(reason, 0) + 1

        for name, values in pending.items():
            if not values["draws"]:
                continue
            logits = _forward_chunks(
                model,
                values["draws"],
                device,
                precision,
                forward_batch_size,
            )
            offset = 0
            for group in values["groups"]:
                draw_count = int(group["draw_count"])
                accumulators[name].add(
                    label=int(group["label"]),
                    original_true_probability=float(group["original_probability"]),
                    logits=logits[offset : offset + draw_count],
                    position_counts=group["position_counts"],
                    audits=group["audits"],
                )
                offset += draw_count
            if offset != len(logits):
                raise FCVScoringError(f"{name} batched logits are misaligned.")

    primary = accumulators["primary"].summary(token_bank.sample_count)
    if primary["status"] != "complete":
        raise FCVScoringError("Primary FCV produced no eligible targets.")
    if primary["donor_draw_count"] != donor_count * len(records):
        raise FCVScoringError("Primary donor-expanded denominator is incomplete.")
    if float(primary["changed_replacement_fraction"]) <= 0.0:
        raise FCVScoringError("Primary FCV intervention was a complete token-level no-op.")
    selector = harmonic_fcv_score(
        original_accuracy,
        float(primary["counterfactual_accuracy"]),
        epsilon=float(config["fcv"]["primary_selector"]["epsilon"]),
    )
    return {
        "artifact_type": "fcv_vit_decoymnist_online_fcv_aggregate",
        "artifact_version": 1,
        "sample_count": token_bank.sample_count,
        "original_biased_validation_accuracy": original_accuracy,
        "primary_fcv": primary,
        "harmonic_fcv_score": selector,
        "controls": {
            name: accumulators[name].summary(token_bank.sample_count)
            for name in CONTROL_NAMES
        },
        "control_diagnostics_warning_only": True,
        "control_warning_count": warning_count,
        "control_warning_reason_counts": warning_reasons,
        "identity_forward": dict(token_bank.identity_forward),
        "persistent_per_target_rows": False,
        "persistent_per_donor_rows": False,
    }
