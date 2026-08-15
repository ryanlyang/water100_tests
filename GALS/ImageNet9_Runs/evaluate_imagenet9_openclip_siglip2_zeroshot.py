#!/usr/bin/env python3
"""Evaluate frozen OpenCLIP LAION and SigLIP2 models on ImageNet-9 tests."""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from evaluate_imagenet9_clip_vit_zeroshot import (
    PROMPT_CONCEPTS,
    PROMPT_TEMPLATES,
    _atomic_csv,
    _atomic_json,
    _build_text_classifier,
    _evaluate_variant,
    _load_open_clip,
    _seed_everything,
    _sha256,
    summarize_model,
)


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    model_name: str
    pretrained: str


MODEL_SPECS: Mapping[str, ModelSpec] = {
    "openclip_laion": ModelSpec(
        model_id="openclip_laion",
        display_name="OpenCLIP LAION ViT-B/32",
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
    ),
    "siglip2": ModelSpec(
        model_id="siglip2",
        display_name="SigLIP2 ViT-B/16-256",
        model_name="ViT-B-16-SigLIP2-256",
        pretrained="webli",
    ),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/ryreu/guided_cnn/data/imagenet9"),
    )
    parser.add_argument("--protocol-name", default="reconstructed_original_bbox1_v1")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=list(MODEL_SPECS),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--download-root", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def build_contract(args: argparse.Namespace) -> Dict[str, object]:
    from imagenet9_data import CLASS_NAMES, FORBIDDEN_SELECTION_VARIANTS

    metadata_root = args.data_root / "metadata" / args.protocol_name
    official_manifest = metadata_root / "official_test_manifest.csv"
    if not official_manifest.is_file():
        raise FileNotFoundError(official_manifest)
    if tuple(PROMPT_CONCEPTS) != tuple(CLASS_NAMES):
        raise RuntimeError("Prompt concepts do not match the ImageNet-9 class order")

    selected_specs = [MODEL_SPECS[model_id] for model_id in args.models]
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "imagenet9_backgrounds_challenge",
        "evaluation_split": "official_test_only",
        "official_manifest": str(official_manifest.resolve()),
        "official_manifest_sha256": _sha256(official_manifest),
        "official_test_root": str(
            (args.data_root / "official_test" / "bg_challenge").resolve()
        ),
        "models": [asdict(spec) for spec in selected_specs],
        "implementation": "open_clip",
        "class_names": list(CLASS_NAMES),
        "prompt_concepts": dict(PROMPT_CONCEPTS),
        "prompt_templates": list(PROMPT_TEMPLATES),
        "text_ensemble": "normalize_each_prompt_then_mean_then_normalize",
        "variants": list(FORBIDDEN_SELECTION_VARIANTS),
        "validation_or_tuning_data_used": False,
        "prompt_selection_on_official_test": False,
        "seed": args.seed,
    }


def _create_model(open_clip, spec: ModelSpec, device: str, cache_dir: Optional[Path]):
    kwargs = {
        "model_name": spec.model_name,
        "pretrained": spec.pretrained,
        "device": device,
    }
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        parameters = inspect.signature(open_clip.create_model_and_transforms).parameters
        if "cache_dir" in parameters:
            kwargs["cache_dir"] = str(cache_dir)
        else:
            print("[INFO] open_clip does not expose cache_dir; using default cache", flush=True)

    available = set(open_clip.list_pretrained())
    requested = (spec.model_name, spec.pretrained)
    if requested not in available:
        raise RuntimeError(
            f"The installed open_clip does not provide {requested}. "
            "Use the r4rr-weclip environment or update open_clip_torch."
        )

    model, _train_preprocess, eval_preprocess = open_clip.create_model_and_transforms(
        **kwargs
    )
    tokenizer = open_clip.get_tokenizer(spec.model_name)
    return model, eval_preprocess, tokenizer


def _write_tables(
    output_root: Path,
    class_names: Sequence[str],
    model_results: Mapping[str, Mapping[str, object]],
) -> None:
    variant_rows = []
    class_rows = []
    summary_rows = []
    for model_id, payload in model_results.items():
        spec = MODEL_SPECS[model_id]
        variants = payload["variant_results"]
        for variant, metrics in variants.items():
            variant_rows.append(
                {
                    "model_id": model_id,
                    "model": spec.display_name,
                    "architecture": spec.model_name,
                    "pretrained": spec.pretrained,
                    "variant": variant,
                    "samples": metrics["samples"],
                    "accuracy_pct": 100.0 * float(metrics["accuracy"]),
                    "macro_class_accuracy_pct": 100.0
                    * float(metrics["macro_class_accuracy"]),
                    "seconds": metrics["seconds"],
                }
            )
            for index, class_name in enumerate(class_names):
                class_rows.append(
                    {
                        "model_id": model_id,
                        "model": spec.display_name,
                        "architecture": spec.model_name,
                        "pretrained": spec.pretrained,
                        "variant": variant,
                        "class_index": index,
                        "class_name": class_name,
                        "support": metrics["class_support"][index],
                        "accuracy_pct": 100.0
                        * float(metrics["per_class_accuracy"][index]),
                    }
                )

        summary = payload["robustness_summary"]
        summary_rows.append(
            {
                "model_id": model_id,
                "model": spec.display_name,
                "architecture": spec.model_name,
                "pretrained": spec.pretrained,
                "original_macro_pct": 100.0
                * float(summary["original_macro_class_accuracy"]),
                "mixed_average_macro_pct": 100.0
                * float(summary["mixed_average_macro_class_accuracy"]),
                "original_minus_mixed_rand_points": 100.0
                * float(summary["original_minus_mixed_rand"]),
                "only_fg_macro_pct": 100.0
                * float(summary["only_fg_macro_class_accuracy"]),
                "only_bg_average_macro_pct": 100.0
                * float(summary["only_bg_average_macro_class_accuracy"]),
                "no_fg_macro_pct": 100.0
                * float(summary["no_fg_macro_class_accuracy"]),
                "worst_variant": summary["worst_variant"],
                "worst_variant_macro_pct": 100.0
                * float(summary["worst_variant_macro_class_accuracy"]),
            }
        )

    _atomic_csv(
        output_root / "variant_results.csv",
        (
            "model_id",
            "model",
            "architecture",
            "pretrained",
            "variant",
            "samples",
            "accuracy_pct",
            "macro_class_accuracy_pct",
            "seconds",
        ),
        variant_rows,
    )
    _atomic_csv(
        output_root / "per_class_results.csv",
        (
            "model_id",
            "model",
            "architecture",
            "pretrained",
            "variant",
            "class_index",
            "class_name",
            "support",
            "accuracy_pct",
        ),
        class_rows,
    )
    _atomic_csv(
        output_root / "robustness_summary.csv",
        (
            "model_id",
            "model",
            "architecture",
            "pretrained",
            "original_macro_pct",
            "mixed_average_macro_pct",
            "original_minus_mixed_rand_points",
            "only_fg_macro_pct",
            "only_bg_average_macro_pct",
            "no_fg_macro_pct",
            "worst_variant",
            "worst_variant_macro_pct",
        ),
        summary_rows,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

    _seed_everything(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    contract = build_contract(args)
    _atomic_json(args.output_root / "evaluation_contract.json", contract)

    from imagenet9_data import (
        CLASS_NAMES,
        FORBIDDEN_SELECTION_VARIANTS,
        load_official_variant_samples,
    )

    official_manifest = Path(str(contract["official_manifest"]))
    official_root = Path(str(contract["official_test_root"]))
    variant_samples = {
        variant: load_official_variant_samples(
            official_manifest, official_root, variant, verify_files=True
        )
        for variant in FORBIDDEN_SELECTION_VARIANTS
    }
    open_clip = _load_open_clip()
    model_results: Dict[str, Dict[str, object]] = {}

    for model_id in args.models:
        spec = MODEL_SPECS[model_id]
        model_path = args.output_root / "models" / f"{model_id}.json"
        if model_path.is_file() and not args.force:
            stored = json.loads(model_path.read_text())
            if stored.get("contract") != contract:
                raise RuntimeError(
                    f"Refusing to resume {model_id} with a changed contract: {model_path}"
                )
            if set(stored.get("variant_results", {})) == set(
                FORBIDDEN_SELECTION_VARIANTS
            ):
                model_results[model_id] = stored
                print(f"[SKIP] Complete result already exists for {model_id}", flush=True)
                continue

        print(
            f"[MODEL] id={model_id} model={spec.model_name} "
            f"pretrained={spec.pretrained} device={args.device}",
            flush=True,
        )
        model, preprocess, tokenizer = _create_model(
            open_clip, spec, args.device, args.download_root
        )
        model.eval()
        text_classifier = _build_text_classifier(
            model, tokenizer, CLASS_NAMES, args.device
        )
        variant_results: Dict[str, Dict[str, object]] = {}
        for variant in FORBIDDEN_SELECTION_VARIANTS:
            metrics = _evaluate_variant(
                model,
                preprocess,
                text_classifier,
                variant_samples[variant],
                args.batch_size,
                args.num_workers,
                args.device,
            )
            variant_results[variant] = metrics
            print(
                f"[RESULT] model={model_id} variant={variant} n={metrics['samples']} "
                f"acc={100.0 * float(metrics['accuracy']):.2f} "
                f"macro={100.0 * float(metrics['macro_class_accuracy']):.2f}",
                flush=True,
            )

        payload = {
            "contract": contract,
            "model": asdict(spec),
            "variant_results": variant_results,
            "robustness_summary": summarize_model(variant_results),
        }
        _atomic_json(model_path, payload)
        model_results[model_id] = payload

        del model, text_classifier
        import torch

        torch.cuda.empty_cache()

    _write_tables(args.output_root, CLASS_NAMES, model_results)
    _atomic_json(
        args.output_root / "results.json",
        {"contract": contract, "models": model_results},
    )
    print(f"[DONE] {args.output_root / 'variant_results.csv'}", flush=True)
    print(f"[DONE] {args.output_root / 'per_class_results.csv'}", flush=True)
    print(f"[DONE] {args.output_root / 'robustness_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
