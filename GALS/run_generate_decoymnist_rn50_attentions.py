#!/usr/bin/env python3
"""Generate GALS-style CLIP RN50 Grad-CAM attention maps for DecoyMNIST.

This writes one .pth file per image, matching the structure and payload style
used by GALS extract_attention.py:
  - unnormalized_attentions
  - attentions
  - text_list
  - probs

Output layout mirrors DecoyMNIST PNG layout:
  <output-root>/<split>/<class>/<filename>.pth
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm


DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _strip_center_crop(preprocess: transforms.Compose) -> transforms.Compose:
    """Match GALS extract_attention behavior: resize to 224x224, no center crop."""
    out: List[transforms.Compose] = []
    for t in preprocess.transforms:
        if isinstance(t, torchvision.transforms.Resize):
            out.append(transforms.Resize((224, 224), interpolation=Image.BICUBIC))
        elif not isinstance(t, torchvision.transforms.CenterCrop):
            out.append(t)
    return transforms.Compose(out)


def _parse_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _class_prompts(class_name: str, templates: Sequence[str]) -> List[str]:
    token = DIGIT_WORDS.get(class_name, class_name.replace("_", " "))
    return [t.format(token) for t in templates]


def _generic_prompts(prompts_text: str) -> List[str]:
    prompts = _parse_csv(prompts_text)
    if not prompts:
        raise ValueError("Generic prompts list is empty.")
    return prompts


def _resolve_default_png_root() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "MakeMNIST" / "data" / "DecoyMNIST_png",
        here / "data" / "DecoyMNIST_png",
        Path.cwd() / "data" / "DecoyMNIST_png",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return candidates[0].resolve()


def _collect_samples(
    png_root: Path,
    splits: Sequence[str],
) -> List[Tuple[str, str, int, str]]:
    """Return list of (split, abs_path, class_idx, class_name)."""
    out: List[Tuple[str, str, int, str]] = []
    for split in splits:
        split_dir = png_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        ds = ImageFolder(str(split_dir))
        for abs_path, class_idx in ds.samples:
            class_name = ds.classes[class_idx]
            out.append((split, abs_path, class_idx, class_name))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CLIP RN50 Grad-CAM attentions for DecoyMNIST"
    )
    parser.add_argument("--png-root", type=str, default="")
    parser.add_argument("--output-root", type=str, default="")
    parser.add_argument("--splits", type=str, default="train,test")
    parser.add_argument("--clip-model", type=str, default="RN50")
    parser.add_argument("--target-layer", type=str, default="layer4.2.relu")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--disable-vis", dest="disable_vis", action="store_true")
    parser.add_argument("--enable-vis", dest="disable_vis", action="store_false")
    parser.add_argument("--max-vis", type=int, default=30)
    parser.add_argument("--vis-every", type=int, default=50)
    parser.add_argument("--use-prompts-per-class", action="store_true", default=True)
    parser.add_argument(
        "--templates",
        type=str,
        default="a handwritten digit {},a photo of the handwritten digit {},the number {}",
        help="Comma-separated templates for per-class prompts. Must include {}.",
    )
    parser.add_argument(
        "--generic-prompts",
        type=str,
        default="an image of a handwritten digit,a photo of a handwritten digit",
        help="Comma-separated prompts used only when --no-use-prompts-per-class is set.",
    )
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--empty-cache-every", type=int, default=25)
    parser.add_argument("--save-vis-dir", type=str, default="")
    parser.add_argument("--no-use-prompts-per-class", dest="use_prompts_per_class", action="store_false")
    parser.set_defaults(skip_existing=True, disable_vis=True)
    args = parser.parse_args()

    _set_seed(args.seed)
    device = args.device

    from CLIP.clip import clip
    from utils import attention_utils as au

    png_root = Path(args.png_root).expanduser().resolve() if args.png_root else _resolve_default_png_root()
    if not png_root.is_dir():
        raise FileNotFoundError(
            f"Could not find DecoyMNIST PNG root: {png_root}. "
            "Pass --png-root explicitly."
        )

    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else (png_root / "clip_rn50_attention_gradcam").resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    vis_dir = (
        Path(args.save_vis_dir).expanduser().resolve()
        if args.save_vis_dir
        else (output_root / "vis").resolve()
    )
    vis_dir.mkdir(parents=True, exist_ok=True)

    splits = _parse_csv(args.splits)
    if not splits:
        raise ValueError("--splits cannot be empty.")

    all_samples = _collect_samples(png_root, splits)
    n_total = len(all_samples)
    start_idx = max(0, int(args.start_idx))
    end_idx = n_total if int(args.end_idx) < 0 else min(int(args.end_idx), n_total)
    if start_idx > end_idx:
        raise ValueError(f"Invalid range: start_idx={start_idx}, end_idx={end_idx}, n_total={n_total}")

    model, preprocess = clip.load(args.clip_model, device=device, jit=False)
    preprocess = _strip_center_crop(preprocess)

    tokenized_cache: Dict[str, torch.Tensor] = {}
    prompts_cache: Dict[str, List[str]] = {}
    templates = _parse_csv(args.templates)
    if args.use_prompts_per_class and not templates:
        raise ValueError("--templates is empty but per-class prompts are enabled.")
    generic_prompts = _generic_prompts(args.generic_prompts)
    generic_tokens = clip.tokenize(generic_prompts).to(device)

    print(f"[INFO] png_root={png_root}")
    print(f"[INFO] output_root={output_root}")
    print(f"[INFO] splits={splits}")
    print(f"[INFO] clip_model={args.clip_model}")
    print(f"[INFO] target_layer={args.target_layer}")
    print(f"[INFO] range=[{start_idx}, {end_idx}) of {n_total}")
    print(f"[INFO] use_prompts_per_class={args.use_prompts_per_class}")
    print(f"[INFO] skip_existing={args.skip_existing and not args.overwrite}")

    n_saved = 0
    n_skipped = 0
    n_vis = 0
    pbar = tqdm(range(start_idx, end_idx), total=(end_idx - start_idx))

    for i in pbar:
        split, abs_path, _, class_name = all_samples[i]
        abs_path_p = Path(abs_path)
        rel = abs_path_p.resolve().relative_to(png_root)
        save_path = (output_root / rel).with_suffix(".pth")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        do_skip = (args.skip_existing and not args.overwrite and save_path.exists())
        if do_skip:
            n_skipped += 1
            continue

        if args.use_prompts_per_class:
            if class_name not in tokenized_cache:
                prompts = _class_prompts(class_name, templates)
                tokenized_cache[class_name] = clip.tokenize(prompts).to(device)
                prompts_cache[class_name] = prompts
            text_list = prompts_cache[class_name]
            tokenized_text = tokenized_cache[class_name]
        else:
            text_list = generic_prompts
            tokenized_text = generic_tokens

        plot_vis = (not args.disable_vis) and (i % max(1, args.vis_every) == 0) and (n_vis < args.max_vis)
        vis_path = str(vis_dir / f"{split}_{class_name}_{abs_path_p.stem}.png")
        if plot_vis:
            n_vis += 1

        attention = au.clip_gcam(
            model=model,
            preprocess=preprocess,
            file_path=str(abs_path_p),
            text_list=text_list,
            tokenized_text=tokenized_text,
            layer=args.target_layer,
            device=device,
            plot_vis=plot_vis,
            save_vis_path=vis_path,
            resize=False,
        )
        torch.save(attention, str(save_path))
        n_saved += 1

        del attention
        if device.startswith("cuda") and args.empty_cache_every > 0 and (i + 1) % args.empty_cache_every == 0:
            torch.cuda.empty_cache()

    print("[DONE] DecoyMNIST RN50 attention generation complete.")
    print(f"[DONE] saved={n_saved} skipped_existing={n_skipped} vis_saved={n_vis}")
    print(f"[DONE] output_root={output_root}")


if __name__ == "__main__":
    main()
