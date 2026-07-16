#!/usr/bin/env python3
"""Create a uniform-weight checkpoint soup from matching RedMeat .pth files."""

from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import torch


def _extract_state_dict(obj) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if all(torch.is_tensor(v) for v in obj.values()):
            return obj
    raise TypeError("Checkpoint is not a raw state_dict and has no 'state_dict' entry.")


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    sd = _extract_state_dict(obj)
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu()
    return out


def _common_compatible_keys(states: List[Dict[str, torch.Tensor]]) -> Tuple[List[str], List[str]]:
    common = set(states[0].keys())
    for sd in states[1:]:
        common.intersection_update(sd.keys())

    good: List[str] = []
    bad: List[str] = []
    for k in sorted(common):
        ref = states[0][k]
        ok = True
        for sd in states[1:]:
            t = sd[k]
            if (t.shape != ref.shape) or (t.dtype != ref.dtype):
                ok = False
                break
        if ok:
            good.append(k)
        else:
            bad.append(k)
    return good, bad


def _average_tensors(tensors: Iterable[torch.Tensor], out_dtype: torch.dtype) -> torch.Tensor:
    ts = list(tensors)
    if not ts:
        raise ValueError("No tensors to average.")
    acc = ts[0].to(torch.float64).clone()
    for t in ts[1:]:
        acc.add_(t.to(torch.float64))
    acc.div_(float(len(ts)))
    return acc.to(out_dtype)


def build_soup(checkpoint_paths: List[str]) -> Tuple[Dict[str, torch.Tensor], List[str], List[str]]:
    states = [_load_state_dict(p) for p in checkpoint_paths]
    good_keys, bad_keys = _common_compatible_keys(states)

    soup: Dict[str, torch.Tensor] = {}
    skipped_non_float: List[str] = []
    for k in good_keys:
        ref = states[0][k]
        if ref.is_floating_point():
            soup[k] = _average_tensors((sd[k] for sd in states), out_dtype=ref.dtype)
        else:
            soup[k] = ref.clone()
            skipped_non_float.append(k)
    return soup, bad_keys, skipped_non_float


def main() -> None:
    p = argparse.ArgumentParser(description="Average matching RedMeat guided checkpoints into one soup checkpoint.")
    p.add_argument(
        "--checkpoint-dir",
        default="RedMeat_Guided_Checkpoints",
        help="Directory containing .pth checkpoints.",
    )
    p.add_argument(
        "--pattern",
        default="resnet50_redmeat_final_kl11_attn2_*.pth",
        help="Glob pattern inside --checkpoint-dir for files to average.",
    )
    p.add_argument(
        "--output",
        default="",
        help="Output .pth path. If omitted, auto-named in --checkpoint-dir.",
    )
    p.add_argument("--min-checkpoints", type=int, default=2, help="Require at least this many matches.")
    args = p.parse_args()

    ckpt_dir = os.path.abspath(args.checkpoint_dir)
    pat = os.path.join(ckpt_dir, args.pattern)
    paths = sorted(glob.glob(pat))

    print(f"[SOUP] checkpoint_dir={ckpt_dir}")
    print(f"[SOUP] pattern={args.pattern}")
    print(f"[SOUP] matched={len(paths)}")
    for pth in paths:
        print(f"  - {pth}")

    if len(paths) < int(args.min_checkpoints):
        raise RuntimeError(
            f"Need at least {args.min_checkpoints} checkpoints, found {len(paths)} for pattern: {pat}"
        )

    soup_sd, bad_keys, skipped_non_float = build_soup(paths)

    if args.output:
        out_path = os.path.abspath(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"resnet50_redmeat_soup_kl11_attn2_n{len(paths)}_{ts}.pth"
        out_path = os.path.join(ckpt_dir, out_name)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(soup_sd, out_path)

    print(f"[SOUP] saved={out_path}")
    print(f"[SOUP] num_keys_saved={len(soup_sd)}")
    if bad_keys:
        print(f"[SOUP] skipped_incompatible_keys={len(bad_keys)}")
    if skipped_non_float:
        print(f"[SOUP] non_float_keys_copied_from_first={len(skipped_non_float)}")


if __name__ == "__main__":
    main()

