#!/usr/bin/env python3
"""Evaluate DecoyMNIST checkpoints on test split and report worst-class accuracy."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Grayscale, Lambda, ToTensor


class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        return x

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.forward_features(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.forward_logits(x), dim=1)


class ABNLeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.att_conv = nn.Conv2d(50, 1, kernel_size=1, stride=1)
        self.abn_fc = nn.Linear(50, 10)
        self.fc1 = nn.Linear(4 * 4 * 50, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        feat = F.relu(self.conv2(x))
        att = torch.sigmoid(self.att_conv(feat))
        feat_att = feat * (1.0 + att)
        x_main = F.max_pool2d(feat_att, 2, 2)
        x_main = x_main.view(-1, 4 * 4 * 50)
        x_main = F.relu(self.fc1(x_main))
        return self.fc2(x_main)

    def forward(self, x: torch.Tensor):
        logits_main = self.forward_logits(x)
        feat_aux = torch.zeros((x.size(0), 10), device=x.device, dtype=logits_main.dtype)
        return F.log_softmax(logits_main, dim=1), feat_aux


@dataclass
class ModelSpec:
    name: str
    model: nn.Module
    ckpt: Path
    seed: Optional[int]


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(payload) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
        if payload and all(isinstance(k, str) for k in payload.keys()):
            return payload
    raise RuntimeError("Could not extract state_dict from checkpoint payload")


def candidate_state_dicts(state: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
    def strip_prefix(d: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in d.items():
            out[k[len(prefix) :]] = v if k.startswith(prefix) else v
            if not k.startswith(prefix):
                out[k] = v
        return out

    def add_prefix(d: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        return {prefix + k: v for k, v in d.items()}

    cands = [state]
    for p in ("module.", "model.", "net.", "base."):
        cands.append(strip_prefix(state, p))
    cands.append(add_prefix(state, "base."))
    cands.append(add_prefix(state, "module."))

    uniq = []
    seen = set()
    for d in cands:
        sig = tuple(sorted(d.keys()))
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(d)
    return uniq


def load_checkpoint_flex(model: nn.Module, ckpt_path: Path) -> None:
    payload = _torch_load(ckpt_path)
    raw = extract_state_dict(payload)
    for cand in candidate_state_dicts(raw):
        try:
            missing, unexpected = model.load_state_dict(cand, strict=False)
        except Exception:
            continue
        loaded = len(set(model.state_dict().keys()) & set(cand.keys()))
        if loaded > 0 and len(unexpected) < max(5, int(0.1 * max(1, len(cand)))):
            return
    raise RuntimeError(f"Failed to load checkpoint into model: {ckpt_path}")


def model_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_logits"):
        return model.forward_logits(x)
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def _extract_seed_from_path(path: Path) -> Optional[int]:
    m = re.search(r"seed[_-]?(\d+)", path.name)
    if m:
        return int(m.group(1))
    m = re.search(r"seed[_-]?(\d+)", str(path.parent))
    if m:
        return int(m.group(1))
    return None


def _parse_ckpt_list(text: str) -> List[Path]:
    items = [x.strip() for x in str(text).split(",") if x.strip()]
    return [Path(x).expanduser().resolve() for x in items]


def _class_name(classes: List[str], idx: int) -> str:
    if 0 <= idx < len(classes):
        return str(classes[idx])
    return str(idx)


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int = 10):
    correct = np.zeros(num_classes, dtype=np.int64)
    total = np.zeros(num_classes, dtype=np.int64)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model_logits(model, x)
            pred = logits.argmax(dim=1)
            y_np = y.detach().cpu().numpy().astype(np.int64, copy=False)
            pred_np = pred.detach().cpu().numpy().astype(np.int64, copy=False)
            np.add.at(confusion, (y_np, pred_np), 1)

            for c in range(num_classes):
                mask = y.eq(c)
                n = int(mask.sum().item())
                if n > 0:
                    total[c] += n
                    correct[c] += int(pred[mask].eq(y[mask]).sum().item())

    class_acc = []
    for c in range(num_classes):
        if total[c] > 0:
            class_acc.append(100.0 * float(correct[c]) / float(total[c]))
        else:
            class_acc.append(float("nan"))

    finite = [v for v in class_acc if np.isfinite(v)]
    worst = float(np.min(finite)) if finite else float("nan")
    mean_cls = float(np.mean(finite)) if finite else float("nan")
    overall = 100.0 * float(correct.sum()) / float(max(1, total.sum()))
    worst_idx = int(np.nanargmin(np.asarray(class_acc, dtype=np.float64)))
    worst_total = int(total[worst_idx])
    worst_correct = int(correct[worst_idx])
    worst_errors = int(max(0, worst_total - worst_correct))

    mis = confusion[worst_idx].copy()
    if 0 <= worst_idx < mis.shape[0]:
        mis[worst_idx] = 0
    if int(mis.sum()) > 0:
        top_mis_idx = int(np.argmax(mis))
        top_mis_count = int(mis[top_mis_idx])
        top_mis_pct_of_errors = 100.0 * float(top_mis_count) / float(max(1, mis.sum()))
        top_mis_pct_of_group = 100.0 * float(top_mis_count) / float(max(1, worst_total))
    else:
        top_mis_idx = -1
        top_mis_count = 0
        top_mis_pct_of_errors = 0.0
        top_mis_pct_of_group = 0.0

    return {
        "overall": overall,
        "mean_cls": mean_cls,
        "worst_acc": worst,
        "class_acc": class_acc,
        "worst_idx": worst_idx,
        "worst_total": worst_total,
        "worst_correct": worst_correct,
        "worst_errors": worst_errors,
        "top_mis_idx": top_mis_idx,
        "top_mis_count": top_mis_count,
        "top_mis_pct_of_errors": top_mis_pct_of_errors,
        "top_mis_pct_of_group": top_mis_pct_of_group,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate 6 DecoyMNIST models on test split.")
    p.add_argument("--png-root", default="/workspace/Waterbird_Runs/MakeMNIST/data/DecoyMNIST_png")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-csv", default="")

    p.add_argument("--guided-ckpt", default="/workspace/logsMNIST/decoy_fixed_guided_19ep_one_20260303_065125_artifacts/checkpoints/trial_0_bestVal.ckpt")
    p.add_argument("--vanilla-ckpt", default="/workspace/logsMNIST/decoy_vanilla_seed0_ckpts/decoymnist_rrr_seed0_bestval_99.60_epoch18_20260303_063411.pth")
    p.add_argument("--gals-ckpt", default="/workspace/Waterbird_Runs/GALS/DecoyMNIST_GALS_Checkpoints/decoymnist_rrr_seed4_bestval_10.75_epoch1_20260303_101952.pth")
    p.add_argument("--afr-ckpt", default="/workspace/logsMNIST/ckpts/afr/decoy_afr_seed4_stage2_20260303_225544.pth")
    p.add_argument("--abn-ckpt", default="/workspace/logsMNIST/ckpts/abn/decoy_abn_seed0_bestval_99.28_test_85.91_epoch17_20260303_230608.pth")
    p.add_argument("--upweight-ckpt", default="/workspace/logsMNIST/ckpts/upweight/decoy_upweight_seed0_bestval_99.58_test_49.98_epoch12_20260303_230554.pth")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if (not str(args.device).startswith("cuda") or torch.cuda.is_available()) else "cpu")
    png_root = Path(args.png_root).expanduser().resolve()
    test_root = png_root / "test"
    if not test_root.is_dir():
        raise RuntimeError(f"Missing DecoyMNIST test dir: {test_root}")

    tf = Compose([Grayscale(num_output_channels=1), ToTensor(), Lambda(lambda t: t * 2.0 - 1.0)])
    test_ds = ImageFolder(str(test_root), transform=tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(device.type == "cuda"),
    )

    def _expand_specs(model_name: str, model_ctor, ckpt_text: str) -> List[ModelSpec]:
        out: List[ModelSpec] = []
        for ckpt_path in _parse_ckpt_list(ckpt_text):
            seed = _extract_seed_from_path(ckpt_path)
            suffix = f"_seed{seed}" if seed is not None else ""
            out.append(ModelSpec(f"{model_name}{suffix}", model_ctor().to(device), ckpt_path, seed))
        return out

    specs: List[ModelSpec] = []
    specs.extend(_expand_specs("guided", LeNet, args.guided_ckpt))
    specs.extend(_expand_specs("vanilla", LeNet, args.vanilla_ckpt))
    specs.extend(_expand_specs("gals", LeNet, args.gals_ckpt))
    specs.extend(_expand_specs("afr", LeNet, args.afr_ckpt))
    specs.extend(_expand_specs("abn", ABNLeNet, args.abn_ckpt))
    specs.extend(_expand_specs("upweight", LeNet, args.upweight_ckpt))

    rows = []
    print(f"[INFO] device={device}")
    print(f"[INFO] test_size={len(test_ds)}")
    print(f"[INFO] classes={test_ds.classes}")
    for spec in specs:
        if not spec.ckpt.is_file():
            raise FileNotFoundError(f"Missing checkpoint for {spec.name}: {spec.ckpt}")
        load_checkpoint_flex(spec.model, spec.ckpt)

        metrics = evaluate_model(spec.model, test_loader, device, num_classes=10)
        overall = float(metrics["overall"])
        mean_cls = float(metrics["mean_cls"])
        worst = float(metrics["worst_acc"])
        class_acc = list(metrics["class_acc"])
        worst_idx = int(metrics["worst_idx"])
        top_mis_idx = int(metrics["top_mis_idx"])

        top_mis_name = _class_name(test_ds.classes, top_mis_idx) if top_mis_idx >= 0 else "none"
        top_mis_count = int(metrics["top_mis_count"])
        top_mis_pct_err = float(metrics["top_mis_pct_of_errors"])
        top_mis_pct_group = float(metrics["top_mis_pct_of_group"])

        row = {
            "model": spec.name,
            "seed": ("" if spec.seed is None else int(spec.seed)),
            "checkpoint": str(spec.ckpt),
            "test_acc": overall,
            "test_mean_class_acc": mean_cls,
            "test_worst_class_acc": worst,
            "worst_group_idx": worst_idx,
            "worst_group_name": _class_name(test_ds.classes, worst_idx),
            "worst_group_total": int(metrics["worst_total"]),
            "worst_group_errors": int(metrics["worst_errors"]),
            "worst_group_top_miscls_idx": ("" if top_mis_idx < 0 else top_mis_idx),
            "worst_group_top_miscls_name": top_mis_name,
            "worst_group_top_miscls_count": top_mis_count,
            "worst_group_top_miscls_pct_of_errors": top_mis_pct_err,
            "worst_group_top_miscls_pct_of_group": top_mis_pct_group,
        }
        for c, v in enumerate(class_acc):
            row[f"class_{c}_acc"] = v
        rows.append(row)

        print(
            f"[RESULT] {spec.name:8s} test_acc={overall:6.2f}% "
            f"mean_class={mean_cls:6.2f}% worst_group={worst_idx}({row['worst_group_name']}) "
            f"worst_acc={worst:6.2f}% top_miscls={top_mis_name} "
            f"(count={top_mis_count}, pct_errors={top_mis_pct_err:5.2f}%, pct_group={top_mis_pct_group:5.2f}%)"
        )

    if args.output_csv:
        out_csv = Path(args.output_csv).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = Path("/workspace/logsMNIST") / f"decoy_multimodel_test_eval_{ts}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    header = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    print(f"[DONE] wrote {out_csv}")


if __name__ == "__main__":
    main()
