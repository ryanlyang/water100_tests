#!/usr/bin/env python3
"""Optuna sweep: CLIP-RN50 LR warm-start + 3-stage guided finetuning on RedMeat.

Recipe implemented per trial:
1) Build CLIP RN50 CAM model.
2) Warm-start classifier by fitting LogisticRegression on frozen backbone features
   (using fixed CLIP+LR hyperparameters).
3) Stage 1: linear_probe (typically CE-only warmup).
4) Stage 2: layer4_head with guidance.
5) Stage 3: full finetune with guidance.
6) Select checkpoint by best validation balanced class accuracy across all stages.
"""

from __future__ import annotations

import argparse
import copy
import csv
import inspect
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_guided_redmeat as rgm


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rgm.base.seed_everything(seed)


def _write_row(path: str, row: Dict[str, object], header: Sequence[str]) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(header))
        if not exists:
            w.writeheader()
        w.writerow(row)


def _format_arr(arr: np.ndarray) -> str:
    return np.array2string(arr, precision=2, separator=",")


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(norms, eps, out=norms)
    x /= norms
    return x


@dataclass
class StageCfg:
    name: str
    tune_mode: str
    epochs: int
    use_attention: bool
    attention_epoch: int
    kl_lambda: float
    base_lr: float
    classifier_lr: float
    kl_incr: float = 0.0


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        rgm.device = self.device
        rgm.base.device = self.device
        self.num_workers = int(args.num_workers)
        self.batch_size = int(args.batch_size)
        self.classes = _parse_csv_list(args.classes) if args.classes else None

        self._build_data()

    def _build_data(self) -> None:
        # CLIP image normalization for clip_rn50 backbone.
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        image_tf = {
            "train": transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)]
            ),
            "eval": transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean, std)]
            ),
        }
        mask_tf = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), rgm.base.Brighten(8.0)]
        )

        self.train_dataset = rgm.RedMeatMetadataDataset(
            data_root=self.args.data_path,
            split="train",
            image_transform=image_tf["train"],
            mask_root=self.args.gt_path,
            mask_transform=mask_tf,
            return_mask=True,
            return_path=True,
            classes=self.classes,
            split_col=self.args.split_col,
            label_col=self.args.label_col,
            path_col=self.args.path_col,
        )
        self.val_dataset = rgm.RedMeatMetadataDataset(
            data_root=self.args.data_path,
            split="val",
            image_transform=image_tf["eval"],
            return_mask=False,
            return_path=True,
            classes=self.train_dataset.classes,
            split_col=self.args.split_col,
            label_col=self.args.label_col,
            path_col=self.args.path_col,
        )
        self.test_dataset = rgm.RedMeatMetadataDataset(
            data_root=self.args.data_path,
            split="test",
            image_transform=image_tf["eval"],
            return_mask=False,
            return_path=True,
            classes=self.train_dataset.classes,
            split_col=self.args.split_col,
            label_col=self.args.label_col,
            path_col=self.args.path_col,
        )

        self.num_classes = len(self.train_dataset.classes)

    def _make_loaders(self, seed: int):
        g = torch.Generator()
        g.manual_seed(seed)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            worker_init_fn=rgm.base.seed_worker,
            generator=g,
        )
        train_eval_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=rgm.base.seed_worker,
            generator=g,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=rgm.base.seed_worker,
            generator=g,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=rgm.base.seed_worker,
            generator=g,
        )
        dataloaders = {"train": train_loader, "val": val_loader}
        dataset_sizes = {"train": len(self.train_dataset), "val": len(self.val_dataset)}
        return dataloaders, dataset_sizes, train_eval_loader, val_loader, test_loader

    @torch.no_grad()
    def _extract_backbone_features(self, model: torch.nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        for batch in loader:
            # train dataset with masks -> (img, label, mask, path)
            if len(batch) >= 2:
                images = batch[0]
                labels = batch[1]
            else:
                raise RuntimeError("Unexpected batch shape while extracting features.")
            images = images.to(self.device)
            outputs, feats = model(images)
            pooled = model.gap(feats).flatten(1).float().cpu().numpy().astype(np.float32, copy=False)
            xs.append(pooled)
            ys.append(labels.numpy().astype(np.int64, copy=False))
        X = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        X = _l2_normalize(X)
        return X, y

    def _fit_lr_head(
        self,
        model: torch.nn.Module,
        train_eval_loader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_train, y_train = self._extract_backbone_features(model, train_eval_loader)

        lr_kwargs = dict(
            C=float(self.args.lr_C),
            penalty=str(self.args.lr_penalty),
            solver=str(self.args.lr_solver),
            fit_intercept=bool(self.args.lr_fit_intercept),
            max_iter=int(self.args.lr_max_iter),
            random_state=int(self.args.seed),
            n_jobs=1,
            verbose=0,
        )
        # sklearn >= 1.8 removed the `multi_class` constructor arg.
        if "multi_class" in inspect.signature(LogisticRegression.__init__).parameters:
            lr_kwargs["multi_class"] = "auto"
        clf = LogisticRegression(**lr_kwargs)
        clf.fit(X_train, y_train)

        feat_dim = model.classifier.weight.shape[1]
        W = np.zeros((self.num_classes, feat_dim), dtype=np.float32)
        b = np.zeros((self.num_classes,), dtype=np.float32)

        if clf.coef_.shape[1] != feat_dim:
            raise ValueError(
                f"LR/head dim mismatch: coef_dim={clf.coef_.shape[1]} vs classifier_dim={feat_dim}"
            )

        for row_idx, cls in enumerate(clf.classes_.tolist()):
            cls = int(cls)
            W[cls] = clf.coef_[row_idx].astype(np.float32, copy=False)
            if self.args.lr_fit_intercept:
                b[cls] = float(clf.intercept_[row_idx])
        return W, b

    @torch.no_grad()
    def _load_lr_head(self, model: torch.nn.Module, W: np.ndarray, b: np.ndarray) -> None:
        model.classifier.weight.copy_(torch.from_numpy(W).to(model.classifier.weight.device))
        model.classifier.bias.copy_(torch.from_numpy(b).to(model.classifier.bias.device))

    def _run_stage(
        self,
        model: torch.nn.Module,
        dataloaders: Dict[str, DataLoader],
        dataset_sizes: Dict[str, int],
        stage: StageCfg,
    ) -> Tuple[torch.nn.Module, float, int]:
        rgm.configure_tune_mode(model, tune_mode=stage.tune_mode)
        print(
            f"[STAGE] {stage.name}: tune_mode={stage.tune_mode} epochs={stage.epochs} "
            f"use_attention={stage.use_attention} attn_epoch={stage.attention_epoch} "
            f"kl={stage.kl_lambda:.4f} base_lr={stage.base_lr:.6g} cls_lr={stage.classifier_lr:.6g}",
            flush=True,
        )
        model, best_score, best_epoch = rgm.base.train_model(
            model,
            dataloaders,
            dataset_sizes,
            stage.attention_epoch,
            stage.kl_lambda,
            stage.epochs,
            base_lr=stage.base_lr,
            classifier_lr=stage.classifier_lr,
            lr2_mult=1.0,
            kl_incr=stage.kl_incr,
            use_attention=stage.use_attention,
            num_classes=self.num_classes,
        )
        return model, float(best_score), int(best_epoch)

    def _eval_val(self, model: torch.nn.Module, val_loader: DataLoader):
        val_loss, val_acc, class_acc, per_group, worst_group = rgm.evaluate_test(
            model, val_loader, self.num_classes
        )
        return float(val_loss), float(val_acc), class_acc, float(per_group), float(worst_group)

    def _eval_test(self, model: torch.nn.Module, test_loader: DataLoader):
        test_loss, test_acc, class_acc, per_group, worst_group = rgm.evaluate_test(
            model, test_loader, self.num_classes
        )
        return float(test_loss), float(test_acc), class_acc, float(per_group), float(worst_group)

    def run_once(self, trial: Any, seed: int) -> Dict[str, object]:
        _seed_everything(seed)

        dataloaders, dataset_sizes, train_eval_loader, val_loader, test_loader = self._make_loaders(seed)

        model = rgm.make_redmeat_cam_model(
            num_classes=self.num_classes,
            model_name="clip_rn50",
            pretrained=True,
            clip_model=self.args.clip_model,
        ).to(self.device)

        # CLIP+LR warm start on classifier.
        W, b = self._fit_lr_head(model, train_eval_loader)
        self._load_lr_head(model, W, b)

        # Baseline eval right after LR warm-start.
        _, warm_val_acc, warm_val_cls, warm_val_mean, warm_val_worst = self._eval_val(model, val_loader)
        _, warm_test_acc, warm_test_cls, warm_test_mean, warm_test_worst = self._eval_test(model, test_loader)

        # Sample staged hyperparams.
        s1_epochs = trial.suggest_int("s1_epochs", 3, 15)
        s2_epochs = trial.suggest_int("s2_epochs", 20, 80)
        s3_epochs = trial.suggest_int("s3_epochs", 10, 60)

        s1_cls_lr = trial.suggest_float("s1_classifier_lr", 1e-5, 5e-2, log=True)
        s2_base_lr = trial.suggest_float("s2_base_lr", 1e-6, 5e-3, log=True)
        s2_cls_lr = trial.suggest_float("s2_classifier_lr", 1e-5, 5e-3, log=True)
        s3_base_lr = trial.suggest_float("s3_base_lr", 1e-7, 5e-4, log=True)
        s3_cls_lr = trial.suggest_float("s3_classifier_lr", 1e-6, 5e-4, log=True)

        s2_kl = trial.suggest_float("s2_kl_lambda", 1.0, 500.0, log=True)
        s3_kl_mult = trial.suggest_float("s3_kl_mult", 0.3, 1.5)
        s3_kl = float(np.clip(s2_kl * s3_kl_mult, 0.5, 600.0))

        s2_attn_ep = trial.suggest_int("s2_attention_epoch", 0, max(0, min(20, s2_epochs - 1)))
        s3_attn_ep = trial.suggest_int("s3_attention_epoch", 0, max(0, min(15, s3_epochs - 1)))

        stages = [
            StageCfg(
                name="stage1_linear_probe",
                tune_mode="linear_probe",
                epochs=s1_epochs,
                use_attention=False,
                attention_epoch=s1_epochs,  # ignored because use_attention=False
                kl_lambda=0.0,
                base_lr=s1_cls_lr,
                classifier_lr=s1_cls_lr,
                kl_incr=0.0,
            ),
            StageCfg(
                name="stage2_layer4_head_guided",
                tune_mode="layer4_head",
                epochs=s2_epochs,
                use_attention=True,
                attention_epoch=s2_attn_ep,
                kl_lambda=s2_kl,
                base_lr=s2_base_lr,
                classifier_lr=s2_cls_lr,
                kl_incr=0.0,
            ),
            StageCfg(
                name="stage3_full_guided",
                tune_mode="full",
                epochs=s3_epochs,
                use_attention=True,
                attention_epoch=s3_attn_ep,
                kl_lambda=s3_kl,
                base_lr=s3_base_lr,
                classifier_lr=s3_cls_lr,
                kl_incr=0.0,
            ),
        ]

        best_val_mean = -1.0
        best_state = copy.deepcopy(model.state_dict())
        best_stage_name = "warm_start"

        stage_rows: Dict[str, float] = {}
        for stage in stages:
            model, stage_best_score, stage_best_epoch = self._run_stage(
                model=model,
                dataloaders=dataloaders,
                dataset_sizes=dataset_sizes,
                stage=stage,
            )

            _, val_acc, val_cls, val_mean, val_worst = self._eval_val(model, val_loader)
            _, test_acc, test_cls, test_mean, test_worst = self._eval_test(model, test_loader)

            stage_rows[f"{stage.name}_best_score"] = float(stage_best_score)
            stage_rows[f"{stage.name}_best_epoch"] = float(stage_best_epoch)
            stage_rows[f"{stage.name}_val_acc"] = float(val_acc)
            stage_rows[f"{stage.name}_val_mean"] = float(val_mean)
            stage_rows[f"{stage.name}_val_worst"] = float(val_worst)
            stage_rows[f"{stage.name}_test_acc"] = float(test_acc)
            stage_rows[f"{stage.name}_test_mean"] = float(test_mean)
            stage_rows[f"{stage.name}_test_worst"] = float(test_worst)

            if val_mean > best_val_mean:
                best_val_mean = val_mean
                best_state = copy.deepcopy(model.state_dict())
                best_stage_name = stage.name

        # Select best-by-val across warm + all stages.
        if warm_val_mean > best_val_mean:
            best_val_mean = warm_val_mean
            best_stage_name = "warm_start"
            # model currently already moved; rebuild and load warm-start for consistent eval.
            best_model = rgm.make_redmeat_cam_model(
                num_classes=self.num_classes,
                model_name="clip_rn50",
                pretrained=True,
                clip_model=self.args.clip_model,
            ).to(self.device)
            self._load_lr_head(best_model, W, b)
        else:
            best_model = rgm.make_redmeat_cam_model(
                num_classes=self.num_classes,
                model_name="clip_rn50",
                pretrained=True,
                clip_model=self.args.clip_model,
            ).to(self.device)
            best_model.load_state_dict(best_state)

        _, best_test_acc, best_test_cls, best_test_mean, best_test_worst = self._eval_test(best_model, test_loader)
        _, best_val_acc, best_val_cls, best_val_mean_2, best_val_worst = self._eval_val(best_model, val_loader)

        out: Dict[str, object] = {
            "seed": int(seed),
            "selected_stage": best_stage_name,
            "warm_val_acc": warm_val_acc,
            "warm_val_mean": warm_val_mean,
            "warm_val_worst": warm_val_worst,
            "warm_test_acc": warm_test_acc,
            "warm_test_mean": warm_test_mean,
            "warm_test_worst": warm_test_worst,
            "warm_val_class_accs": _format_arr(warm_val_cls),
            "warm_test_class_accs": _format_arr(warm_test_cls),
            "best_val_acc": best_val_acc,
            "best_val_mean": best_val_mean_2,
            "best_val_worst": best_val_worst,
            "best_test_acc": best_test_acc,
            "best_test_mean": best_test_mean,
            "best_test_worst": best_test_worst,
            "best_val_class_accs": _format_arr(best_val_cls),
            "best_test_class_accs": _format_arr(best_test_cls),
            **stage_rows,
        }
        return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Optuna sweep: CLIP-RN50 LR warm-start + 3-stage guided finetuning on RedMeat."
    )
    p.add_argument("data_path")
    p.add_argument("gt_path")
    p.add_argument("--split-col", default="split")
    p.add_argument("--label-col", default="label")
    p.add_argument("--path-col", default="abs_file_path")
    p.add_argument(
        "--classes",
        default="prime_rib,pork_chop,steak,baby_back_ribs,filet_mignon",
        help="Comma-separated classes; empty to infer from metadata.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--sampler", choices=["tpe", "random"], default="tpe")
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--clip-model", default="RN50")
    p.add_argument("--no-cuda", action="store_true", default=False)

    # Fixed CLIP+LR hyperparameters (defaults from user-provided best RedMeat trial).
    p.add_argument("--lr-C", type=float, default=5.302446323656201)
    p.add_argument("--lr-penalty", type=str, default="l2")
    p.add_argument("--lr-solver", type=str, default="lbfgs")
    p.add_argument("--lr-fit-intercept", action="store_true", default=False)
    p.add_argument("--lr-max-iter", type=int, default=5000)

    p.add_argument("--output-csv", required=True)
    p.add_argument("--post-seeds", type=int, default=0, help="If >0, rerun best trial on seeds [post-seed-start..).")
    p.add_argument("--post-seed-start", type=int, default=0)
    p.add_argument("--post-output-csv", type=str, default="")
    args = p.parse_args()

    import optuna

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or ".", exist_ok=True)

    header = [
        "trial",
        "sampler",
        "seconds",
        "seed",
        "selected_stage",
        "best_val_acc",
        "best_val_mean",
        "best_val_worst",
        "best_test_acc",
        "best_test_mean",
        "best_test_worst",
        "best_val_class_accs",
        "best_test_class_accs",
        "warm_val_acc",
        "warm_val_mean",
        "warm_val_worst",
        "warm_test_acc",
        "warm_test_mean",
        "warm_test_worst",
        "warm_val_class_accs",
        "warm_test_class_accs",
        "s1_epochs",
        "s2_epochs",
        "s3_epochs",
        "s1_classifier_lr",
        "s2_base_lr",
        "s2_classifier_lr",
        "s3_base_lr",
        "s3_classifier_lr",
        "s2_kl_lambda",
        "s3_kl_mult",
        "s3_kl_lambda",
        "s2_attention_epoch",
        "s3_attention_epoch",
        "stage1_linear_probe_best_score",
        "stage1_linear_probe_best_epoch",
        "stage1_linear_probe_val_acc",
        "stage1_linear_probe_val_mean",
        "stage1_linear_probe_val_worst",
        "stage1_linear_probe_test_acc",
        "stage1_linear_probe_test_mean",
        "stage1_linear_probe_test_worst",
        "stage2_layer4_head_guided_best_score",
        "stage2_layer4_head_guided_best_epoch",
        "stage2_layer4_head_guided_val_acc",
        "stage2_layer4_head_guided_val_mean",
        "stage2_layer4_head_guided_val_worst",
        "stage2_layer4_head_guided_test_acc",
        "stage2_layer4_head_guided_test_mean",
        "stage2_layer4_head_guided_test_worst",
        "stage3_full_guided_best_score",
        "stage3_full_guided_best_epoch",
        "stage3_full_guided_val_acc",
        "stage3_full_guided_val_mean",
        "stage3_full_guided_val_worst",
        "stage3_full_guided_test_acc",
        "stage3_full_guided_test_mean",
        "stage3_full_guided_test_worst",
    ]

    runner = Runner(args)
    rng = np.random.default_rng(args.seed)
    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=args.seed)
    else:
        sampler = optuna.samplers.RandomSampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.trial.Trial) -> float:
        t0 = time.time()
        row = runner.run_once(trial, seed=args.seed)
        seconds = int(time.time() - t0)
        row_out: Dict[str, object] = {
            "trial": int(trial.number),
            "sampler": args.sampler,
            "seconds": seconds,
            "s1_epochs": int(trial.params["s1_epochs"]),
            "s2_epochs": int(trial.params["s2_epochs"]),
            "s3_epochs": int(trial.params["s3_epochs"]),
            "s1_classifier_lr": float(trial.params["s1_classifier_lr"]),
            "s2_base_lr": float(trial.params["s2_base_lr"]),
            "s2_classifier_lr": float(trial.params["s2_classifier_lr"]),
            "s3_base_lr": float(trial.params["s3_base_lr"]),
            "s3_classifier_lr": float(trial.params["s3_classifier_lr"]),
            "s2_kl_lambda": float(trial.params["s2_kl_lambda"]),
            "s3_kl_mult": float(trial.params["s3_kl_mult"]),
            "s3_kl_lambda": float(np.clip(float(trial.params["s2_kl_lambda"]) * float(trial.params["s3_kl_mult"]), 0.5, 600.0)),
            "s2_attention_epoch": int(trial.params["s2_attention_epoch"]),
            "s3_attention_epoch": int(trial.params["s3_attention_epoch"]),
            **row,
        }
        _write_row(args.output_csv, row_out, header)
        val_obj = float(row_out["best_val_mean"])
        print(
            f"[TRIAL {trial.number}] best_val_mean={val_obj:.4f} "
            f"best_test_mean={float(row_out['best_test_mean']):.4f} "
            f"selected_stage={row_out['selected_stage']}",
            flush=True,
        )
        return val_obj

    study.optimize(objective, n_trials=int(args.n_trials))

    print("\n[SWEEP] Best trial")
    print(f"  trial={study.best_trial.number}")
    print(f"  value(best_val_mean)={study.best_value:.4f}")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    if int(args.post_seeds) > 0:
        post_csv = args.post_output_csv or os.path.splitext(args.output_csv)[0] + "_best_post.csv"
        post_header = ["seed", *header]
        start = int(args.post_seed_start)
        seeds = [start + i for i in range(int(args.post_seeds))]
        print(f"\n[POST] Rerunning best trial on seeds={seeds}")
        best_params = study.best_trial.params

        means = []
        for seed in seeds:
            fixed = optuna.trial.FixedTrial(best_params)
            t0 = time.time()
            row = runner.run_once(fixed, seed=seed)
            seconds = int(time.time() - t0)
            row_out = {
                "seed": seed,
                "trial": study.best_trial.number,
                "sampler": "post_best",
                "seconds": seconds,
                "s1_epochs": int(best_params["s1_epochs"]),
                "s2_epochs": int(best_params["s2_epochs"]),
                "s3_epochs": int(best_params["s3_epochs"]),
                "s1_classifier_lr": float(best_params["s1_classifier_lr"]),
                "s2_base_lr": float(best_params["s2_base_lr"]),
                "s2_classifier_lr": float(best_params["s2_classifier_lr"]),
                "s3_base_lr": float(best_params["s3_base_lr"]),
                "s3_classifier_lr": float(best_params["s3_classifier_lr"]),
                "s2_kl_lambda": float(best_params["s2_kl_lambda"]),
                "s3_kl_mult": float(best_params["s3_kl_mult"]),
                "s3_kl_lambda": float(np.clip(float(best_params["s2_kl_lambda"]) * float(best_params["s3_kl_mult"]), 0.5, 600.0)),
                "s2_attention_epoch": int(best_params["s2_attention_epoch"]),
                "s3_attention_epoch": int(best_params["s3_attention_epoch"]),
                **row,
            }
            _write_row(post_csv, row_out, post_header)
            means.append(float(row_out["best_test_mean"]))
            print(
                f"[POST] seed={seed} best_val_mean={float(row_out['best_val_mean']):.4f} "
                f"best_test_mean={float(row_out['best_test_mean']):.4f}",
                flush=True,
            )
        arr = np.asarray(means, dtype=np.float64)
        print(f"[POST] best_test_mean: {arr.mean():.4f} +/- {arr.std():.4f} (n={arr.size})")
        print(f"[POST] wrote {post_csv}")


if __name__ == "__main__":
    main()
