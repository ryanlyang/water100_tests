#!/usr/bin/env python3
"""Single-run RedMeat MobileNetV2 + R4RR runner."""

import argparse
import sys
from pathlib import Path


MOBILENET_ROOT = Path(__file__).resolve().parents[1]
if str(MOBILENET_ROOT) not in sys.path:
    sys.path.insert(0, str(MOBILENET_ROOT))
import common  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="R4RR RedMeat runner with MobileNetV2 student.")
    p.add_argument("data_path", help="RedMeat dataset root containing all_images.csv")
    p.add_argument("teacher_map_path", help="Teacher-map root folder")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attention-epoch", "--attention_epoch", dest="attention_epoch", type=int, default=150)
    p.add_argument("--kl-lambda", "--kl_lambda", dest="kl_lambda", type=float, default=0.0)
    p.add_argument("--kl-increment", "--kl_increment", dest="kl_increment", type=float, default=None)

    p.add_argument("--base-lr", "--base_lr", dest="base_lr", type=float, default=0.01)
    p.add_argument("--classifier-lr", "--classifier_lr", dest="classifier_lr", type=float, default=0.01)
    p.add_argument("--lr2-mult", "--lr2_mult", dest="lr2_mult", type=float, default=1.0)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=1e-5)
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=96)
    p.add_argument("--num-epochs", "--num_epochs", dest="num_epochs", type=int, default=150)
    p.add_argument("--img-size", "--img_size", dest="img_size", type=int, default=224)
    p.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=None)
    p.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir", default="MobileNetV2_R4RR_RedMeat_Checkpoints")

    p.add_argument("--split-col", "--split_col", dest="split_col", default="split")
    p.add_argument("--label-col", "--label_col", dest="label_col", default="label")
    p.add_argument("--path-col", "--path_col", dest="path_col", default="abs_file_path")
    p.add_argument("--classes", default=common.DEFAULT_REMEAT_CLASSES)

    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    return p.parse_args()


def main():
    args = parse_args()
    common.run_guided_redmeat(args, int(args.attention_epoch), float(args.kl_lambda), args.kl_increment)


if __name__ == "__main__":
    main()
