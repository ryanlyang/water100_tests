#!/usr/bin/env python3
"""Quick environment checker for the r4rr-runs setup.

Usage:
  python pipelines/check_r4rr_runs_env.py
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
from dataclasses import dataclass
from importlib import metadata


@dataclass(frozen=True)
class PackageCheck:
    pip_name: str
    import_name: str


REQUIRED_PACKAGES = [
    PackageCheck("torch", "torch"),
    PackageCheck("torchvision", "torchvision"),
    PackageCheck("numpy", "numpy"),
    PackageCheck("pandas", "pandas"),
    PackageCheck("scipy", "scipy"),
    PackageCheck("scikit-learn", "sklearn"),
    PackageCheck("scikit-image", "skimage"),
    PackageCheck("Pillow", "PIL"),
    PackageCheck("opencv-python", "cv2"),
    PackageCheck("matplotlib", "matplotlib"),
    PackageCheck("tqdm", "tqdm"),
    PackageCheck("omegaconf", "omegaconf"),
    PackageCheck("optuna", "optuna"),
    PackageCheck("wandb", "wandb"),
    PackageCheck("pyyaml", "yaml"),
    PackageCheck("ftfy", "ftfy"),
    PackageCheck("regex", "regex"),
    PackageCheck("requests", "requests"),
    PackageCheck("pycocotools", "pycocotools"),
    PackageCheck("torchray", "torchray"),
    PackageCheck("timm", "timm"),
    PackageCheck("einops", "einops"),
    PackageCheck("wilds", "wilds"),
]


def _get_version(pip_name: str) -> str:
    try:
        return metadata.version(pip_name)
    except metadata.PackageNotFoundError:
        return "unknown"


def main() -> int:
    print("[R4RR RUNS ENV CHECK]")
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"conda_env: {os.environ.get('CONDA_DEFAULT_ENV', '<none>')}")
    print()

    if sys.version_info < (3, 10):
        print("[WARN] Python < 3.10 detected. Repo defaults target Python 3.10 for r4rr-runs.")

    missing = []
    failed = []
    ok = []

    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg.import_name)
            ok.append((pkg.pip_name, pkg.import_name, _get_version(pkg.pip_name)))
        except ModuleNotFoundError:
            missing.append((pkg.pip_name, pkg.import_name))
        except Exception as exc:  # noqa: BLE001
            failed.append((pkg.pip_name, pkg.import_name, repr(exc)))

    try:
        import torch  # type: ignore

        print("[TORCH]")
        print(f"torch: {torch.__version__}")
        print(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda_device_count: {torch.cuda.device_count()}")
            print(f"cuda_device_0: {torch.cuda.get_device_name(0)}")
        print()
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not query torch CUDA info: {exc!r}")
        print()

    print(f"[OK] {len(ok)} packages imported")
    for pip_name, import_name, version in ok:
        print(f"  - {pip_name} (import: {import_name}) v{version}")

    if missing:
        print()
        print(f"[MISSING] {len(missing)} packages")
        for pip_name, import_name in missing:
            print(f"  - {pip_name} (import: {import_name})")

    if failed:
        print()
        print(f"[FAILED IMPORT] {len(failed)} packages")
        for pip_name, import_name, err in failed:
            print(f"  - {pip_name} (import: {import_name}) -> {err}")

    if missing or failed:
        print()
        print("[RESULT] FAIL")
        print("Install/fix missing deps, then re-run this checker.")
        return 1

    print()
    print("[RESULT] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
