#!/usr/bin/env python3
"""Quick environment checker for the r4rr-weclip setup.

Usage:
  python pipelines/check_r4rr_weclip_env.py
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path


@dataclass(frozen=True)
class PackageCheck:
    pip_names: tuple[str, ...]
    import_name: str


REQUIRED_PACKAGES = [
    PackageCheck(("torch",), "torch"),
    PackageCheck(("torchvision",), "torchvision"),
    PackageCheck(("mmcv", "mmcv-full", "mmcv_full"), "mmcv"),
    PackageCheck(("matplotlib",), "matplotlib"),
    PackageCheck(("tqdm",), "tqdm"),
    PackageCheck(("omegaconf",), "omegaconf"),
    PackageCheck(("numpy",), "numpy"),
    PackageCheck(("timm",), "timm"),
    PackageCheck(("imageio",), "imageio"),
    PackageCheck(("Pillow",), "PIL"),
    PackageCheck(("scikit-learn",), "sklearn"),
    PackageCheck(("tensorboard",), "tensorboard"),
    PackageCheck(("ftfy",), "ftfy"),
    PackageCheck(("regex",), "regex"),
    PackageCheck(("ttach",), "ttach"),
    PackageCheck(("lxml",), "lxml"),
    PackageCheck(("tensorflow",), "tensorflow"),
    PackageCheck(("colour",), "colour"),
    PackageCheck(("open_clip_torch",), "open_clip"),
    PackageCheck(("pydensecrf",), "pydensecrf"),
]


def _get_version(candidates: tuple[str, ...]) -> str:
    for name in candidates:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return "unknown"


def _is_direct_missing(pkg_import_name: str, exc: ModuleNotFoundError) -> bool:
    """True when the package itself is missing, not an internal dependency."""
    if not getattr(exc, "name", None):
        return False
    missing_root = str(exc.name).split(".")[0]
    pkg_root = str(pkg_import_name).split(".")[0]
    return missing_root == pkg_root


def main() -> int:
    print("[R4RR WECLIP ENV CHECK]")
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"conda_env: {os.environ.get('CONDA_DEFAULT_ENV', '<none>')}")
    print()

    if sys.version_info[:2] != (3, 8):
        print("[WARN] Python 3.8 is recommended for the WeCLIP environment.")

    missing: list[tuple[tuple[str, ...], str]] = []
    failed: list[tuple[tuple[str, ...], str, str]] = []
    ok: list[tuple[tuple[str, ...], str, str]] = []

    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg.import_name)
            ok.append((pkg.pip_names, pkg.import_name, _get_version(pkg.pip_names)))
        except ModuleNotFoundError as exc:
            if _is_direct_missing(pkg.import_name, exc):
                missing.append((pkg.pip_names, pkg.import_name))
            else:
                failed.append((pkg.pip_names, pkg.import_name, repr(exc)))
        except Exception as exc:  # noqa: BLE001
            failed.append((pkg.pip_names, pkg.import_name, repr(exc)))

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

    repo_root = Path(__file__).resolve().parents[1]
    clip_weight = repo_root / "WeCLIPPlus" / "pretrained" / "ViT-B-16.pt"
    if clip_weight.exists():
        print(f"[OK] CLIP weight found: {clip_weight}")
    else:
        print(f"[WARN] Missing CLIP weight: {clip_weight}")
        print("       Download link is documented in WeCLIPPlus/README.md")
    print()

    print(f"[OK] {len(ok)} packages imported")
    for pip_names, import_name, version in ok:
        display = "/".join(pip_names)
        print(f"  - {display} (import: {import_name}) v{version}")

    if missing:
        print()
        print(f"[MISSING] {len(missing)} packages")
        for pip_names, import_name in missing:
            display = "/".join(pip_names)
            print(f"  - {display} (import: {import_name})")

    if failed:
        print()
        print(f"[FAILED IMPORT] {len(failed)} packages")
        for pip_names, import_name, err in failed:
            display = "/".join(pip_names)
            print(f"  - {display} (import: {import_name}) -> {err}")

    if missing or failed:
        print()
        print("[RESULT] FAIL")
        if any("pydensecrf" in names for names, _ in missing):
            print("Hint: install pydensecrf with: conda install -y -c conda-forge pydensecrf")
        if any(("timm" in names and "torch._six" in err) for names, _, err in failed):
            print("Hint: timm version is too old for this torch build. Use: pip install 'timm>=0.9,<1.1'")
        print("Install/fix missing deps, then re-run this checker.")
        return 1

    print("[RESULT] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
