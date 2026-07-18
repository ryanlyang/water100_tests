"""In-memory privileged Oracle view for the DecoyMNIST FCV campaign."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
from PIL import Image

from decoy_data import expected_patch_intensity, locate_decoy_patch
from decoy_manifest_provenance import validate_manifest_bundle


class OracleViewError(ValueError):
    """Raised when an Oracle source cannot be safely reversed in memory."""


def reverse_training_patch(grayscale: np.ndarray, label: int) -> np.ndarray:
    """Return a copy with the training patch replaced by official test encoding."""

    source = np.asarray(grayscale)
    if source.shape != (28, 28):
        raise OracleViewError(f"Expected a 28x28 grayscale source, found {source.shape}.")
    if source.dtype != np.uint8:
        raise OracleViewError(f"Expected uint8 source pixels, found {source.dtype}.")
    rows, columns = locate_decoy_patch(source, int(label), "train")
    reversed_view = source.copy()
    reversed_view[rows, columns] = expected_patch_intensity(int(label), "test")
    return reversed_view


def load_oracle_view(
    path: str | Path,
    label: int,
    *,
    expected_sha256: str | None = None,
) -> Image.Image:
    """Read one immutable source PNG and return its reversed RGB Oracle view."""

    image_path = Path(path).expanduser().resolve()
    raw = image_path.read_bytes()
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and observed_sha256 != str(expected_sha256):
        raise OracleViewError(
            f"Source image changed after manifest creation: {image_path}; "
            f"expected {expected_sha256}, observed {observed_sha256}."
        )
    with Image.open(image_path) as image:
        if image.mode != "L":
            raise OracleViewError(
                f"Expected mode-L DecoyMNIST source, found {image.mode!r}: {image_path}"
            )
        grayscale = np.asarray(image, dtype=np.uint8).copy()
    return Image.fromarray(reverse_training_patch(grayscale, int(label)), mode="L").convert(
        "RGB"
    )


class OracleViewDataset:
    """Dataset over the analysis-only Oracle-source manifest.

    The transformed view exists only as a local PIL object (or the return value
    of ``transform``). No transformed image is written to disk.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        manifest_path: str | Path,
        transform: Callable[[Image.Image], Any] | None = None,
    ) -> None:
        self.config = config
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.binding = validate_manifest_bundle(
            config, self.manifest_path, "oracle_validation"
        )
        self.frame = pd.read_csv(self.manifest_path)
        if set(self.frame["study_split"].astype(str)) != {
            "oracle_validation_source_analysis_only"
        }:
            raise OracleViewError("OracleViewDataset requires the analysis-only source.")
        self.data_root = Path(config["paths"]["data_root"]).expanduser().resolve()
        self.transform = transform

    def __len__(self) -> int:
        return int(len(self.frame))

    def __getitem__(self, index: int):
        row = self.frame.iloc[int(index)]
        source_path = self.data_root / str(row["image_rel_path"])
        image = load_oracle_view(
            source_path,
            int(row["label"]),
            expected_sha256=str(row["image_sha256"]),
        )
        output = self.transform(image) if self.transform is not None else image
        return output, int(row["label"]), str(row["sample_id"])

