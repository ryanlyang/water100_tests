#!/usr/bin/env python3
"""Create deterministic DecoyMNIST data and oracle surrogate teacher maps.

The image construction follows the CDEP DecoyMNIST protocol used by the
project: a label-coded 5x5 patch is placed in a random corner, with the code
reversed at test time.  Teacher maps are binary masks of the corresponding
clean MNIST digit and therefore exclude the synthetic corner shortcut.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tarfile
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from PIL import Image
from torchvision.datasets import MNIST


PROTOCOL_VERSION = 1
EXPECTED_TRAIN = 60_000
EXPECTED_TEST = 10_000


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    Image.fromarray(array.astype(np.uint8, copy=False), mode="L").save(
        str(temporary), format="PNG"
    )
    os.replace(str(temporary), str(path))


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(str(temporary), str(path))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def corner_name(row: int, col: int) -> str:
    return {
        (0, 0): "top_left",
        (0, 23): "top_right",
        (23, 0): "bottom_left",
        (23, 23): "bottom_right",
    }[(int(row), int(col))]


def expected_or_write_png(path: Path, expected: np.ndarray) -> str:
    """Resume safely: existing files must decode exactly to the expected pixels."""
    if path.exists():
        with Image.open(str(path)) as image:
            observed = np.array(image.convert("L"), dtype=np.uint8, copy=True)
        if not np.array_equal(observed, expected):
            raise RuntimeError("Existing PNG does not match generation contract: {}".format(path))
        return "validated"
    atomic_save_png(path, expected)
    return "written"


def make_decoy_image(
    clean: np.ndarray,
    label: int,
    row: int,
    col: int,
    split: str,
) -> Tuple[np.ndarray, int]:
    image = np.asarray(clean, dtype=np.uint8).copy()
    if split == "train":
        patch_value = 255 - 25 * int(label)
    elif split == "test":
        patch_value = 25 * int(label)
    else:
        raise ValueError("Unsupported split: {}".format(split))
    image[int(row) : int(row) + 5, int(col) : int(col) + 5] = patch_value
    return image, int(patch_value)


def update_tree_digest(
    digest: "hashlib._Hash", relative_path: str, array: np.ndarray
) -> None:
    encoded_path = relative_path.encode("utf-8")
    digest.update(len(encoded_path).to_bytes(4, byteorder="big"))
    digest.update(encoded_path)
    digest.update(np.ascontiguousarray(array).tobytes())


def generate_split(
    split: str,
    clean_images: np.ndarray,
    labels: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    dataset_root: Path,
    teacher_root: Path,
    mask_threshold: int,
    csv_writer: csv.DictWriter,
    image_digest: "hashlib._Hash",
    teacher_digest: "hashlib._Hash",
    progress_every: int,
) -> Dict[str, object]:
    written_images = 0
    validated_images = 0
    written_maps = 0
    validated_maps = 0
    class_counts = {str(digit): 0 for digit in range(10)}
    corner_counts = {
        "top_left": 0,
        "top_right": 0,
        "bottom_left": 0,
        "bottom_right": 0,
    }

    for index in range(len(labels)):
        label = int(labels[index])
        row = int(rows[index])
        col = int(cols[index])
        image, patch_value = make_decoy_image(
            clean_images[index], label, row, col, split
        )
        image_rel = Path(split) / str(label) / "{:06d}_y{}.png".format(index, label)
        image_path = dataset_root / image_rel
        image_status = expected_or_write_png(image_path, image)
        written_images += int(image_status == "written")
        validated_images += int(image_status == "validated")
        update_tree_digest(image_digest, image_rel.as_posix(), image)

        teacher_rel: Optional[Path] = None
        if split == "train":
            teacher = (clean_images[index] > int(mask_threshold)).astype(np.uint8) * 255
            if int(np.count_nonzero(teacher)) == 0:
                raise RuntimeError("Empty clean-digit teacher map at train index {}".format(index))
            teacher_rel = Path("{}_{}".format(label, image_path.name))
            teacher_path = teacher_root / teacher_rel
            teacher_status = expected_or_write_png(teacher_path, teacher)
            written_maps += int(teacher_status == "written")
            validated_maps += int(teacher_status == "validated")
            update_tree_digest(teacher_digest, teacher_rel.as_posix(), teacher)

        class_counts[str(label)] += 1
        corner_counts[corner_name(row, col)] += 1
        csv_writer.writerow(
            {
                "split": split,
                "source_index": index,
                "label": label,
                "corner": corner_name(row, col),
                "corner_row": row,
                "corner_col": col,
                "patch_value_uint8": patch_value,
                "image_relative_path": image_rel.as_posix(),
                "teacher_relative_path": teacher_rel.as_posix() if teacher_rel else "",
            }
        )

        completed = index + 1
        if completed == len(labels) or completed % int(progress_every) == 0:
            print("[PROGRESS] {} {}/{}".format(split, completed, len(labels)), flush=True)

    return {
        "examples": int(len(labels)),
        "images_written": int(written_images),
        "images_validated": int(validated_images),
        "teacher_maps_written": int(written_maps),
        "teacher_maps_validated": int(validated_maps),
        "class_counts": class_counts,
        "corner_counts": corner_counts,
    }


def count_pngs(path: Path) -> int:
    return sum(1 for item in path.rglob("*.png") if item.is_file())


def make_archive(output_root: Path, archive_path: Path) -> Dict[str, object]:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_name(".{}.{}.tmp".format(archive_path.name, os.getpid()))
    mode = "w:gz" if archive_path.name.endswith((".tar.gz", ".tgz")) else "w"
    with tarfile.open(str(temporary), mode) as archive:
        archive.add(str(output_root), arcname=output_root.name, recursive=True)
    os.replace(str(temporary), str(archive_path))
    checksum = sha256_file(archive_path)
    atomic_write_text(
        archive_path.with_name(archive_path.name + ".sha256"),
        "{}  {}\n".format(checksum, archive_path.name),
    )
    return {
        "path": str(archive_path.resolve()),
        "bytes": int(archive_path.stat().st_size),
        "sha256": checksum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic CDEP-style DecoyMNIST and oracle surrogate maps."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mnist-root", type=Path, required=True)
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--mask-threshold", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--archive-path", type=Path, default=None)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Require canonical MNIST to already exist instead of downloading it.",
    )
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    mnist_root = args.mnist_root.expanduser().resolve()
    dataset_root = output_root / "DecoyMNIST_png"
    teacher_root = output_root / "teacher_maps" / "prediction_cmap"
    metadata_root = output_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)

    contract = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "DecoyMNIST",
        "dataset_variant": "CDEP_5x5_label_code_reversed_test",
        "dataset_seed": int(args.dataset_seed),
        "patch_size": 5,
        "corner_offsets": [0, 23],
        "train_patch_value": "255 - 25 * label",
        "test_patch_value": "25 * label",
        "teacher_source": "oracle_clean_torchvision_mnist_foreground",
        "teacher_mask_threshold": int(args.mask_threshold),
        "teacher_includes_shortcut_patch": False,
        "teacher_map_encoding": "uint8_binary_0_or_255_png",
        "teacher_map_scope": "train_60000_only",
        "dataset_root": str(dataset_root),
        "teacher_map_root": str(teacher_root),
    }
    contract_path = metadata_root / "generation_contract.json"
    if contract_path.exists():
        observed_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if observed_contract != contract:
            raise RuntimeError(
                "Existing output was generated under a different contract: {}".format(
                    contract_path
                )
            )
    else:
        atomic_write_json(contract_path, contract)

    print("[INFO] output_root={}".format(output_root))
    print("[INFO] dataset_root={}".format(dataset_root))
    print("[INFO] teacher_map_root={}".format(teacher_root))
    print("[INFO] mnist_root={}".format(mnist_root))
    print("[INFO] dataset_seed={}".format(args.dataset_seed))
    print("[INFO] teacher=clean MNIST foreground threshold>{}".format(args.mask_threshold))

    download = not bool(args.no_download)
    train_mnist = MNIST(root=str(mnist_root), train=True, download=download, transform=None)
    test_mnist = MNIST(root=str(mnist_root), train=False, download=download, transform=None)
    train_images = np.asarray(train_mnist.data, dtype=np.uint8)
    train_labels = np.asarray(train_mnist.targets, dtype=np.int64)
    test_images = np.asarray(test_mnist.data, dtype=np.uint8)
    test_labels = np.asarray(test_mnist.targets, dtype=np.int64)
    if len(train_labels) != EXPECTED_TRAIN or len(test_labels) != EXPECTED_TEST:
        raise RuntimeError(
            "Canonical MNIST size mismatch: train={} test={}".format(
                len(train_labels), len(test_labels)
            )
        )

    rng = np.random.RandomState(int(args.dataset_seed))
    train_rows = rng.choice(2, size=EXPECTED_TRAIN).astype(np.int16) * 23
    train_cols = rng.choice(2, size=EXPECTED_TRAIN).astype(np.int16) * 23
    test_rows = rng.choice(2, size=EXPECTED_TEST).astype(np.int16) * 23
    test_cols = rng.choice(2, size=EXPECTED_TEST).astype(np.int16) * 23
    corners_path = metadata_root / "corner_assignments.npz"
    expected_corners = {
        "train_rows": train_rows,
        "train_cols": train_cols,
        "test_rows": test_rows,
        "test_cols": test_cols,
    }
    if corners_path.exists():
        observed = np.load(str(corners_path), allow_pickle=False)
        for key, expected in expected_corners.items():
            if key not in observed.files or not np.array_equal(observed[key], expected):
                raise RuntimeError("Corner-assignment mismatch in {}".format(corners_path))
    else:
        atomic_save_npz(corners_path, **expected_corners)

    sample_manifest_path = metadata_root / "sample_manifest.csv"
    temporary_csv = sample_manifest_path.with_name(
        ".{}.{}.tmp".format(sample_manifest_path.name, os.getpid())
    )
    fields = [
        "split",
        "source_index",
        "label",
        "corner",
        "corner_row",
        "corner_col",
        "patch_value_uint8",
        "image_relative_path",
        "teacher_relative_path",
    ]
    image_digest = hashlib.sha256()
    teacher_digest = hashlib.sha256()
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        train_summary = generate_split(
            "train",
            train_images,
            train_labels,
            train_rows,
            train_cols,
            dataset_root,
            teacher_root,
            int(args.mask_threshold),
            writer,
            image_digest,
            teacher_digest,
            int(args.progress_every),
        )
        test_summary = generate_split(
            "test",
            test_images,
            test_labels,
            test_rows,
            test_cols,
            dataset_root,
            teacher_root,
            int(args.mask_threshold),
            writer,
            image_digest,
            teacher_digest,
            int(args.progress_every),
        )
    os.replace(str(temporary_csv), str(sample_manifest_path))

    observed_train = count_pngs(dataset_root / "train")
    observed_test = count_pngs(dataset_root / "test")
    observed_teachers = count_pngs(teacher_root)
    if (observed_train, observed_test, observed_teachers) != (
        EXPECTED_TRAIN,
        EXPECTED_TEST,
        EXPECTED_TRAIN,
    ):
        raise RuntimeError(
            "Final PNG count mismatch: train={} test={} teachers={}".format(
                observed_train, observed_test, observed_teachers
            )
        )

    completion = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "contract": contract,
        "canonical_mnist": {
            "train_images_sha256": hash_array(train_images),
            "train_labels_sha256": hash_array(train_labels),
            "test_images_sha256": hash_array(test_images),
            "test_labels_sha256": hash_array(test_labels),
        },
        "corner_assignments_file": str(corners_path),
        "corner_assignments_sha256": sha256_file(corners_path),
        "sample_manifest_file": str(sample_manifest_path),
        "sample_manifest_sha256": sha256_file(sample_manifest_path),
        "generated_image_tree_sha256": image_digest.hexdigest(),
        "surrogate_teacher_tree_sha256": teacher_digest.hexdigest(),
        "counts": {
            "train_images": observed_train,
            "test_images": observed_test,
            "teacher_maps": observed_teachers,
        },
        "train_summary": train_summary,
        "test_summary": test_summary,
    }
    completion_path = metadata_root / "completion_manifest.json"
    atomic_write_json(completion_path, completion)
    print("[AUDIT] counts and deterministic pixel digests passed")
    print("[DONE] completion_manifest={}".format(completion_path))

    if args.archive_path is not None:
        archive_path = args.archive_path.expanduser().resolve()
        archive = make_archive(output_root, archive_path)
        atomic_write_json(
            archive_path.with_name(archive_path.name + ".manifest.json"), archive
        )
        print("[DONE] archive={}".format(archive["path"]))
        print("[DONE] archive_sha256={}".format(archive["sha256"]))


if __name__ == "__main__":
    main()
