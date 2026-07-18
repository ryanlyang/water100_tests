from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decoy_online_schema import (  # noqa: E402
    OnlineSchemaError,
    atomic_write_namespace_rows,
    authorized_namespaces,
    load_selector_namespace,
    namespace_columns,
    namespace_output_path,
    require_selector_access,
    validate_namespace_row,
)


def _row(namespace: str, epoch: int = 1):
    row = {column: 0.0 for column in namespace_columns(namespace)}
    row.update(
        {
            "run_index": 0,
            "run_id": "run_000",
            "candidate_id": f"run_000_epoch_{epoch:03d}",
            "epoch": epoch,
            "seed": 0,
            "learning_rate": 1.0e-5,
            "weight_decay": 0.01,
            "crop_scale_min": 1.0,
        }
    )
    if namespace == "controls":
        for column in row:
            if column.endswith("_status"):
                row[column] = "complete"
        row["control_warning_reason_counts_json"] = "{}"
        row["control_diagnostics_warning_only"] = True
    return row


class OnlineSchemaTest(unittest.TestCase):
    def test_selector_access_is_leakage_separated(self) -> None:
        self.assertEqual(authorized_namespaces("vanilla"), ("biased_validation",))
        self.assertEqual(authorized_namespaces("fcv"), ("biased_validation", "fcv"))
        self.assertEqual(authorized_namespaces("oracle"), ("oracle_analysis_only",))
        self.assertEqual(authorized_namespaces("posthoc"), ("test_analysis_only",))
        with self.assertRaisesRegex(OnlineSchemaError, "not authorized"):
            require_selector_access("fcv", "test_analysis_only")
        with self.assertRaisesRegex(OnlineSchemaError, "not authorized"):
            require_selector_access("vanilla", "oracle_analysis_only")

    def test_unprivileged_rows_reject_test_or_oracle_values(self) -> None:
        row = _row("biased_validation")
        validate_namespace_row("biased_validation", row)
        row["test_accuracy"] = 1.0
        with self.assertRaises(OnlineSchemaError):
            validate_namespace_row("biased_validation", row)

    def test_atomic_prefix_and_selector_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = namespace_output_path(root, "fcv", "run_000")
            atomic_write_namespace_rows("fcv", [_row("fcv", 1)], path)
            atomic_write_namespace_rows(
                "fcv", [_row("fcv", 1), _row("fcv", 2)], path
            )
            loaded = load_selector_namespace(root, "fcv", "fcv", "run_000")
            self.assertEqual(len(loaded), 2)
            with self.assertRaises(OnlineSchemaError):
                load_selector_namespace(root, "vanilla", "fcv", "run_000")


if __name__ == "__main__":
    unittest.main()
