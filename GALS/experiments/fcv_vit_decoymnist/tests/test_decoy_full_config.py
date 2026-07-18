from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
SRC = EXPERIMENT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decoy_full_config import (  # noqa: E402
    ConfigError,
    candidate_epochs,
    enumerate_runs,
    load_and_validate_config,
    validate_config,
)


class DecoyFullConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_path = (
            EXPERIMENT_ROOT
            / "configs"
            / "decoymnist_vit_s16_fcv_full_online.yaml"
        )

    def test_locked_config_enumerates_1080_online_candidates(self) -> None:
        config = load_and_validate_config(self.config_path)
        runs = enumerate_runs(config)
        epochs = candidate_epochs(config)
        self.assertEqual(len(runs), 108)
        self.assertEqual(epochs, list(range(1, 11)))
        self.assertEqual(len(runs) * len(epochs), 1080)
        self.assertEqual(len({run.run_id for run in runs}), 108)
        self.assertEqual(runs[0].run_index, 0)
        self.assertEqual(runs[-1].run_index, 107)
        self.assertEqual(runs[0].candidate_id(1).split("_epoch_")[-1], "001")

    def test_no_training_state_is_persisted(self) -> None:
        config = load_and_validate_config(self.config_path)
        pool = config["candidate_pool"]
        storage = config["storage"]
        self.assertFalse(pool["persist_model_checkpoints"])
        self.assertFalse(pool["persist_optimizer_states"])
        self.assertFalse(pool["persist_resume_states"])
        self.assertFalse(pool["retain_selector_winners"])
        self.assertFalse(storage["persist_token_banks"])
        self.assertTrue(storage["delete_ephemeral_token_banks_after_each_epoch"])

    def test_rejects_changed_grid_or_harmonic_rule(self) -> None:
        config = load_and_validate_config(self.config_path)
        config.pop("_provenance")
        changed_grid = copy.deepcopy(config)
        changed_grid["training"]["crop_scale_mins"][-1] = 0.2
        with self.assertRaisesRegex(ConfigError, "crop_scale_mins"):
            validate_config(changed_grid)
        changed_selector = copy.deepcopy(config)
        changed_selector["fcv"]["primary_selector"]["epsilon"] = 1.0e-6
        with self.assertRaisesRegex(ConfigError, "harmonic"):
            validate_config(changed_selector)


if __name__ == "__main__":
    unittest.main()

