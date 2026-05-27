import unittest

from wafer_defect_studio.training_protocol import TrainingConfig, TrainingProtocolError
from wafer_defect_studio.training_run import RunConfig, TrainingRunError


class SpatialMilV5TrainingPolicyTest(unittest.TestCase):
    def test_protocol_records_only_truthful_effective_batch_variants(self) -> None:
        for physical_batch, accumulation in ((4, 1), (2, 2)):
            config = TrainingConfig(
                snapshot_id="snapshot",
                split_id="split",
                class_count=2,
                epochs=30,
                batch_size=physical_batch,
                learning_rate=0.0003,
                patch_size=128,
                patch_stride=64,
                training_policy="spatial_mil_v5",
            )
            self.assertEqual(config.to_dict()["training_policy"], "spatial_mil_v5")
            self.assertEqual(4 // physical_batch, accumulation)
        with self.assertRaisesRegex(TrainingProtocolError, "physical batch_size"):
            TrainingConfig("snapshot", "split", 2, 30, 1, patch_size=128,
                           patch_stride=64, training_policy="spatial_mil_v5")

    def test_run_config_preserves_v5_without_enabling_v4_refinement(self) -> None:
        config = RunConfig(
            "snapshot", "split", class_count=2, epochs=30, batch_size=4,
            learning_rate=0.0003, patch_size=128, patch_stride=64,
            training_policy="spatial_mil_v5",
        )
        self.assertEqual(config.model_config["training_policy"], "spatial_mil_v5")
        self.assertEqual(config.optimization_config["learning_rate"], 0.0003)
        with self.assertRaisesRegex(TrainingRunError, "hard-negative refinement"):
            RunConfig(
                "snapshot", "split", class_count=2, epochs=30, batch_size=4,
                learning_rate=0.0003, patch_size=128, patch_stride=64,
                training_policy="spatial_mil_v5", priority_normal_bag_ids=("normal",),
                hard_negative_selection_sha256="0" * 64,
            )

    def test_protocol_can_explicitly_retain_epoch_states_for_spatial_replay(self) -> None:
        config = TrainingConfig(
            "snapshot", "split", 2, 30, 4, learning_rate=0.0003,
            patch_size=128, patch_stride=64, training_policy="spatial_mil_v5",
            retain_epoch_states=True,
        )
        restored = TrainingConfig.from_json(config.to_json())
        self.assertTrue(restored.retain_epoch_states)
        self.assertTrue(restored.to_dict()["retain_epoch_states"])


if __name__ == "__main__":
    unittest.main()
