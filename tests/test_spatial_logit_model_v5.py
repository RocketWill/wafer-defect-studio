import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from wafer_defect_studio.model_registry import (
    create_resnet18_spatial_logits,
    create_resnet18_spatial_logits_v5,
    load_project_checkpoint,
)
from wafer_defect_studio.training_run import TrainingRunError, validate_project_checkpoint


class SpatialLogitModelV5Test(unittest.TestCase):
    def test_preserves_stride_two_small_defect_logits_and_v4_behavior(self) -> None:
        model = create_resnet18_spatial_logits_v5(2, device="cpu")
        output = model(torch.randn(1, 1, 128, 128))

        self.assertEqual(model.architecture, "resnet18_spatial_logits_v5")
        self.assertEqual(model.feature_stride, 2)
        self.assertEqual(tuple(output.shape), (1, 2, 64, 64))
        output.sum().backward()
        self.assertIsNotNone(model.backbone.conv1.weight.grad)
        self.assertIsNotNone(model.backbone.layer4[-1].conv2.weight.grad)
        self.assertIsNotNone(model.spatial_head.weight.grad)

        v4 = create_resnet18_spatial_logits(2, device="cpu")
        self.assertEqual(tuple(v4(torch.randn(1, 1, 128, 128)).shape), (1, 2, 32, 32))

    def test_validates_and_strictly_loads_separate_v5_checkpoint(self) -> None:
        source = create_resnet18_spatial_logits_v5(2, device="cpu")
        checkpoint = {
            "checkpoint_format": "wafer_defect_studio.resnet18.v5",
            "architecture": "resnet18_spatial_logits_v5",
            "class_count": 2,
            "class_codes": ["scratch", "particle"],
            "normalization_bounds": [
                {
                    "dtype": "uint8",
                    "source_min": 0,
                    "source_max": 255,
                    "low": 0.0,
                    "high": 255.0,
                    "low_percentile": 1.0,
                    "high_percentile": 99.0,
                }
            ],
            "input_size": {"width": 128, "height": 128},
            "feature_stride": 2,
            "patch_size": 128,
            "patch_stride": 64,
            "training_policy": "spatial_mil_v5",
            "loss_policy": {
                "positive_pooling": "normalized_logsumexp",
                "negative_dense_hardest_fraction": 0.01,
                "sparse_probability_budget": 0.01,
                "sparse_loss_weight": 0.25,
                "overlap_loss_weight": 0.10,
            },
            "optimizer_policy": {"name": "adamw", "learning_rate": 0.0003,
                                 "weight_decay": 0.0001, "gradient_clip_norm": 5.0},
            "batch_policy": {"physical_batch_size": 4,
                             "gradient_accumulation_steps": 1,
                             "effective_batch_size": 4},
            "checkpoint_selection": {"source": "validation",
                                     "metric": "v5_validation_loss",
                                     "selected_epoch": 7, "value": 0.25},
            "epochs": 30,
            "state_dict": source.state_dict(),
        }
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "v5.pt"
            torch.save(checkpoint, path)
            validated = validate_project_checkpoint(path, expected_class_codes=("scratch", "particle"))
            loaded, loaded_checkpoint = load_project_checkpoint(path, device="cpu")
            self.assertEqual(validated["feature_stride"], 2)
            self.assertEqual(loaded.architecture, "resnet18_spatial_logits_v5")
            self.assertEqual(loaded_checkpoint["checkpoint_format"], checkpoint["checkpoint_format"])
            self.assertTrue(
                all(
                    torch.equal(value, loaded.state_dict()[name])
                    for name, value in source.state_dict().items()
                )
            )

            checkpoint["feature_stride"] = 4
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(TrainingRunError, "feature_stride must be 2"):
                validate_project_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
