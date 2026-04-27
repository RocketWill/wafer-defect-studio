import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from PySide6.QtGui import QImage

from wafer_defect_studio.dataset_snapshot import SnapshotSample
from wafer_defect_studio.detection_windows import Rect
from wafer_defect_studio.model_registry import create_resnet18
from wafer_defect_studio.model_scoring import score_training_bundle
from wafer_defect_studio.normalization import NormalizationBounds
from wafer_defect_studio.training_input_bundle import (
    TrainingBundleSource,
    TrainingInputBundle,
    TrainingPatchBag,
)


class ModelScoringTest(unittest.TestCase):
    def test_v3_scores_one_row_per_grid_after_max_pooling_raw_patch_logits(self):
        class CraftedPatchScorer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))
                self.calls = 0

            def forward(self, inputs):
                outputs = (
                    torch.tensor(((2.0, -1.0), (-2.0, 3.0))),
                    torch.tensor(((-4.0, 1.0), (4.0, -2.0))),
                )
                output = outputs[self.calls]
                self.calls += 1
                return output

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.png"
            image = QImage(64, 32, QImage.Format.Format_Grayscale8)
            bits = image.bits()
            stride = image.bytesPerLine()
            for row in range(32):
                bits[row * stride : (row + 1) * stride] = bytes(range(64))
            self.assertTrue(image.save(str(source), "PNG"))
            del bits, image
            bundle = TrainingInputBundle(
                "snapshot-v3",
                "split-v3",
                ("scratch", "particle"),
                (NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 1.0, 99.0),),
                (
                    TrainingBundleSource("validation-image", "validation", str(source), _hash(source), "uint8"),
                    TrainingBundleSource("test-image", "test", str(source), _hash(source), "uint8"),
                ),
                (),
                2,
                (
                    TrainingPatchBag("validation-0", "validation-image", 0, 0, (Rect(0, 0, 32, 32), Rect(32, 0, 32, 32)), ("scratch",)),
                    TrainingPatchBag("validation-1", "validation-image", 0, 1, (Rect(0, 0, 32, 32), Rect(32, 0, 32, 32)), ("particle",)),
                    TrainingPatchBag("test-0", "test-image", 0, 0, (Rect(0, 0, 32, 32),), ("scratch", "particle")),
                ),
            )
            bundle_path = root / "bundle.json"
            bundle_path.write_text(bundle.to_json(), encoding="utf-8")
            model = CraftedPatchScorer()
            checkpoint = {
                "checkpoint_format": "wafer_defect_studio.resnet18.v3",
                "class_codes": ["scratch", "particle"],
                "input_size": {"width": 32, "height": 32},
                "patch_size": 32,
                "patch_stride": 32,
                "bag_pooling": "max",
            }

            with patch(
                "wafer_defect_studio.model_scoring.load_project_checkpoint",
                return_value=(model, checkpoint),
            ):
                scores = score_training_bundle(
                    bundle_path,
                    root / "model.pt",
                    split="validation",
                )

            np.testing.assert_array_equal(scores.y_true, ((1.0, 0.0), (0.0, 1.0)))
            np.testing.assert_allclose(
                scores.y_score,
                torch.sigmoid(torch.tensor(((2.0, 3.0), (4.0, 1.0)))).numpy(),
            )
            self.assertEqual(scores.class_codes, ("scratch", "particle"))
            self.assertEqual(model.calls, 2)

    def test_scores_requested_split_in_bundle_order_with_sigmoid(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.png"
            image = QImage(32, 32, QImage.Format.Format_Grayscale8)
            bits = image.bits()
            stride = image.bytesPerLine()
            for row in range(32):
                bits[row * stride : (row + 1) * stride] = bytes([row] * 32) + bytes(stride - 32)
            self.assertTrue(image.save(str(source), "PNG"))
            del bits, image
            model = create_resnet18(2, device="cpu")
            for parameter in model.parameters():
                parameter.data.zero_()
            checkpoint_path = root / "model.pt"
            torch.save(
                {
                    "checkpoint_format": "wafer_defect_studio.resnet18.v1",
                    "architecture": "resnet18",
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
                    "input_size": {"width": 32, "height": 32},
                    "state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
            bundle = TrainingInputBundle(
                "snapshot-1",
                "split-1",
                ("scratch", "particle"),
                (NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 1.0, 99.0),),
                (
                    TrainingBundleSource("validation-image", "validation", str(source), _hash(source), "uint8"),
                    TrainingBundleSource("test-image", "test", str(source), _hash(source), "uint8"),
                ),
                (
                    SnapshotSample("validation-image", 0, 0, 0, 0, 32, 32, ("particle",)),
                    SnapshotSample("test-image", 0, 1, 0, 0, 32, 32, ("scratch",)),
                ),
            )
            bundle_path = root / "bundle.json"
            bundle_path.write_text(bundle.to_json(), encoding="utf-8")
            scores = score_training_bundle(bundle_path, checkpoint_path, split="validation")
            np.testing.assert_array_equal(scores.y_true, [[0.0, 1.0]])
            np.testing.assert_allclose(scores.y_score, [[0.5, 0.5]])
            self.assertEqual(scores.class_codes, ("scratch", "particle"))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
