import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PySide6.QtGui import QImage

from wafer_defect_studio.dataset_snapshot import SnapshotSample
from wafer_defect_studio.model_registry import create_resnet18
from wafer_defect_studio.model_scoring import score_training_bundle
from wafer_defect_studio.normalization import NormalizationBounds
from wafer_defect_studio.training_input_bundle import TrainingBundleSource, TrainingInputBundle


class ModelScoringTest(unittest.TestCase):
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
