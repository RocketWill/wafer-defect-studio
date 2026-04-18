import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PySide6.QtGui import QImage

from wafer_defect_studio.dataset_snapshot import SnapshotSample
from wafer_defect_studio.normalization import NormalizationBounds
from wafer_defect_studio.training_augmentation import AugmentationConfig
from wafer_defect_studio.training_input_bundle import (
    TrainingBundleSource,
    TrainingInputBundle,
)
from wafer_defect_studio.training_patch_dataset import TrainingPatchDataset


class TrainingPatchDatasetTest(unittest.TestCase):
    def test_reads_native_source_and_builds_ordered_multilabel_target(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.png"
            image = QImage(3, 2, QImage.Format.Format_Grayscale8)
            bits = image.bits()
            stride = image.bytesPerLine()
            bits[0 : stride] = bytes((10, 20, 0)) + bytes(stride - 3)
            bits[stride : 2 * stride] = bytes((30, 40, 0)) + bytes(stride - 3)
            self.assertTrue(image.save(str(path), "PNG"))
            del bits, image
            bundle = TrainingInputBundle(
                "snapshot-1",
                "split-1",
                ("scratch", "particle"),
                (NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 1.0, 99.0),),
                (TrainingBundleSource("wafer-1", "train", str(path), _hash(path), "uint8"),),
                (SnapshotSample("wafer-1", 0, 0, 0, 0, 2, 2, ("particle",)),),
            )
            patch, target = TrainingPatchDataset(bundle, "train")[0]
            self.assertEqual(patch.shape, (3, 2, 2))
            self.assertTrue(torch.equal(patch[0], patch[1]))
            self.assertTrue(torch.allclose(patch[0], torch.tensor([[10, 20], [30, 40]]) / 255.0))
            self.assertTrue(torch.equal(target, torch.tensor([0.0, 1.0])))

    def test_target_uses_class_order_and_native_normalization(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.npy"
            path.write_bytes(b"not an image")
            bundle = TrainingInputBundle(
                "snapshot-1",
                "split-1",
                ("scratch", "particle"),
                (NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 1.0, 99.0),),
                (TrainingBundleSource("wafer-1", "train", str(path), _hash(path), "uint8"),),
                (SnapshotSample("wafer-1", 0, 0, 0, 0, 2, 2, ("particle",)),),
            )
            dataset = TrainingPatchDataset(bundle, "train")
            self.assertEqual(len(dataset), 1)
            with self.assertRaisesRegex(ValueError, "unable to decode source"):
                dataset[0]

    def test_augmentation_is_train_only_and_keeps_target(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.png"
            image = QImage(2, 1, QImage.Format.Format_Grayscale8)
            bits = image.bits()
            stride = image.bytesPerLine()
            bits[0 : stride] = bytes((0, 255)) + bytes(stride - 2)
            self.assertTrue(image.save(str(path), "PNG"))
            del bits, image
            sources = tuple(
                TrainingBundleSource(image_id, split, str(path), _hash(path), "uint8")
                for image_id, split in (("train-image", "train"), ("validation-image", "validation"))
            )
            bundle = TrainingInputBundle(
                "snapshot-1",
                "split-1",
                ("scratch",),
                (NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 1.0, 99.0),),
                sources,
                (
                    SnapshotSample("train-image", 0, 0, 0, 0, 2, 1, ("scratch",)),
                    SnapshotSample("validation-image", 0, 0, 0, 0, 2, 1, ("scratch",)),
                ),
            )
            config = AugmentationConfig(horizontal_flip=True)
            train_patch, train_target = TrainingPatchDataset(bundle, "train", augmentation=config)[0]
            validation_patch, validation_target = TrainingPatchDataset(bundle, "validation", augmentation=config)[0]
            self.assertTrue(torch.equal(train_patch[0], torch.tensor([[1.0, 0.0]])))
            self.assertTrue(torch.equal(validation_patch[0], torch.tensor([[0.0, 1.0]])))
            self.assertTrue(torch.equal(train_target, validation_target))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
