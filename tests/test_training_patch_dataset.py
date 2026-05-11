import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PySide6.QtGui import QImage

from wafer_defect_studio.dataset_snapshot import SnapshotSample
from wafer_defect_studio.detection_windows import Rect
from wafer_defect_studio.normalization import NormalizationBounds
from wafer_defect_studio.training_augmentation import AugmentationConfig
from wafer_defect_studio.training_input_bundle import (
    TrainingBundleSource,
    TrainingInputBundle,
    TrainingPatchBag,
)
from wafer_defect_studio.training_patch_dataset import TrainingPatchDataset


class TrainingPatchDatasetTest(unittest.TestCase):
    def test_v4_bag_affine_is_overlap_consistent_and_epoch_seeded(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.png"
            image = QImage(3, 1, QImage.Format.Format_Grayscale8)
            bits = image.bits()
            bits[:image.bytesPerLine()] = bytes((20, 100, 220)) + bytes(image.bytesPerLine() - 3)
            self.assertTrue(image.save(str(path), "PNG"))
            del bits, image
            bundle = TrainingInputBundle(
                "snapshot", "split", ("scratch",),
                (NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 1.0, 99.0),),
                (TrainingBundleSource("wafer", "train", str(path), _hash(path), "uint8"),),
                (), 2,
                (
                    TrainingPatchBag("bag-0", "wafer", 0, 0, (Rect(0, 0, 2, 1), Rect(1, 0, 2, 1)), ("scratch",)),
                    TrainingPatchBag("bag-1", "wafer", 0, 0, (Rect(0, 0, 2, 1), Rect(1, 0, 2, 1)), ("scratch",)),
                ),
            )
            dataset = TrainingPatchDataset(bundle, "train", spatial_mil_v4_seed=11)
            first = dataset[0][0]
            self.assertTrue(torch.equal(first[0, :, 0, 1], first[1, :, 0, 0]))
            self.assertTrue(torch.equal(first, dataset[0][0]))
            self.assertFalse(torch.equal(first, dataset[1][0]))
            dataset.set_epoch(1)
            self.assertFalse(torch.equal(first, dataset[0][0]))
            validation_bundle = TrainingInputBundle(
                "snapshot", "split", ("scratch",), bundle.normalization_bounds,
                (TrainingBundleSource("wafer", "validation", str(path), _hash(path), "uint8"),),
                (), 2,
                (TrainingPatchBag("bag", "wafer", 0, 0, (Rect(0, 0, 2, 1), Rect(1, 0, 2, 1)), ("scratch",)),),
            )
            augmented_validation = TrainingPatchDataset(
                validation_bundle, "validation", spatial_mil_v4_seed=11
            )[0][0]
            plain_validation = TrainingPatchDataset(validation_bundle, "validation")[0][0]
            self.assertTrue(torch.equal(augmented_validation, plain_validation))

    def test_v4_view_exposes_target_keys_and_patch_rects(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.png"
            image = QImage(2, 1, QImage.Format.Format_Grayscale8)
            self.assertTrue(image.save(str(path), "PNG"))
            bundle = TrainingInputBundle(
                "snapshot", "split", ("scratch", "particle"),
                (NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 1.0, 99.0),),
                (TrainingBundleSource("wafer", "train", str(path), _hash(path), "uint8"),),
                (), 2,
                (TrainingPatchBag("bag", "wafer", 0, 0, (Rect(0, 0, 1, 1), Rect(1, 0, 1, 1)), ("particle",)),),
            )
            legacy = TrainingPatchDataset(bundle, "train")
            spatial = TrainingPatchDataset(bundle, "train", include_patch_rects=True)

            self.assertEqual(spatial.bag_target_keys, ((0, 1),))
            self.assertEqual(len(legacy[0]), 2)
            inputs, target, rects = spatial[0]
            self.assertEqual(tuple(inputs.shape), (2, 3, 1, 1))
            torch.testing.assert_close(target, torch.tensor([0.0, 1.0]))
            torch.testing.assert_close(rects, torch.tensor([[0, 0, 1, 1], [1, 0, 1, 1]]))

    def test_v2_bag_returns_ordered_patch_stack_and_train_only_augmentation(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.png"
            image = QImage(256, 128, QImage.Format.Format_Grayscale8)
            bits = image.bits()
            stride = image.bytesPerLine()
            row = bytes(range(256)) + bytes(stride - 256)
            for y in range(128):
                bits[y * stride : (y + 1) * stride] = row
            self.assertTrue(image.save(str(path), "PNG"))
            del bits, image
            sources = tuple(
                TrainingBundleSource(image_id, split, str(path), _hash(path), "uint8")
                for image_id, split in (
                    ("train-image", "train"),
                    ("validation-image", "validation"),
                )
            )
            patches = (Rect(0, 0, 128, 128), Rect(128, 0, 128, 128))
            bundle = TrainingInputBundle(
                "snapshot-1",
                "split-1",
                ("scratch", "particle"),
                (NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 1.0, 99.0),),
                sources,
                (),
                2,
                (
                    TrainingPatchBag(
                        "train-image:0:0",
                        "train-image",
                        0,
                        0,
                        patches,
                        ("scratch", "particle"),
                    ),
                    TrainingPatchBag(
                        "validation-image:0:0",
                        "validation-image",
                        0,
                        0,
                        patches,
                        ("particle",),
                    ),
                ),
            )
            config = AugmentationConfig(horizontal_flip=True)

            train_patches, train_target = TrainingPatchDataset(
                bundle, "train", augmentation=config
            )[0]
            validation_patches, validation_target = TrainingPatchDataset(
                bundle, "validation", augmentation=config
            )[0]

            self.assertEqual(train_patches.shape, (2, 3, 128, 128))
            self.assertEqual(validation_patches.shape, (2, 3, 128, 128))
            self.assertTrue(torch.equal(train_patches[:, 0], train_patches[:, 1]))
            self.assertAlmostEqual(float(train_patches[0, 0, 0, 0]), 127 / 255)
            self.assertAlmostEqual(float(train_patches[1, 0, 0, 0]), 1.0)
            self.assertAlmostEqual(float(validation_patches[0, 0, 0, 0]), 0.0)
            self.assertAlmostEqual(float(validation_patches[1, 0, 0, 0]), 128 / 255)
            self.assertTrue(torch.equal(train_target, torch.tensor([1.0, 1.0])))
            self.assertTrue(torch.equal(validation_target, torch.tensor([0.0, 1.0])))

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
