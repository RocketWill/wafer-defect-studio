import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs" / "demo"))

from run_phase2_demo import build_realistic_corpus, select_family_isolated_split_seed


class RealisticDemoCorpusTest(unittest.TestCase):
    def test_corpus_has_distinct_deterministic_sources_and_matching_variant_labels(self):
        source = QImage(1536, 1536, QImage.Format.Format_Grayscale8)
        source.fill(128)
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = build_realistic_corpus(Path(first_dir), source)
            second = build_realistic_corpus(Path(second_dir), source)

            self.assertEqual(len(first), 10)
            first_fingerprints = tuple(
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path, _annotations, _family, _split in first
            )
            second_fingerprints = tuple(
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path, _annotations, _family, _split in second
            )
            self.assertEqual(len(set(first_fingerprints)), 10)
            self.assertEqual(first_fingerprints, second_fingerprints)
            train_paths = [path for path, _annotations, _family, split in first if split == "train"]
            self.assertEqual(len(train_paths), 8)
            self.assertTrue(all("train-base-" in path.name for path in train_paths))

            first_annotations = tuple(
                (family, split, tuple(sorted(annotations.items())))
                for _path, annotations, family, split in first
            )
            second_annotations = tuple(
                (family, split, tuple(sorted(annotations.items())))
                for _path, annotations, family, split in second
            )
            self.assertEqual(first_annotations, second_annotations)
            self.assertGreater(len(set(first_annotations)), 1)
            self.assertTrue(all(annotations for annotations in first_annotations))

            families_by_split = {}
            for _path, annotations, family, split in first:
                families_by_split.setdefault(split, set()).add(family)
                for class_codes in annotations.values():
                    self.assertIsInstance(class_codes, tuple)
                    self.assertTrue(set(class_codes) <= {"scratch", "particle"})
            self.assertEqual(
                families_by_split,
                {"train": {"train-a"}, "validation": {"validation-b"}, "test": {"test-c"}},
            )

            hard_negatives = [
                (annotations, split)
                for _path, annotations, _family, split in first
                if all("scratch" not in codes for codes in annotations.values())
            ]
            self.assertTrue(hard_negatives)
            self.assertTrue(all(split == "train" for _annotations, split in hard_negatives))
            for held_out_split in ("validation", "test"):
                held_out_annotations = next(
                    annotations
                    for _path, annotations, _family, split in first
                    if split == held_out_split
                )
                self.assertTrue(any("scratch" in codes for codes in held_out_annotations.values()))
                self.assertTrue(any("particle" in codes for codes in held_out_annotations.values()))

            first_family_fingerprints = {}
            for path, _annotations, family, _split in first:
                first_family_fingerprints.setdefault(
                    family, hashlib.sha256(path.read_bytes()).hexdigest()
                )
            self.assertEqual(len(first_family_fingerprints), 3)
            self.assertEqual(len(set(first_family_fingerprints.values())), 3)

            scratch_geometry = {}
            for path, annotations, family, _split in first:
                image = QImage(str(path))
                values = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(
                    image.height(), image.bytesPerLine()
                )[:, : image.width()]
                scratch_cells = [
                    cell for cell, codes in annotations.items() if "scratch" in codes
                ]
                if family not in scratch_geometry and scratch_cells:
                    row, column = scratch_cells[-1]
                    cell = values[row * 512 : (row + 1) * 512, column * 512 : (column + 1) * 512]
                    ys, xs = np.where(cell < 60)
                    scratch_geometry[family] = (
                        int(xs.mean()),
                        int(ys.mean()),
                        int(xs.max() - xs.min()),
                        int(ys.max() - ys.min()),
                        int(np.cov(xs, ys)[0, 1] > 0),
                    )
            self.assertEqual(set(scratch_geometry), {"train-a", "validation-b", "test-c"})
            self.assertEqual(len(set(scratch_geometry.values())), 3)

            train_scratch_geometry = []
            for path, annotations, _family, split in first:
                if split != "train" or all(
                    "scratch" not in codes for codes in annotations.values()
                ):
                    continue
                image = QImage(str(path))
                values = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(
                    image.height(), image.bytesPerLine()
                )[:, : image.width()]
                row, column = next(
                    cell for cell, codes in annotations.items() if "scratch" in codes
                )
                cell = values[row * 512 : (row + 1) * 512, column * 512 : (column + 1) * 512]
                ys, xs = np.where(cell < 60)
                train_scratch_geometry.append(
                    (
                        row,
                        column,
                        int(xs.mean()),
                        int(ys.mean()),
                        int(xs.max() - xs.min()),
                        int(ys.max() - ys.min()),
                        int(np.cov(xs, ys)[0, 1] > 0),
                    )
                )
            self.assertGreaterEqual(len(set(train_scratch_geometry)), 6)
            self.assertEqual({geometry[-1] for geometry in train_scratch_geometry}, {0, 1})

    def test_split_seed_keeps_every_base_family_in_its_declared_split(self):
        image_ids_by_family = {
            "train-a": tuple(f"train-{index}" for index in range(8)),
            "validation-b": ("validation-0",),
            "test-c": ("test-0",),
        }
        expected = {
            "train-a": "train",
            "validation-b": "validation",
            "test-c": "test",
        }

        seed = select_family_isolated_split_seed(image_ids_by_family, expected)
        ranked = sorted(
            (image_id, family)
            for family, image_ids in image_ids_by_family.items()
            for image_id in image_ids
        )
        ranked.sort(key=lambda item: hashlib.sha256(f"{seed}\0{item[0]}".encode()).digest())
        actual = {}
        for index, (_image_id, family) in enumerate(ranked):
            split = "train" if index < 8 else "validation" if index == 8 else "test"
            actual.setdefault(family, set()).add(split)
        self.assertEqual(actual, {family: {split} for family, split in expected.items()})


if __name__ == "__main__":
    unittest.main()
