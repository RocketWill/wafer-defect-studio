import hashlib
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from wafer_defect_studio.dataset_snapshot import load_dataset_snapshot
from wafer_defect_studio.dataset_split import load_dataset_split
from wafer_defect_studio.main_window import MainWindow


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs" / "demo"))

from run_phase2_demo import (
    REALISTIC_CORPUS_MANIFEST,
    _create_dataset,
    _prepare_annotation,
    _seed_project,
    build_realistic_corpus,
    select_family_isolated_split_seed,
    validate_realistic_corpus_annotations,
)


class RealisticDemoCorpusTest(unittest.TestCase):
    def test_frozen_manifest_renders_oracle_truth_and_rejects_class_drift(self):
        source = QImage(1536, 1536, QImage.Format.Format_Grayscale8)
        source.fill(128)
        with tempfile.TemporaryDirectory() as temporary:
            corpus = build_realistic_corpus(Path(temporary), source)

        self.assertEqual(len(REALISTIC_CORPUS_MANIFEST), 20)
        self.assertEqual(
            tuple((case.family, case.split) for case in REALISTIC_CORPUS_MANIFEST),
            (("train-a", "train"),) * 16
            + (("validation-b", "validation"),) * 2
            + (("test-c", "test"),) * 2,
        )
        for case, (path, annotations, family, split) in zip(
            REALISTIC_CORPUS_MANIFEST, corpus, strict=True
        ):
            self.assertEqual((path.name, family, split), (case.filename, case.family, case.split))
            self.assertTrue(annotations)
            validate_realistic_corpus_annotations(case, annotations)

        case = REALISTIC_CORPUS_MANIFEST[0]
        annotations = corpus[0][1]
        grid = next(iter(annotations))
        expected = annotations[grid]
        variants = {
            "missing": {**annotations, grid: expected[:-1]},
            "extra": {**annotations, grid: (*expected, "contamination")},
            "tampered": {**annotations, grid: ("scratch",)},
        }
        for drift, actual in variants.items():
            with self.subTest(drift=drift), self.assertRaisesRegex(
                ValueError,
                re.escape(
                    f"filename={case.filename} grid={grid!r} "
                    f"expected={expected!r} actual={actual[grid]!r}"
                ),
            ):
                validate_realistic_corpus_annotations(case, actual)

    def test_registered_realistic_project_freezes_the_declared_16_2_2_split(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.png"
            source = QImage(1536, 1536, QImage.Format.Format_Grayscale8)
            source.fill(128)
            self.assertTrue(source.save(str(source_path), "PNG"))
            (
                project_path,
                asset,
                _source_path,
                profile,
                source_kind,
                _original_size,
                corpus,
            ) = _seed_project(root, app, source_path)
            window = MainWindow(
                settings=QSettings(str(root / "test.ini"), QSettings.Format.IniFormat)
            )
            try:
                _prepare_annotation(
                    window,
                    project_path,
                    asset,
                    profile,
                    app,
                    source_kind=source_kind,
                    realistic_corpus=corpus,
                )
                snapshot_id, split_id = _create_dataset(
                    project_path, realistic_corpus=corpus
                )
            finally:
                window.close()
                window.deleteLater()
                app.processEvents()

            snapshot = load_dataset_snapshot(project_path, snapshot_id)
            split = load_dataset_split(project_path, split_id)
            self.assertEqual(
                tuple(map(len, (split.train_image_ids, split.validation_image_ids, split.test_image_ids))),
                (16, 2, 2),
            )
            self.assertEqual(len(snapshot.sources), 20)
            for held_out_ids in (split.validation_image_ids, split.test_image_ids):
                support = {
                    code: sum(
                        any(
                            sample.image_asset_id == image_id
                            and code in sample.class_codes
                            for sample in snapshot.samples
                        )
                        for image_id in held_out_ids
                    )
                    for code in ("scratch", "particle")
                }
                self.assertEqual(support, {"scratch": 2, "particle": 2})

    def test_corpus_has_distinct_deterministic_sources_and_matching_variant_labels(self):
        source = QImage(1536, 1536, QImage.Format.Format_Grayscale8)
        source.fill(128)
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = build_realistic_corpus(Path(first_dir), source)
            second = build_realistic_corpus(Path(second_dir), source)

            self.assertEqual(len(first), 20)
            first_fingerprints = tuple(
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path, _annotations, _family, _split in first
            )
            second_fingerprints = tuple(
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path, _annotations, _family, _split in second
            )
            self.assertEqual(len(set(first_fingerprints)), 20)
            self.assertEqual(first_fingerprints, second_fingerprints)
            train_paths = [path for path, _annotations, _family, split in first if split == "train"]
            self.assertEqual(len(train_paths), 16)
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
                held_out_annotations = [
                    annotations
                    for _path, annotations, _family, split in first
                    if split == held_out_split
                ]
                self.assertEqual(len(held_out_annotations), 2)
                self.assertTrue(all(
                    any("scratch" in codes for codes in annotations.values())
                    and any("particle" in codes for codes in annotations.values())
                    for annotations in held_out_annotations
                ))

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
            "train-a": tuple(f"train-{index}" for index in range(16)),
            "validation-b": ("validation-0", "validation-1"),
            "test-c": ("test-0", "test-1"),
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
            split = "train" if index < 16 else "validation" if index < 18 else "test"
            actual.setdefault(family, set()).add(split)
        self.assertEqual(actual, {family: {split} for family, split in expected.items()})


if __name__ == "__main__":
    unittest.main()
