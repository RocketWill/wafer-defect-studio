import hashlib
import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from wafer_defect_studio import project
from wafer_defect_studio.dataset_snapshot import (
    DatasetSnapshot,
    SamplingPolicy,
    SnapshotSample,
    SnapshotSource,
)
from wafer_defect_studio.dataset_split import DatasetSplit
from wafer_defect_studio.defect_class import DefectClass
from wafer_defect_studio.normalization import NormalizationBounds
from wafer_defect_studio.training_input_bundle import (
    TrainingInputBundle,
    TrainingInputBundleError,
    create_training_input_bundle,
)
from wafer_defect_studio.training_protocol import TrainingConfig
from wafer_defect_studio.training_scope import TrainingScope


class TrainingInputBundleTest(unittest.TestCase):
    def test_bundle_round_trip_is_deterministic_and_keeps_split_and_samples(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = root / "project"
            project.create_project(project_path)
            source = root / "wafer.bin"
            source.write_bytes(b"source")
            _seed_project(project_path, source)
            snapshot = DatasetSnapshot(
                "snapshot-1",
                "2026-01-01T00:00:00+00:00",
                TrainingScope(("line-a",), ("scratch", "particle")),
                (
                    DefectClass("scratch", "Scratch", "#cc4444"),
                    DefectClass("particle", "Particle", "#4488cc"),
                ),
                (SnapshotSource("wafer-1", "line-a", _fingerprint(source)),),
                (),
                (),
                (NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 1.0, 99.0),),
                SamplingPolicy(1.0),
                (
                    SnapshotSample("wafer-1", 0, 0, 10, 20, 2, 2, ("scratch", "particle")),
                    SnapshotSample("wafer-1", 0, 1, 12, 20, 2, 2, ()),
                ),
            )
            split = DatasetSplit("split-1", "snapshot-1", 7, ("wafer-1",), (), ())
            _insert_snapshot_and_split(project_path, snapshot, split)

            destination = root / "bundle.json"
            created = create_training_input_bundle(
                project_path,
                "snapshot-1",
                "split-1",
                destination,
                config=TrainingConfig(
                    snapshot_id="snapshot-1",
                    split_id="split-1",
                    class_count=2,
                    epochs=1,
                    batch_size=1,
                    patch_size=1,
                    patch_stride=1,
                ),
            )
            self.assertEqual(TrainingInputBundle.from_json(destination.read_text()), created)
            self.assertEqual(created.class_codes, ("scratch", "particle"))
            self.assertEqual(created.sources[0].split, "train")
            self.assertEqual(created.version, 2)
            self.assertEqual(created.samples, ())
            self.assertEqual(
                tuple(bag.class_codes for bag in created.patch_bags),
                (("scratch", "particle"), ()),
            )
            self.assertEqual(created.patch_bags[0].bag_id, "wafer-1:0:0")
            self.assertEqual(created.patch_bags[1].bag_id, "wafer-1:0:1")
            self.assertEqual(
                tuple(created.patch_bags[0].patches),
                ((10, 20, 1, 1), (11, 20, 1, 1), (10, 21, 1, 1), (11, 21, 1, 1)),
            )
            payload = json.loads(destination.read_text())
            self.assertNotIn("samples", payload)
            self.assertNotIn("class_codes", payload["patch_bags"][0]["patches"][0])
            self.assertTrue(
                {"split", "fingerprint", "path"}.isdisjoint(payload["patch_bags"][0])
            )
            self.assertEqual(payload["sources"][0]["split"], "train")
            self.assertEqual(payload["sources"][0]["fingerprint"], _fingerprint(source))

    def test_bundle_rejects_legacy_snapshot_without_frozen_samples(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_path = root / "project"
            project.create_project(project_path)
            source = root / "wafer.bin"
            source.write_bytes(b"source")
            _seed_project(project_path, source)
            snapshot = DatasetSnapshot(
                "snapshot-1",
                "2026-01-01T00:00:00+00:00",
                TrainingScope(("line-a",), ("scratch",)),
                (DefectClass("scratch", "Scratch", "#cc4444"),),
                (SnapshotSource("wafer-1", "line-a", _fingerprint(source)),),
                (),
                (),
                (NormalizationBounds("uint8", 0, 255, 0.0, 255.0, 1.0, 99.0),),
                SamplingPolicy(1.0),
            )
            _insert_snapshot_and_split(
                project_path,
                snapshot,
                DatasetSplit("split-1", "snapshot-1", 7, ("wafer-1",), (), ()),
            )
            with self.assertRaisesRegex(TrainingInputBundleError, "no frozen training samples"):
                create_training_input_bundle(
                    project_path, "snapshot-1", "split-1", root / "bundle.json"
                )


def _seed_project(project_path: Path, source: Path) -> None:
    connection = sqlite3.connect(project_path / "project.sqlite")
    try:
        connection.execute(
            "INSERT INTO image_assets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("wafer-1", str(source.resolve()), 2, 2, "uint8", "TIFF", _fingerprint(source), 0),
        )
        connection.execute(
            project._DEFECT_CLASSES_TABLE_SQL
        )
        connection.execute(
            project._GRID_ANNOTATIONS_TABLE_SQL
        )
        connection.execute(
            project._IMAGE_REVIEWS_TABLE_SQL
        )
        connection.execute(
            project._DATA_GROUPS_TABLE_SQL
        )
        connection.execute(
            project._IMAGE_DATA_GROUPS_TABLE_SQL
        )
        connection.execute(
            project._TRAINING_SCOPE_TABLE_SQL
        )
        connection.execute(project._DATASET_SNAPSHOTS_TABLE_SQL)
        connection.execute(project._DATASET_SPLITS_TABLE_SQL)
        connection.execute("INSERT INTO image_reviews VALUES ('wafer-1', 1)")
        connection.execute("INSERT INTO effective_wafer_areas VALUES ('wafer-1', 'ellipse', '{\"center_x\":1,\"center_y\":1,\"radius_x\":1,\"radius_y\":1}', 1)")
        connection.execute("INSERT INTO grid_profiles VALUES ('grid', 1, 2, 2)")
        connection.execute("INSERT INTO image_grid_placements VALUES ('wafer-1', 'grid', 1, 0, 0)")
        connection.execute("INSERT INTO defect_classes VALUES ('scratch', 'Scratch', '#cc4444', '', '', 0, 1)")
        connection.execute("INSERT INTO data_groups VALUES ('line-a', 'Line A', 0)")
        connection.execute("INSERT INTO image_data_groups VALUES ('wafer-1', 'line-a')")
        connection.execute("INSERT INTO training_scope VALUES (1, '[\"line-a\"]', '[\"scratch\"]')")
        connection.execute("UPDATE project_metadata SET schema_version = 12")
        connection.execute("PRAGMA user_version = 12")
        connection.commit()
    finally:
        connection.close()


def _insert_snapshot_and_split(
    project_path: Path, snapshot: DatasetSnapshot, split: DatasetSplit
) -> None:
    connection = sqlite3.connect(project_path / "project.sqlite")
    try:
        connection.execute(
            "INSERT INTO dataset_snapshots VALUES (?, ?, ?)",
            (snapshot.snapshot_id, snapshot.created_at, _snapshot_json(snapshot)),
        )
        connection.execute(
            "INSERT INTO dataset_splits VALUES (?, ?, ?, ?)",
            (split.split_id, split.snapshot_id, split.seed, _split_json(split)),
        )
        connection.commit()
    finally:
        connection.close()


def _snapshot_json(snapshot: DatasetSnapshot) -> str:
    from dataclasses import asdict

    return json.dumps(asdict(snapshot), sort_keys=True, separators=(",", ":"))


def _split_json(split: DatasetSplit) -> str:
    from dataclasses import asdict

    return json.dumps(asdict(split), sort_keys=True, separators=(",", ":"))


def _fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
