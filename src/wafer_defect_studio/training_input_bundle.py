"""Materialize immutable values needed by the training worker."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .dataset_snapshot import DatasetSnapshot, SnapshotSample, load_dataset_snapshot
from .dataset_split import DatasetSplit, load_dataset_split
from .detection_windows import Rect
from .image_asset import SourceHealth, load_image_assets
from .normalization import NormalizationBounds
from .project import ProjectError, open_project
from .training_dataset import enumerate_model_patch_rects
from .training_protocol import TrainingConfig


class TrainingInputBundleError(ProjectError):
    """Raised when a Snapshot/Split cannot produce worker input values."""


@dataclass(frozen=True)
class TrainingBundleSource:
    image_asset_id: str
    split: str
    path: str
    fingerprint: str
    dtype: str


@dataclass(frozen=True)
class TrainingPatchBag:
    bag_id: str
    image_asset_id: str
    row: int
    column: int
    patches: tuple[Rect, ...]
    class_codes: tuple[str, ...]


@dataclass(frozen=True)
class TrainingInputBundle:
    snapshot_id: str
    split_id: str
    class_codes: tuple[str, ...]
    normalization_bounds: tuple[NormalizationBounds, ...]
    sources: tuple[TrainingBundleSource, ...]
    samples: tuple[SnapshotSample, ...]
    version: int = 1
    patch_bags: tuple[TrainingPatchBag, ...] = ()

    def to_json(self) -> str:
        value = {
            "snapshot_id": self.snapshot_id,
            "split_id": self.split_id,
            "class_codes": self.class_codes,
            "normalization_bounds": tuple(asdict(item) for item in self.normalization_bounds),
            "sources": tuple(asdict(item) for item in self.sources),
        }
        if self.version == 1:
            value["samples"] = tuple(asdict(item) for item in self.samples)
        elif self.version == 2:
            value["version"] = 2
            value["patch_bags"] = tuple(
                {
                    "bag_id": bag.bag_id,
                    "image_asset_id": bag.image_asset_id,
                    "row": bag.row,
                    "column": bag.column,
                    "patches": tuple(
                        {
                            "x": rect.x,
                            "y": rect.y,
                            "width": rect.width,
                            "height": rect.height,
                        }
                        for rect in bag.patches
                    ),
                    "class_codes": bag.class_codes,
                }
                for bag in self.patch_bags
            )
        else:
            raise TrainingInputBundleError(f"Unsupported Training Input Bundle version: {self.version}")
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, serialized: str) -> "TrainingInputBundle":
        try:
            value = json.loads(serialized)
            version = value.get("version", 1)
            samples = ()
            patch_bags = ()
            if version == 1:
                samples = tuple(
                    SnapshotSample(
                        item["image_asset_id"],
                        item["row"],
                        item["column"],
                        item["x"],
                        item["y"],
                        item["width"],
                        item["height"],
                        tuple(item["class_codes"]),
                    )
                    for item in value["samples"]
                )
            elif version == 2:
                patch_bags = tuple(
                    TrainingPatchBag(
                        item["bag_id"],
                        item["image_asset_id"],
                        item["row"],
                        item["column"],
                        tuple(
                            Rect(rect["x"], rect["y"], rect["width"], rect["height"])
                            for rect in item["patches"]
                        ),
                        tuple(item["class_codes"]),
                    )
                    for item in value["patch_bags"]
                )
            else:
                raise TrainingInputBundleError(
                    f"Unsupported Training Input Bundle version: {version}"
                )
            return cls(
                value["snapshot_id"],
                value["split_id"],
                tuple(value["class_codes"]),
                tuple(NormalizationBounds(**item) for item in value["normalization_bounds"]),
                tuple(TrainingBundleSource(**item) for item in value["sources"]),
                samples,
                version,
                patch_bags,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise TrainingInputBundleError("Invalid Training Input Bundle") from error


def create_training_input_bundle(
    project_path: str | Path,
    snapshot_id: str,
    split_id: str,
    destination: str | Path,
    *,
    config: TrainingConfig | None = None,
) -> TrainingInputBundle:
    """Write one deterministic worker bundle after validating source identity."""

    info = open_project(project_path)
    snapshot = load_dataset_snapshot(info.path, snapshot_id)
    split = load_dataset_split(info.path, split_id)
    if split.snapshot_id != snapshot.snapshot_id:
        raise TrainingInputBundleError(
            f"Dataset Split {split_id} does not belong to Dataset Snapshot {snapshot_id}"
        )
    if not snapshot.samples:
        raise TrainingInputBundleError(
            f"Dataset Snapshot {snapshot_id} has no frozen training samples"
        )
    if config is not None and (
        config.snapshot_id != snapshot_id or config.split_id != split_id
    ):
        raise TrainingInputBundleError(
            "Training Config does not match the requested Dataset Snapshot/Split"
        )

    split_by_image = _split_assignments(split)
    snapshot_sources = {source.image_asset_id: source for source in snapshot.sources}
    assets = {item.asset.image_asset_id: item for item in load_image_assets(info.path)}
    sources: list[TrainingBundleSource] = []
    for image_id in sorted(snapshot_sources):
        split_name = split_by_image.get(image_id)
        if split_name is None:
            raise TrainingInputBundleError(
                f"Dataset Split {split_id} omits Snapshot image: {image_id}"
            )
        source = snapshot_sources[image_id]
        reopened = assets.get(image_id)
        if reopened is None or reopened.source_health is not SourceHealth.AVAILABLE:
            health = "missing" if reopened is None else reopened.source_health.value
            raise TrainingInputBundleError(
                f"Snapshot source is {health}: {image_id}"
            )
        if reopened.asset.fingerprint != source.fingerprint:
            raise TrainingInputBundleError(
                f"Snapshot source fingerprint changed: {image_id}"
            )
        sources.append(
            TrainingBundleSource(
                image_id,
                split_name,
                str(reopened.asset.path),
                source.fingerprint,
                reopened.asset.dtype,
            )
        )

    source_ids = {source.image_asset_id for source in sources}
    samples = tuple(
        sample for sample in snapshot.samples if sample.image_asset_id in source_ids
    )
    if len(samples) != len(snapshot.samples):
        raise TrainingInputBundleError("Snapshot sample source inventory is incomplete")
    class_codes = tuple(item.code for item in snapshot.classes)
    if config is None:
        bundle = TrainingInputBundle(
            snapshot.snapshot_id,
            split.split_id,
            class_codes,
            snapshot.normalization_bounds,
            tuple(sources),
            samples,
        )
    else:
        patch_bags = tuple(
            TrainingPatchBag(
                f"{sample.image_asset_id}:{sample.row}:{sample.column}",
                sample.image_asset_id,
                sample.row,
                sample.column,
                enumerate_model_patch_rects(
                    Rect(sample.x, sample.y, sample.width, sample.height),
                    config,
                ),
                sample.class_codes,
            )
            for sample in samples
        )
        bundle = TrainingInputBundle(
            snapshot.snapshot_id,
            split.split_id,
            class_codes,
            snapshot.normalization_bounds,
            tuple(sources),
            (),
            2,
            patch_bags,
        )
    output = Path(destination).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(bundle.to_json() + "\n", encoding="utf-8")
    return bundle


def _split_assignments(split: DatasetSplit) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for name, image_ids in (
        ("train", split.train_image_ids),
        ("validation", split.validation_image_ids),
        ("test", split.test_image_ids),
    ):
        for image_id in image_ids:
            if image_id in assignments:
                raise TrainingInputBundleError(
                    f"Dataset Split repeats image: {image_id}"
                )
            assignments[image_id] = name
    return assignments


__all__ = [
    "TrainingBundleSource",
    "TrainingInputBundle",
    "TrainingInputBundleError",
    "TrainingPatchBag",
    "create_training_input_bundle",
]
