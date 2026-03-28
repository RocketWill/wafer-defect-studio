"""Project-bound Dataset Snapshot creation orchestration."""

from __future__ import annotations

from pathlib import Path

from .dataset_diagnostics import DatasetPreview, preview_project_dataset
from .dataset_snapshot import (
    DatasetSnapshotError,
    SamplingPolicy,
    create_dataset_snapshot,
)
from .image_asset import SourceHealth, load_image_assets
from .normalization import NativePixelSample, compute_percentile_bounds
from .training_scope import (
    TrainingScope,
    eligible_image_ids,
    load_image_data_group_assignments,
    save_training_scope,
)
from .wafer_view import _decode_wafer_image


def create_project_dataset_snapshot(
    project_path: str | Path,
    data_group_ids: tuple[str, ...],
    class_codes: tuple[str, ...],
) -> tuple[str, DatasetPreview]:
    """Persist the selected Training Scope and create its immutable Snapshot."""

    if not data_group_ids or not class_codes:
        raise DatasetSnapshotError("Select at least one Data Group and Defect Class.")
    save_training_scope(
        project_path,
        TrainingScope(tuple(data_group_ids), tuple(class_codes)),
    )
    assignments = tuple(
        item
        for item in load_image_data_group_assignments(project_path)
        if item.data_group_id in set(data_group_ids)
    )
    selected_ids = tuple(sorted(item.image_asset_id for item in assignments))
    if selected_ids != tuple(sorted(eligible_image_ids(project_path))):
        raise DatasetSnapshotError(
            "Every selected image must be reviewed with an unchanged source."
        )

    assets = {
        item.asset.image_asset_id: item
        for item in load_image_assets(project_path)
    }
    samples = []
    for image_id in selected_ids:
        reopened = assets.get(image_id)
        if reopened is None or reopened.source_health is not SourceHealth.AVAILABLE:
            raise DatasetSnapshotError(f"Unavailable Wafer Image source: {image_id}")
        loaded, _display_image = _decode_wafer_image(reopened.asset.path)
        samples.append(NativePixelSample(loaded.dtype, loaded.pixels))

    bounds = compute_percentile_bounds(samples, 1.0, 99.0)
    snapshot = create_dataset_snapshot(
        project_path,
        bounds,
        SamplingPolicy(1.0),
    )
    return (
        snapshot.snapshot_id,
        preview_project_dataset(project_path, tuple(data_group_ids), tuple(class_codes)),
    )
