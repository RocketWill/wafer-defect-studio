"""Pure Dataset Snapshot preview distributions and warnings."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class PreviewImage:
    image_asset_id: str
    data_group_id: str
    class_codes: tuple[str, ...]
    normal_count: int
    source_valid: bool = True


@dataclass(frozen=True)
class DatasetWarning:
    code: str
    message: str


@dataclass(frozen=True)
class NormalSampling:
    normal_to_positive_ratio: float
    explicit_positive_count: int
    available_normal_count: int
    sampled_normal_count: int


@dataclass(frozen=True)
class DatasetPreview:
    image_distribution: tuple[tuple[str, int], ...]
    class_distribution: tuple[tuple[str, int], ...]
    group_distribution: tuple[tuple[str, int], ...]
    warnings: tuple[DatasetWarning, ...]
    sampling: NormalSampling


def preview_dataset(
    selected_class_codes: tuple[str, ...],
    images: tuple[PreviewImage, ...],
    normal_to_positive_ratio: float = 1.0,
) -> DatasetPreview:
    """Preview eligible inputs without changing explicit positive annotations."""

    if (
        not selected_class_codes
        or len(set(selected_class_codes)) != len(selected_class_codes)
        or any(not isinstance(code, str) or not code for code in selected_class_codes)
    ):
        raise ValueError("selected_class_codes must contain unique non-empty strings")
    if not isinstance(images, tuple) or not all(isinstance(image, PreviewImage) for image in images):
        raise ValueError("images must be a tuple of PreviewImage values")
    if any(
        not image.image_asset_id
        or not image.data_group_id
        or not isinstance(image.class_codes, tuple)
        or any(not isinstance(code, str) or not code for code in image.class_codes)
        or isinstance(image.normal_count, bool)
        or not isinstance(image.normal_count, int)
        or image.normal_count < 0
        or not isinstance(image.source_valid, bool)
        for image in images
    ):
        raise ValueError("images contain invalid preview facts")
    if (
        isinstance(normal_to_positive_ratio, bool)
        or not isinstance(normal_to_positive_ratio, (int, float))
        or not math.isfinite(normal_to_positive_ratio)
        or normal_to_positive_ratio < 0
    ):
        raise ValueError("normal_to_positive_ratio must be finite and non-negative")

    eligible = tuple(image for image in images if image.source_valid)
    selected = set(selected_class_codes)
    class_counts = Counter(
        code for image in eligible for code in image.class_codes if code in selected
    )
    group_counts = Counter(image.data_group_id for image in eligible)
    positive_count = sum(class_counts.values())
    available_normal_count = sum(image.normal_count for image in eligible)
    sampled_normal_count = min(
        available_normal_count, round(positive_count * normal_to_positive_ratio)
    )

    missing = tuple(code for code in selected_class_codes if class_counts[code] == 0)
    nonzero_counts = tuple(class_counts[code] for code in selected_class_codes if class_counts[code])
    warnings: list[DatasetWarning] = []
    if missing:
        warnings.append(
            DatasetWarning(
                "missing-positive-class",
                "Add reviewed examples for: " + ", ".join(missing),
            )
        )
    if len(eligible) < 10:
        warnings.append(
            DatasetWarning(
                "limited-validation",
                "Select at least 10 eligible images to populate validation and test splits.",
            )
        )
    invalid_count = len(images) - len(eligible)
    if invalid_count:
        warnings.append(
            DatasetWarning(
                "invalid-source",
                f"Repair or re-import {invalid_count} invalid source image(s).",
            )
        )
    if len(nonzero_counts) > 1 and max(nonzero_counts) > 2 * min(nonzero_counts):
        warnings.append(
            DatasetWarning(
                "imbalance",
                "Review class distribution; the largest class exceeds 2x the smallest.",
            )
        )

    return DatasetPreview(
        (("eligible", len(eligible)), ("invalid", invalid_count)),
        tuple((code, class_counts[code]) for code in selected_class_codes),
        tuple(sorted(group_counts.items())),
        tuple(warnings),
        NormalSampling(
            float(normal_to_positive_ratio),
            positive_count,
            available_normal_count,
            sampled_normal_count,
        ),
    )
