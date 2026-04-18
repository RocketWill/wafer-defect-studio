"""Value-only native patch samples for a Training Input Bundle."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .training_augmentation import AugmentationConfig, apply_augmentation
from .training_dataset import extract_model_patch
from .training_input_bundle import TrainingInputBundle, TrainingBundleSource
from .wafer_view import _decode_wafer_image


class TrainingPatchDatasetError(ValueError):
    """Raised when a bundled source cannot produce a model patch."""


class TrainingPatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Read one deterministic split from a value-only bundle."""

    def __init__(
        self,
        bundle: TrainingInputBundle,
        split: str,
        *,
        augmentation: AugmentationConfig | None = None,
    ) -> None:
        if not isinstance(bundle, TrainingInputBundle):
            raise TrainingPatchDatasetError("bundle must be a TrainingInputBundle")
        if split not in {"train", "validation", "test"}:
            raise TrainingPatchDatasetError("split must be train, validation, or test")
        self._bundle = bundle
        self._split = split
        if augmentation is not None and not isinstance(augmentation, AugmentationConfig):
            raise TrainingPatchDatasetError("augmentation must be an AugmentationConfig value")
        self._augmentation = augmentation
        self._sources = {
            source.image_asset_id: source
            for source in bundle.sources
            if source.split == split
        }
        self._samples = tuple(
            sample
            for sample in bundle.samples
            if sample.image_asset_id in self._sources
        )
        if not self._samples:
            raise TrainingPatchDatasetError(f"bundle has no {split} samples")
        self._pixels: dict[str, np.ndarray] = {}
        self._bounds = {item.dtype: item for item in bundle.normalization_bounds}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self._samples[index]
        source = self._sources[sample.image_asset_id]
        pixels = self._load_pixels(source)
        bounds = self._bounds.get(source.dtype)
        if bounds is None:
            raise TrainingPatchDatasetError(
                f"normalization bounds are missing for {source.dtype}"
            )
        patch = extract_model_patch(
            pixels,
            bounds,
            top=sample.y,
            left=sample.x,
            size=(sample.width, sample.height),
        )
        if self._split == "train" and self._augmentation is not None:
            patch = apply_augmentation(patch, self._augmentation)
        target = torch.tensor(
            [int(code in sample.class_codes) for code in self._bundle.class_codes],
            dtype=torch.float32,
        )
        return patch, target

    def _load_pixels(self, source: TrainingBundleSource) -> np.ndarray:
        cached = self._pixels.get(source.image_asset_id)
        if cached is not None:
            return cached
        path = Path(source.path).expanduser().resolve()
        if not path.is_file():
            raise TrainingPatchDatasetError(f"source is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != source.fingerprint:
            raise TrainingPatchDatasetError(
                f"source fingerprint changed: {source.image_asset_id}"
            )
        try:
            loaded, _ = _decode_wafer_image(path)
        except Exception as error:
            raise TrainingPatchDatasetError(
                f"unable to decode source {source.image_asset_id}: {path}"
            ) from error
        if loaded.dtype != source.dtype:
            raise TrainingPatchDatasetError(
                f"source dtype changed: {source.image_asset_id}"
            )
        dtype = np.uint16 if loaded.dtype == "uint16" else np.uint8
        pixels = np.asarray(loaded.pixels, dtype=dtype).reshape(loaded.height, loaded.width)
        self._pixels[source.image_asset_id] = pixels
        return pixels


__all__ = ["TrainingPatchDataset", "TrainingPatchDatasetError"]
