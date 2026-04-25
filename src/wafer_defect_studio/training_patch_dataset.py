"""Value-only native patch samples for a Training Input Bundle."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .training_augmentation import AugmentationConfig, apply_augmentation
from .training_dataset import extract_model_patch
from .training_input_bundle import TrainingInputBundle, TrainingBundleSource
from .wafer_view import _decode_wafer_image


class TrainingPatchDatasetError(ValueError):
    """Raised when a bundled source cannot produce a model patch."""


class EqualShapeBatchSampler(Sampler[list[int]]):
    """Yield deterministic batches whose Patch Bag tensor shapes are identical."""

    def __init__(
        self,
        shape_keys: Sequence[tuple[int, ...]],
        *,
        batch_size: int,
        seed: int,
        shuffle: bool,
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise TrainingPatchDatasetError("batch_size must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TrainingPatchDatasetError("seed must be an integer")
        if not isinstance(shuffle, bool):
            raise TrainingPatchDatasetError("shuffle must be boolean")
        self._shape_keys = tuple(shape_keys)
        self._batch_size = batch_size
        self._seed = seed
        self._shuffle = shuffle

    def __iter__(self) -> Iterator[list[int]]:
        if self._shuffle:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self._seed)
            order = torch.randperm(len(self._shape_keys), generator=generator).tolist()
        else:
            order = list(range(len(self._shape_keys)))
        pending: dict[tuple[int, ...], list[int]] = {}
        for index in order:
            bucket = pending.setdefault(self._shape_keys[index], [])
            bucket.append(index)
            if len(bucket) == self._batch_size:
                yield bucket
                pending[self._shape_keys[index]] = []
        for bucket in pending.values():
            if bucket:
                yield bucket

    def __len__(self) -> int:
        counts = Counter(self._shape_keys)
        return sum(
            (count + self._batch_size - 1) // self._batch_size
            for count in counts.values()
        )


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
        if bundle.version == 2:
            self._items = tuple(
                bag
                for bag in bundle.patch_bags
                if bag.image_asset_id in self._sources
            )
        else:
            self._items = tuple(
                sample
                for sample in bundle.samples
                if sample.image_asset_id in self._sources
            )
        if not self._items:
            raise TrainingPatchDatasetError(f"bundle has no {split} samples")
        self._pixels: dict[str, np.ndarray] = {}
        self._bounds = {item.dtype: item for item in bundle.normalization_bounds}

    def __len__(self) -> int:
        return len(self._items)

    @property
    def bag_shape_keys(self) -> tuple[tuple[int, ...], ...] | None:
        """Return v2 tensor shapes from descriptors without reading source pixels."""

        if self._bundle.version != 2:
            return None
        return tuple(
            (
                len(bag.patches),
                3,
                bag.patches[0].height,
                bag.patches[0].width,
            )
            for bag in self._items
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self._items[index]
        source = self._sources[item.image_asset_id]
        pixels = self._load_pixels(source)
        bounds = self._bounds.get(source.dtype)
        if bounds is None:
            raise TrainingPatchDatasetError(
                f"normalization bounds are missing for {source.dtype}"
            )
        if self._bundle.version == 2:
            patches = tuple(
                extract_model_patch(
                    pixels,
                    bounds,
                    top=rect.y,
                    left=rect.x,
                    size=(rect.width, rect.height),
                )
                for rect in item.patches
            )
            if self._split == "train" and self._augmentation is not None:
                patches = tuple(
                    apply_augmentation(patch, self._augmentation)
                    for patch in patches
                )
            output = torch.stack(patches)
        else:
            output = extract_model_patch(
                pixels,
                bounds,
                top=item.y,
                left=item.x,
                size=(item.width, item.height),
            )
            if self._split == "train" and self._augmentation is not None:
                output = apply_augmentation(output, self._augmentation)
        target = torch.tensor(
            [int(code in item.class_codes) for code in self._bundle.class_codes],
            dtype=torch.float32,
        )
        return output, target

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


__all__ = [
    "EqualShapeBatchSampler",
    "TrainingPatchDataset",
    "TrainingPatchDatasetError",
]
