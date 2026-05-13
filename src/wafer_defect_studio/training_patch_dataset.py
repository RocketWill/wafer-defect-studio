"""Value-only native patch samples for a Training Input Bundle."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .training_augmentation import (
    AugmentationConfig,
    apply_augmentation,
    apply_spatial_mil_v4_augmentation,
)
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


class ClassAwareEqualShapeBatchSampler(Sampler[list[int]]):
    """Cycle class/normal groups while keeping each Patch Bag batch one shape."""

    def __init__(
        self,
        shape_keys: Sequence[tuple[int, ...]],
        target_keys: Sequence[Sequence[int]],
        *,
        batch_size: int,
        seed: int,
        epoch_size: int | None = None,
        priority_normal_indices: Sequence[int] = (),
    ) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise TrainingPatchDatasetError("batch_size must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TrainingPatchDatasetError("seed must be an integer")
        if len(shape_keys) != len(target_keys) or not shape_keys:
            raise TrainingPatchDatasetError("shape_keys and target_keys must have equal non-zero length")
        class_count = len(target_keys[0])
        if any(len(target) != class_count for target in target_keys):
            raise TrainingPatchDatasetError("target_keys must have equal length")
        groups = [
            [index for index, target in enumerate(target_keys) if target[class_index]]
            for class_index in range(class_count)
        ]
        for class_index, group in enumerate(groups):
            if not group:
                raise TrainingPatchDatasetError(
                    f"class {class_index} has no positive samples"
                )
        normal = [index for index, target in enumerate(target_keys) if not any(target)]
        if normal:
            groups.append(normal)
        priority_normal_indices = tuple(priority_normal_indices)
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(target_keys)
            for index in priority_normal_indices
        ):
            raise TrainingPatchDatasetError("priority_normal_indices contains an out of range index")
        if len(set(priority_normal_indices)) != len(priority_normal_indices):
            raise TrainingPatchDatasetError("priority_normal_indices must be unique")
        if any(any(target_keys[index]) for index in priority_normal_indices):
            raise TrainingPatchDatasetError("priority_normal_indices must select normal samples")
        if priority_normal_indices:
            groups.append(list(priority_normal_indices))
        if epoch_size is None:
            epoch_size = len(shape_keys)
        if isinstance(epoch_size, bool) or not isinstance(epoch_size, int) or epoch_size < 1:
            raise TrainingPatchDatasetError("epoch_size must be a positive integer")
        self._shape_keys = tuple(shape_keys)
        self._groups = tuple(tuple(group) for group in groups)
        self._batch_size = batch_size
        self._seed = seed
        self._epoch_size = epoch_size
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise TrainingPatchDatasetError("epoch must be an integer")
        self._epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._seed + self._epoch)
        group_order = torch.randperm(len(self._groups), generator=generator).tolist()
        shape_orders: list[list[tuple[int, ...]]] = []
        pools: list[dict[tuple[int, ...], list[int]]] = []
        for group in self._groups:
            by_shape: dict[tuple[int, ...], list[int]] = {}
            for index in group:
                by_shape.setdefault(self._shape_keys[index], []).append(index)
            shapes = list(by_shape)
            shape_order = torch.randperm(len(shapes), generator=generator).tolist()
            shape_orders.append([shapes[index] for index in shape_order])
            shuffled: dict[tuple[int, ...], list[int]] = {}
            for shape, indices in by_shape.items():
                order = torch.randperm(len(indices), generator=generator).tolist()
                shuffled[shape] = [indices[index] for index in order]
            pools.append(shuffled)

        emitted = 0
        group_cursor = 0
        shape_cursors = [0] * len(self._groups)
        pool_cursors: list[dict[tuple[int, ...], int]] = [dict() for _ in self._groups]
        while emitted < self._epoch_size:
            group_index = group_order[group_cursor % len(group_order)]
            group_cursor += 1
            shapes = shape_orders[group_index]
            shape = shapes[shape_cursors[group_index] % len(shapes)]
            shape_cursors[group_index] += 1
            pool = pools[group_index][shape]
            count = min(self._batch_size, self._epoch_size - emitted)
            cursor = pool_cursors[group_index].get(shape, 0)
            batch = [pool[(cursor + offset) % len(pool)] for offset in range(count)]
            pool_cursors[group_index][shape] = cursor + count
            emitted += count
            yield batch

    def __len__(self) -> int:
        return (self._epoch_size + self._batch_size - 1) // self._batch_size


PatchDatasetItem = (
    tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
)


class TrainingPatchDataset(Dataset[PatchDatasetItem]):
    """Read one deterministic split from a value-only bundle."""

    def __init__(
        self,
        bundle: TrainingInputBundle,
        split: str,
        *,
        augmentation: AugmentationConfig | None = None,
        include_patch_rects: bool = False,
        spatial_mil_v4_seed: int | None = None,
    ) -> None:
        if not isinstance(bundle, TrainingInputBundle):
            raise TrainingPatchDatasetError("bundle must be a TrainingInputBundle")
        if split not in {"train", "validation", "test"}:
            raise TrainingPatchDatasetError("split must be train, validation, or test")
        self._bundle = bundle
        self._split = split
        if augmentation is not None and not isinstance(augmentation, AugmentationConfig):
            raise TrainingPatchDatasetError("augmentation must be an AugmentationConfig value")
        if (
            spatial_mil_v4_seed is not None
            and (isinstance(spatial_mil_v4_seed, bool) or not isinstance(spatial_mil_v4_seed, int))
        ):
            raise TrainingPatchDatasetError("spatial_mil_v4_seed must be an integer")
        self._augmentation = augmentation
        self._include_patch_rects = include_patch_rects
        self._spatial_mil_v4_seed = spatial_mil_v4_seed
        self._epoch = 0
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

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise TrainingPatchDatasetError("epoch must be an integer")
        self._epoch = epoch

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

    @property
    def bag_ids(self) -> tuple[str, ...] | None:
        if self._bundle.version != 2:
            return None
        return tuple(bag.bag_id for bag in self._items)

    @property
    def bag_target_keys(self) -> tuple[tuple[int, ...], ...] | None:
        if self._bundle.version != 2:
            return None
        return tuple(
            tuple(int(code in bag.class_codes) for code in self._bundle.class_codes)
            for bag in self._items
        )

    def __getitem__(self, index: int) -> PatchDatasetItem:
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
            if self._split == "train" and self._spatial_mil_v4_seed is not None:
                effective_seed = self._spatial_mil_v4_seed + self._epoch * 1_000_003 + index
                patches = tuple(
                    apply_spatial_mil_v4_augmentation(patch, effective_seed=effective_seed)
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
        if self._include_patch_rects and self._bundle.version == 2:
            rects = torch.tensor(
                [(rect.x, rect.y, rect.width, rect.height) for rect in item.patches],
                dtype=torch.int64,
            )
            return output, target, rects
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
    "ClassAwareEqualShapeBatchSampler",
    "EqualShapeBatchSampler",
    "TrainingPatchDataset",
    "TrainingPatchDatasetError",
]
