"""Spawn-safe Snapshot-backed ResNet18 training worker.

The active project path reads the immutable input bundle and writes only a
staged artifact set; the legacy no-bundle fixture path remains for protocol
regression tests and is not selected by the GUI.  The worker owns no
project-service objects and never opens SQLite.  It accepts the value-only
:mod:`training_protocol` request, reports versioned messages over a
multiprocessing queue, and leaves the service a checksum manifest in the
requested staging directory when training completes.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .detection_windows import Rect
from .model_registry import (
    create_resnet18,
    create_resnet18_spatial_logits,
    create_resnet18_spatial_logits_v5,
    max_pool_patch_logits,
    resolve_device,
)
from .spatial_mil import (
    absent_class_hard_negative_loss,
    dense_absent_class_loss,
    derive_train_positive_class_weights,
    overlap_consistency_loss,
    positive_spatial_lse_loss,
    positive_spatial_mil_loss,
    present_sparse_budget_loss,
)
from .training_input_bundle import TrainingInputBundle
from .training_augmentation import SPATIAL_MIL_V4_AUGMENTATION_POLICY
from .training_patch_dataset import (
    ClassAwareEqualShapeBatchSampler,
    EqualShapeBatchSampler,
    TrainingPatchDataset,
)
from .training_protocol import (
    ProgressMessage,
    TerminalMessage,
    TrainingConfig,
    TrainingRequest,
    decode_message,
    encode_message,
)


class TrainingWorkerError(RuntimeError):
    """Raised when a worker request cannot be started or validated."""


_CHECKPOINT_FORMAT_V2 = "wafer_defect_studio.resnet18.v2"
_CHECKPOINT_FORMAT_V3 = "wafer_defect_studio.resnet18.v3"


def create_training_data_loader(
    dataset: TrainingPatchDataset,
    config: TrainingConfig,
    *,
    split: str,
    priority_normal_indices: tuple[int, ...] = (),
) -> DataLoader:
    """Create the worker loader while keeping v2 Patch Bag shapes separate."""

    shape_keys = dataset.bag_shape_keys
    if shape_keys is not None:
        if config.training_policy in {"spatial_mil_v4", "spatial_mil_v5"} and split == "train":
            target_keys = dataset.bag_target_keys
            if target_keys is None:
                raise RuntimeError("spatial_mil_v4 loader requires v2 Patch Bag targets")
            return DataLoader(
                dataset,
                batch_sampler=ClassAwareEqualShapeBatchSampler(
                    shape_keys,
                    target_keys,
                    batch_size=config.batch_size,
                    seed=config.seed,
                    priority_normal_indices=priority_normal_indices,
                ),
                num_workers=0,
            )
        return DataLoader(
            dataset,
            batch_sampler=EqualShapeBatchSampler(
                shape_keys,
                batch_size=config.batch_size,
                seed=config.seed,
                shuffle=split == "train",
            ),
            num_workers=0,
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=split == "train",
        generator=generator,
        num_workers=0,
    )


@dataclass(slots=True)
class TrainingWorkerHandle:
    """Parent-side controls and output queue for one spawned worker."""

    process: mp.Process
    queue: Any
    cancel_event: Any
    request_id: str

    def cancel(self, reason: str = "user") -> bool:
        """Request cooperative cancellation; return whether it was newly set."""

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string")
        if self.cancel_event.is_set():
            return False
        self.cancel_event.set()
        return True

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def join(self, timeout: float | None = None) -> None:
        self.process.join(timeout)

    @property
    def exitcode(self) -> int | None:
        return self.process.exitcode


def start_training_worker(
    request: TrainingRequest,
    *,
    synthetic_steps: int = 2,
    inject_oom_step: int | None = None,
    step_delay: float = 0.0,
    context: mp.context.BaseContext | None = None,
) -> TrainingWorkerHandle:
    """Start one independent worker using the platform's spawn context.

    ``synthetic_steps``, ``inject_oom_step``, and ``step_delay`` are testable
    worker controls.  They are intentionally not part of ``TrainingConfig``;
    persisted model settings therefore cannot be silently changed by an OOM
    fallback or by a synthetic test fixture.
    """

    if not isinstance(request, TrainingRequest):
        raise TrainingWorkerError("request must be a TrainingRequest value")
    _validate_options(synthetic_steps, inject_oom_step, step_delay)
    worker_context = context or mp.get_context("spawn")
    output_queue = worker_context.Queue()
    cancel_event = worker_context.Event()
    process = worker_context.Process(
        target=run_worker,
        args=(encode_message(request), output_queue, cancel_event),
        kwargs={
            "synthetic_steps": synthetic_steps,
            "inject_oom_step": inject_oom_step,
            "step_delay": step_delay,
        },
        name=f"wafer-training-{request.request_id}",
        daemon=True,
    )
    process.start()
    return TrainingWorkerHandle(process, output_queue, cancel_event, request.request_id)


def run_worker(
    request: TrainingRequest | Mapping[str, Any] | str,
    output_queue: Any,
    cancel_event: Any,
    *,
    synthetic_steps: int = 2,
    inject_oom_step: int | None = None,
    step_delay: float = 0.0,
) -> None:
    """Run a short deterministic multi-label job in a child process.

    This function is top-level by design so ``multiprocessing`` can import it
    under Windows' spawn start method.  A serialized request or its value-only
    payload is accepted to keep the process boundary explicit.
    """

    request_value = _coerce_request(request)
    _validate_options(synthetic_steps, inject_oom_step, step_delay)
    config = request_value.config
    staging = Path(request_value.artifact_staging_path).expanduser().resolve()
    log_lines: list[str] = []

    try:
        staging.mkdir(parents=True, exist_ok=True)
        if _is_cancelled(cancel_event):
            _emit_cancelled(output_queue, request_value, config, log_lines)
            return

        device = resolve_device(config.device)
        log_lines.append(f"device={device}")
        log_lines.append(f"batch_size={config.batch_size}")
        bundle = None
        loader = None
        validation_loader = None
        refinement_loader = None
        if request_value.input_bundle_path is not None:
            bundle_path = Path(request_value.input_bundle_path).expanduser().resolve()
            try:
                bundle = TrainingInputBundle.from_json(bundle_path.read_text(encoding="utf-8"))
            except OSError as error:
                raise RuntimeError(f"unable to read training input bundle: {bundle_path}") from error
            if bundle.snapshot_id != config.snapshot_id or bundle.split_id != config.split_id:
                raise RuntimeError("training input bundle does not match the requested snapshot/split")
            if len(bundle.class_codes) != config.class_count:
                raise RuntimeError("training input bundle class count does not match the request")
            if config.training_policy in {"spatial_mil_v4", "spatial_mil_v5"} and bundle.version != 2:
                raise RuntimeError(
                    f"{config.training_policy} requires bundle version 2; received version {bundle.version}"
                )
            if (
                config.training_policy == "spatial_mil_v4"
                and config.patch_size % 4 != 0
            ):
                raise RuntimeError(
                    f"spatial_mil_v4 patch_size must be divisible by feature stride 4; received {config.patch_size}"
                )
            if config.training_policy == "spatial_mil_v5" and config.patch_size % 2 != 0:
                raise RuntimeError(
                    f"spatial_mil_v5 patch_size must be divisible by feature stride 2; received {config.patch_size}"
                )
            if bundle.version == 1:
                sample_sizes = {(sample.width, sample.height) for sample in bundle.samples}
                if len(sample_sizes) != 1:
                    raise RuntimeError(
                        "training input bundle contains variable sample sizes; "
                        "a fixed model input rectangle is required"
                    )
            else:
                if config.patch_size is None or config.patch_stride is None:
                    raise RuntimeError("Patch Bag training requires configured patch geometry")
                if any(
                    rect.width != config.patch_size or rect.height != config.patch_size
                    for bag in bundle.patch_bags
                    for rect in bag.patches
                ):
                    raise RuntimeError(
                        "Patch Bag descriptors do not match the configured patch_size"
                    )
            dataset = TrainingPatchDataset(
                bundle,
                "train",
                include_patch_rects=config.training_policy in {"spatial_mil_v4", "spatial_mil_v5"},
                spatial_mil_v4_seed=(
                    config.seed if config.training_policy in {"spatial_mil_v4", "spatial_mil_v5"} else None
                ),
            )
            loader = create_training_data_loader(
                dataset,
                config,
                split="train",
            )
            if config.training_policy == "spatial_mil_v5":
                validation_dataset = TrainingPatchDataset(
                    bundle, "validation", include_patch_rects=True
                )
                validation_loader = create_training_data_loader(
                    validation_dataset, config, split="validation"
                )
            if config.priority_normal_bag_ids:
                bag_ids = dataset.bag_ids
                if bag_ids is None:
                    raise RuntimeError("hard-negative refinement requires v2 Patch Bag IDs")
                index_by_id = {bag_id: index for index, bag_id in enumerate(bag_ids)}
                unknown = tuple(
                    bag_id for bag_id in config.priority_normal_bag_ids
                    if bag_id not in index_by_id
                )
                if unknown:
                    raise RuntimeError(
                        "hard-negative refinement bag IDs are absent from the train split: "
                        + ", ".join(unknown)
                    )
                refinement_loader = create_training_data_loader(
                    dataset,
                    config,
                    split="train",
                    priority_normal_indices=tuple(
                        index_by_id[bag_id] for bag_id in config.priority_normal_bag_ids
                    ),
                )
        if config.training_policy in {"spatial_mil_v4", "spatial_mil_v5"} and bundle is None:
            raise RuntimeError(
                f"{config.training_policy} requires a v2 Training Input Bundle"
            )
        model = (
            create_resnet18_spatial_logits_v5(
                config.class_count, weights=config.weights_policy, device=device
            )
            if config.training_policy == "spatial_mil_v5"
            else
            create_resnet18_spatial_logits(
                config.class_count,
                weights=config.weights_policy,
                device=device,
            )
            if config.training_policy == "spatial_mil_v4"
            else create_resnet18(
                config.class_count,
                weights=config.weights_policy,
                device=device,
            )
        )
        if bundle is None:
            _freeze_backbone(model)
            optimizer_parameters = model.backbone.fc.parameters()
        else:
            optimizer_parameters = model.parameters()
        optimizer = (
            torch.optim.AdamW(optimizer_parameters, lr=float(config.learning_rate), weight_decay=0.0001)
            if config.training_policy == "spatial_mil_v5"
            else torch.optim.SGD(optimizer_parameters, lr=float(config.learning_rate))
        )
        if bundle is None:
            # BatchNorm cannot update with a one-item batch.  Keeping the model in
            # eval mode for that explicit configuration preserves the requested
            # batch_size instead of silently replacing it with a larger batch.
            model.train() if config.batch_size > 1 else model.eval()
            synthetic_inputs, synthetic_targets = _synthetic_batch(config, device)
            total_steps = config.epochs * synthetic_steps
        else:
            total_steps = config.epochs * len(loader)
            if refinement_loader is not None:
                total_steps += 5 * len(refinement_loader)
        started = time.monotonic()
        step_number = 0
        positive_class_weights = (
            derive_train_positive_class_weights(bundle).to(device)
            if bundle is not None and config.training_policy in {"spatial_mil_v4", "spatial_mil_v5"}
            else None
        )

        total_epochs = config.epochs + (5 if refinement_loader is not None else 0)
        accumulation_steps = 2 if config.training_policy == "spatial_mil_v5" and config.batch_size == 2 else 1
        best_validation_loss = float("inf")
        best_epoch = None
        best_state_dict = None
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, total_epochs + 1):
            active_loader = (
                refinement_loader
                if refinement_loader is not None and epoch > config.epochs
                else loader
            )
            if active_loader is not None and config.training_policy in {"spatial_mil_v4", "spatial_mil_v5"}:
                active_loader.batch_sampler.set_epoch(epoch - 1)
                active_loader.dataset.set_epoch(epoch - 1)
            batches = (
                active_loader
                if active_loader is not None
                else ((synthetic_inputs, synthetic_targets) for _ in range(synthetic_steps))
            )
            for batch in batches:
                if config.training_policy in {"spatial_mil_v4", "spatial_mil_v5"}:
                    inputs, targets, patch_rect_values = batch
                else:
                    inputs, targets = batch
                step_number += 1
                if _is_cancelled(cancel_event):
                    _emit_cancelled(output_queue, request_value, config, log_lines)
                    return
                if inject_oom_step == step_number:
                    raise torch.cuda.OutOfMemoryError(
                        f"injected out-of-memory failure at step {step_number}"
                    )

                if loader is not None:
                    effective_batch_size = (
                        inputs.shape[0] * inputs.shape[1]
                        if inputs.ndim == 5
                        else inputs.shape[0]
                    )
                    model.train() if effective_batch_size > 1 else model.eval()
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                if config.training_policy in {"spatial_mil_v4", "spatial_mil_v5"}:
                    batch_size, patch_count, channels, height, width = inputs.shape
                    raw_logits = model(
                        inputs.reshape(batch_size * patch_count, channels, height, width)
                    )
                    feature_height, feature_width = raw_logits.shape[-2:]
                    logits = raw_logits.reshape(
                        batch_size,
                        patch_count,
                        config.class_count,
                        feature_height,
                        feature_width,
                    )
                    positive_loss = (
                        positive_spatial_lse_loss(logits, targets, positive_class_weights)
                        if config.training_policy == "spatial_mil_v5"
                        else positive_spatial_mil_loss(logits, targets, positive_class_weights)
                    )
                    absent_loss = (
                        dense_absent_class_loss(logits, targets)
                        if config.training_policy == "spatial_mil_v5"
                        else absent_class_hard_negative_loss(logits, targets)
                    )
                    overlap_loss = torch.stack(
                        [
                            overlap_consistency_loss(
                                bag_logits,
                                [Rect(*map(int, values)) for values in bag_rects.tolist()],
                                feature_stride=2 if config.training_policy == "spatial_mil_v5" else 4,
                            )
                            for bag_logits, bag_rects in zip(logits, patch_rect_values, strict=True)
                        ]
                    ).mean()
                    sparse_loss = (
                        present_sparse_budget_loss(logits, targets)
                        if config.training_policy == "spatial_mil_v5"
                        else logits.sum() * 0.0
                    )
                    loss = positive_loss + absent_loss + (
                        0.25 * sparse_loss + 0.10 * overlap_loss
                        if config.training_policy == "spatial_mil_v5"
                        else overlap_loss
                    )
                elif inputs.ndim == 5:
                    batch_size, patch_count, channels, height, width = inputs.shape
                    patch_logits = model(
                        inputs.reshape(batch_size * patch_count, channels, height, width)
                    ).reshape(batch_size, patch_count, config.class_count)
                    logits = max_pool_patch_logits(patch_logits)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
                else:
                    logits = model(inputs)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
                (loss / accumulation_steps).backward()
                if step_number % accumulation_steps == 0 or step_number == total_steps:
                    if config.training_policy == "spatial_mil_v5":
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                loss_value = float(loss.detach().cpu().item())
                elapsed = max(time.monotonic() - started, 0.0)
                eta = (elapsed / step_number) * (total_steps - step_number)
                _emit(
                    output_queue,
                    ProgressMessage(
                        request_id=request_value.request_id,
                        phase="train",
                        epoch=epoch,
                        total_epochs=total_epochs,
                        step=step_number,
                        total_steps=total_steps,
                        loss=loss_value,
                        eta_seconds=max(eta, 0.0),
                        message=f"batch_size={config.batch_size}",
                    ),
                )
                log_lines.append(
                    f"epoch={epoch} step={step_number}/{total_steps} loss={loss_value:.6f}"
                )
                if step_delay:
                    time.sleep(step_delay)

            if validation_loader is not None and (
                not config.retain_epoch_states or epoch == total_epochs
            ):
                validation_loss = _evaluate_v5_validation_loss(
                    model, validation_loader, config, device, positive_class_weights
                )
                log_lines.append(f"epoch={epoch} validation_loss={validation_loss:.6f}")
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_epoch = epoch
                    best_state_dict = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
            if config.retain_epoch_states:
                epoch_dir = staging / "epoch-states"
                epoch_dir.mkdir(exist_ok=True)
                torch.save(
                    {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    epoch_dir / f"epoch-{epoch:03d}.pt",
                )

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict, strict=True)

        if _is_cancelled(cancel_event):
            _emit_cancelled(output_queue, request_value, config, log_lines)
            return
        _write_staged_artifacts(
            staging,
            model,
            config,
            device,
            loss_value,
            total_steps,
            bundle=bundle,
            base_epochs=config.epochs,
            refinement_epochs=5 if refinement_loader is not None else 0,
            selected_epoch=best_epoch,
            selected_validation_loss=(
                best_validation_loss if best_epoch is not None else None
            ),
        )
        terminal_message = (
            f"Training completed; steps={total_steps}; final_loss={loss_value:.6f}; "
            f"batch_size={config.batch_size} unchanged."
        )
        _emit(
            output_queue,
            TerminalMessage(
                request_id=request_value.request_id,
                status="completed",
                message=terminal_message,
                artifact_staging_path=str(staging),
            ),
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
        if _is_out_of_memory(error):
            _cleanup_staging(staging)
            message = (
                f"Out of memory at requested batch_size={config.batch_size}; "
                "batch_size unchanged and no automatic retry was performed. "
                f"Reason: {error}"
            )
            log_lines.append(message)
            _emit(
                output_queue,
                TerminalMessage(
                    request_id=request_value.request_id,
                    status="failed",
                    message="\n".join(log_lines),
                    error_code="out_of_memory",
                ),
            )
            return
        _cleanup_staging(staging)
        _emit_failure(output_queue, request_value, error, log_lines)
    except Exception as error:  # pragma: no cover - defensive worker boundary
        _cleanup_staging(staging)
        _emit_failure(output_queue, request_value, error, log_lines)


def _coerce_request(value: TrainingRequest | Mapping[str, Any] | str) -> TrainingRequest:
    if isinstance(value, TrainingRequest):
        return value
    if isinstance(value, str):
        decoded = decode_message(value)
        if not isinstance(decoded, TrainingRequest):
            raise TrainingWorkerError("serialized worker value must be a TrainingRequest")
        return decoded
    if isinstance(value, Mapping):
        try:
            config = value["config"]
            if isinstance(config, Mapping):
                config = TrainingConfig.from_dict(config)
            return TrainingRequest(
                request_id=value["request_id"],
                config=config,
                artifact_staging_path=value["artifact_staging_path"],
                input_bundle_path=value.get("input_bundle_path"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise TrainingWorkerError("invalid worker request payload") from error
    raise TrainingWorkerError("request must be a TrainingRequest, payload, or JSON value")


def _validate_options(
    synthetic_steps: int,
    inject_oom_step: int | None,
    step_delay: float,
) -> None:
    if isinstance(synthetic_steps, bool) or not isinstance(synthetic_steps, int) or synthetic_steps < 1:
        raise TrainingWorkerError("synthetic_steps must be a positive integer")
    if inject_oom_step is not None and (
        isinstance(inject_oom_step, bool)
        or not isinstance(inject_oom_step, int)
        or inject_oom_step < 1
    ):
        raise TrainingWorkerError("inject_oom_step must be a positive integer or null")
    if (
        isinstance(step_delay, bool)
        or not isinstance(step_delay, (int, float))
        or step_delay < 0
    ):
        raise TrainingWorkerError("step_delay must be a non-negative number")


def _synthetic_batch(config: TrainingConfig, device: torch.device) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    # Preserve the exact requested batch size.  The worker switches the model
    # to eval mode for batch_size=1 so BatchNorm does not force a hidden
    # increase in the persisted/trained configuration.
    count = config.batch_size
    inputs = torch.rand((count, 1, 32, 32), generator=generator, dtype=torch.float32)
    targets = torch.zeros((count, config.class_count), dtype=torch.float32)
    for index in range(count):
        targets[index, (index + config.seed) % config.class_count] = 1.0
        if config.class_count > 1 and index % 2 == 1:
            targets[index, (index + config.seed + 1) % config.class_count] = 1.0
    return inputs.to(device), targets.to(device)


def _freeze_backbone(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.backbone.fc.parameters():
        parameter.requires_grad_(True)


def _evaluate_v5_validation_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    positive_class_weights: Tensor | None,
) -> float:
    """Return deterministic validation loss without updating model state."""

    was_training = model.training
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for inputs, targets, patch_rect_values in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size, patch_count, channels, height, width = inputs.shape
            raw_logits = model(inputs.reshape(batch_size * patch_count, channels, height, width))
            logits = raw_logits.reshape(
                batch_size, patch_count, config.class_count, *raw_logits.shape[-2:]
            )
            overlap = torch.stack(
                [
                    overlap_consistency_loss(
                        bag_logits,
                        [Rect(*map(int, rect)) for rect in bag_rects.tolist()],
                        feature_stride=2,
                    )
                    for bag_logits, bag_rects in zip(logits, patch_rect_values, strict=True)
                ]
            ).mean()
            loss = (
                positive_spatial_lse_loss(logits, targets, positive_class_weights)
                + dense_absent_class_loss(logits, targets)
                + 0.25 * present_sparse_budget_loss(logits, targets)
                + 0.10 * overlap
            )
            values.append(float(loss.cpu().item()))
    model.train(was_training)
    if not values:
        raise RuntimeError("spatial_mil_v5 requires validation members")
    return sum(values) / len(values)


def _write_staged_artifacts(
    staging: Path,
    model: torch.nn.Module,
    config: TrainingConfig,
    device: torch.device,
    final_loss: float,
    total_steps: int,
    *,
    bundle: TrainingInputBundle | None = None,
    base_epochs: int,
    refinement_epochs: int,
    selected_epoch: int | None = None,
    selected_validation_loss: float | None = None,
) -> None:
    spatial_v4 = bundle is not None and config.training_policy == "spatial_mil_v4"
    spatial_v5 = bundle is not None and config.training_policy == "spatial_mil_v5"
    checkpoint = {
        "checkpoint_format": (
            "wafer_defect_studio.resnet18.v5"
            if spatial_v5
            else "wafer_defect_studio.resnet18.v4"
            if spatial_v4
            else _CHECKPOINT_FORMAT_V3
            if bundle is not None and bundle.version == 2
            else _CHECKPOINT_FORMAT_V2
        ),
        "architecture": "resnet18_spatial_logits_v5" if spatial_v5 else "resnet18_spatial_logits" if spatial_v4 else "resnet18",
        "feature_stride": 2 if spatial_v5 else 4 if spatial_v4 else 16,
        "class_count": config.class_count,
        "class_codes": list(bundle.class_codes) if bundle is not None else [
            f"class-{index}" for index in range(config.class_count)
        ],
        "normalization_bounds": (
            [asdict(item) for item in bundle.normalization_bounds] if bundle is not None else []
        ),
        "input_size": (
            {
                "width": (
                    bundle.patch_bags[0].patches[0].width
                    if bundle.version == 2
                    else bundle.samples[0].width
                ),
                "height": (
                    bundle.patch_bags[0].patches[0].height
                    if bundle.version == 2
                    else bundle.samples[0].height
                ),
            }
            if bundle is not None
            else {"width": 32, "height": 32}
        ),
        "snapshot_id": bundle.snapshot_id if bundle is not None else config.snapshot_id,
        "split_id": bundle.split_id if bundle is not None else config.split_id,
        "batch_size": config.batch_size,
        "device": str(device),
        "state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
    }
    if spatial_v5:
        checkpoint.update(
            {
                "patch_size": config.patch_size,
                "patch_stride": config.patch_stride,
                "training_policy": "spatial_mil_v5",
                "loss_policy": {
                    "positive_pooling": "normalized_logsumexp",
                    "negative_dense_hardest_fraction": 0.01,
                    "sparse_probability_budget": 0.01,
                    "sparse_loss_weight": 0.25,
                    "overlap_loss_weight": 0.10,
                },
                "optimizer_policy": {
                    "name": "adamw",
                    "learning_rate": float(config.learning_rate),
                    "weight_decay": 0.0001,
                    "gradient_clip_norm": 5.0,
                },
                "batch_policy": {
                    "physical_batch_size": config.batch_size,
                    "gradient_accumulation_steps": 2 if config.batch_size == 2 else 1,
                    "effective_batch_size": 4,
                },
                "checkpoint_selection": {
                    "source": "validation",
                    "metric": "v5_validation_loss",
                    "selected_epoch": selected_epoch,
                    "value": selected_validation_loss,
                },
                "epochs": base_epochs,
            }
        )
    elif spatial_v4:
        checkpoint.update(
            {
                "patch_size": config.patch_size,
                "patch_stride": config.patch_stride,
                "training_policy": "spatial_mil_v4",
                "loss_weights": {
                    "positive_spatial_mil": 1.0,
                    "absent_class_hard_negative": 1.0,
                    "overlap_consistency": 1.0,
                },
                "positive_class_weighting": {
                    "formula": "negative_bag_count / positive_bag_count",
                    "minimum": 1.0,
                    "maximum": 10.0,
                },
                "augmentation_policy": {
                    **SPATIAL_MIL_V4_AUGMENTATION_POLICY,
                    "seed": config.seed,
                },
            }
        )
        if refinement_epochs:
            checkpoint["hard_negative_refinement"] = {
                "selection_sha256": config.hard_negative_selection_sha256,
                "priority_normal_bag_ids": list(config.priority_normal_bag_ids),
                "base_epochs": base_epochs,
                "refinement_epochs": refinement_epochs,
            }
    elif bundle is not None and bundle.version == 2:
        checkpoint.update(
            {
                "patch_size": config.patch_size,
                "patch_stride": config.patch_stride,
                "bag_pooling": "max",
            }
        )
    model_path = staging / "model.pt"
    metrics_path = staging / "metrics.json"
    manifest_path = staging / "manifest.json"
    torch.save(checkpoint, model_path)
    metrics = {
        "architecture": "resnet18_spatial_logits_v5" if spatial_v5 else "resnet18_spatial_logits" if spatial_v4 else "resnet18",
        "epochs": base_epochs + refinement_epochs,
        "steps": total_steps,
        "final_loss": final_loss,
        "batch_size": config.batch_size,
        "device": str(device),
    }
    if spatial_v5:
        metrics.update(
            {
                "final_epoch_training_loss": final_loss,
                "checkpoint_epoch": selected_epoch,
                "checkpoint_selection_metric": "v5_validation_loss",
                "checkpoint_selection_value": selected_validation_loss,
                "retained_epoch_states": base_epochs if config.retain_epoch_states else 0,
            }
        )
    if refinement_epochs:
        metrics.update({"base_epochs": base_epochs, "refinement_epochs": refinement_epochs})
    metrics_path.write_text(
        json.dumps(
            metrics,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    files = [
        {"path": path.name, "sha256": _sha256(path)}
        for path in (model_path, metrics_path)
    ]
    manifest_path.write_text(
        json.dumps(
            {"required_files": [item["path"] for item in files], "files": files},
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def _emit(output_queue: Any, message: ProgressMessage | TerminalMessage) -> None:
    output_queue.put(encode_message(message))


def _emit_cancelled(
    output_queue: Any,
    request: TrainingRequest,
    config: TrainingConfig,
    log_lines: list[str],
) -> None:
    message = f"Training cancelled; batch_size={config.batch_size} unchanged."
    log_lines.append(message)
    _emit(
        output_queue,
        TerminalMessage(
            request_id=request.request_id,
            status="cancelled",
            message="\n".join(log_lines),
        ),
    )


def _emit_failure(
    output_queue: Any,
    request: TrainingRequest,
    error: Exception,
    log_lines: list[str],
) -> None:
    message = f"Worker failed: {error}"
    log_lines.append(message)
    _emit(
        output_queue,
        TerminalMessage(
            request_id=request.request_id,
            status="failed",
            message="\n".join(log_lines),
            error_code="worker_error",
        ),
    )


def _is_cancelled(cancel_event: Any) -> bool:
    return cancel_event is not None and bool(cancel_event.is_set())


def _is_out_of_memory(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(error).lower()


def _cleanup_staging(staging: Path) -> None:
    for name in ("model.pt", "metrics.json", "manifest.json"):
        try:
            (staging / name).unlink(missing_ok=True)
        except OSError:
            pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "create_training_data_loader",
    "max_pool_patch_logits",
    "TrainingWorkerError",
    "TrainingWorkerHandle",
    "run_worker",
    "start_training_worker",
]
