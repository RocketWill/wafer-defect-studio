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

from .model_registry import create_resnet18, resolve_device
from .training_input_bundle import TrainingInputBundle
from .training_patch_dataset import TrainingPatchDataset
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


_CHECKPOINT_FORMAT = "wafer_defect_studio.resnet18.v1"


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
            sample_sizes = {(sample.width, sample.height) for sample in bundle.samples}
            if len(sample_sizes) != 1:
                raise RuntimeError(
                    "training input bundle contains variable sample sizes; "
                    "a fixed model input rectangle is required"
                )
            dataset = TrainingPatchDataset(bundle, "train")
            generator = torch.Generator(device="cpu")
            generator.manual_seed(config.seed)
            loader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=True,
                generator=generator,
                num_workers=0,
            )
        model = create_resnet18(
            config.class_count,
            weights=config.weights_policy,
            device=device,
        )
        if bundle is None:
            _freeze_backbone(model)
            optimizer_parameters = model.backbone.fc.parameters()
        else:
            optimizer_parameters = model.parameters()
        optimizer = torch.optim.SGD(
            optimizer_parameters,
            lr=float(config.learning_rate),
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
        started = time.monotonic()
        step_number = 0

        for epoch in range(1, config.epochs + 1):
            batches = (
                loader
                if loader is not None
                else ((synthetic_inputs, synthetic_targets) for _ in range(synthetic_steps))
            )
            for inputs, targets in batches:
                step_number += 1
                if _is_cancelled(cancel_event):
                    _emit_cancelled(output_queue, request_value, config, log_lines)
                    return
                if inject_oom_step == step_number:
                    raise torch.cuda.OutOfMemoryError(
                        f"injected out-of-memory failure at step {step_number}"
                    )

                optimizer.zero_grad(set_to_none=True)
                if loader is not None:
                    model.train() if inputs.shape[0] > 1 else model.eval()
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                logits = model(inputs)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
                loss.backward()
                optimizer.step()
                loss_value = float(loss.detach().cpu().item())
                elapsed = max(time.monotonic() - started, 0.0)
                eta = (elapsed / step_number) * (total_steps - step_number)
                _emit(
                    output_queue,
                    ProgressMessage(
                        request_id=request_value.request_id,
                        phase="train",
                        epoch=epoch,
                        total_epochs=config.epochs,
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


def _write_staged_artifacts(
    staging: Path,
    model: torch.nn.Module,
    config: TrainingConfig,
    device: torch.device,
    final_loss: float,
    total_steps: int,
    *,
    bundle: TrainingInputBundle | None = None,
) -> None:
    checkpoint = {
        "checkpoint_format": _CHECKPOINT_FORMAT,
        "architecture": "resnet18",
        "class_count": config.class_count,
        "class_codes": list(bundle.class_codes) if bundle is not None else [
            f"class-{index}" for index in range(config.class_count)
        ],
        "normalization_bounds": (
            [asdict(item) for item in bundle.normalization_bounds] if bundle is not None else []
        ),
        "input_size": (
            {
                "width": bundle.samples[0].width,
                "height": bundle.samples[0].height,
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
    model_path = staging / "model.pt"
    metrics_path = staging / "metrics.json"
    manifest_path = staging / "manifest.json"
    torch.save(checkpoint, model_path)
    metrics_path.write_text(
        json.dumps(
            {
                "architecture": "resnet18",
                "epochs": config.epochs,
                "steps": total_steps,
                "final_loss": final_loss,
                "batch_size": config.batch_size,
                "device": str(device),
            },
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
    "TrainingWorkerError",
    "TrainingWorkerHandle",
    "run_worker",
    "start_training_worker",
]
