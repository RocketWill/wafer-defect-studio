"""The small, explicit model registry used by background training."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18

from .training_run import TrainingRunError, validate_project_checkpoint


class ModelRegistryError(ValueError):
    """Raised when a model request is outside the internal model contract."""


class WeightsPolicy(str, Enum):
    """Explicit source of ResNet18 weights.

    ``NONE`` is the default training-from-scratch policy and never performs
    network or filesystem weight acquisition.  ``IMAGENET`` is intentionally
    opt-in; torchvision may acquire its official weights when they are not
    already cached.
    """

    NONE = "none"
    IMAGENET = "imagenet"


SUPPORTED_ARCHITECTURES = ("resnet18",)


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve the project device policy: CUDA when available, otherwise CPU.

    Passing ``None`` or ``"auto"`` applies the policy.  An explicit device is
    honoured and an unavailable explicit CUDA request fails rather than being
    silently changed.
    """

    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ModelRegistryError(f"Unsupported device: {device!r}") from error
    if resolved.type not in {"cpu", "cuda"}:
        raise ModelRegistryError(f"Only CPU and CUDA devices are supported: {device!r}")
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ModelRegistryError("CUDA was requested but is not available")
    return resolved


def duplicate_grayscale_channels(inputs: Tensor) -> Tensor:
    """Convert a grayscale batch to the three channels expected by ResNet18."""

    if not isinstance(inputs, Tensor):
        raise ModelRegistryError("model input must be a torch.Tensor")
    if inputs.ndim == 2:
        inputs = inputs.unsqueeze(0).unsqueeze(0)
    elif inputs.ndim == 3:
        inputs = inputs.unsqueeze(1)
    if inputs.ndim != 4:
        raise ModelRegistryError("model input must have shape (N, C, H, W)")
    if inputs.shape[1] == 1:
        return inputs.repeat(1, 3, 1, 1)
    if inputs.shape[1] == 3:
        return inputs
    raise ModelRegistryError("ResNet18 accepts one grayscale or three-channel input channels")


class ResNet18Classifier(nn.Module):
    """Torchvision ResNet18 with a multi-label logits head."""

    architecture = "resnet18"

    def __init__(self, class_count: int, *, weights: WeightsPolicy = WeightsPolicy.NONE) -> None:
        super().__init__()
        if isinstance(class_count, bool) or not isinstance(class_count, int) or class_count < 1:
            raise ModelRegistryError("class_count must be a positive integer")
        self.class_count = class_count
        self.weights_policy = _coerce_weights_policy(weights)
        torchvision_weights = (
            None
            if self.weights_policy is WeightsPolicy.NONE
            else ResNet18_Weights.DEFAULT
        )
        self.backbone = resnet18(weights=torchvision_weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, class_count)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.backbone(duplicate_grayscale_channels(inputs))


class ModelRegistry:
    """Thin internal registry; ResNet18 is the only supported architecture."""

    supported_architectures = SUPPORTED_ARCHITECTURES

    def create(
        self,
        architecture: str,
        class_count: int,
        *,
        weights: WeightsPolicy | str | None = WeightsPolicy.NONE,
        device: str | torch.device | None = None,
        weights_path: str | Path | None = None,
    ) -> ResNet18Classifier:
        if not isinstance(architecture, str) or architecture.lower() != "resnet18":
            raise ModelRegistryError(
                f"Unsupported architecture {architecture!r}; only torchvision ResNet18 is supported"
            )
        if weights_path is not None:
            raise ModelRegistryError("External .pth loading is not supported")
        model = ResNet18Classifier(class_count, weights=_coerce_weights_policy(weights))
        return model.to(resolve_device(device))


def create_resnet18(
    class_count: int,
    *,
    weights: WeightsPolicy | str | None = WeightsPolicy.NONE,
    device: str | torch.device | None = None,
    weights_path: str | Path | None = None,
) -> ResNet18Classifier:
    """Create the supported multi-label ResNet18 on the resolved device."""

    return ModelRegistry().create(
        "resnet18",
        class_count,
        weights=weights,
        device=device,
        weights_path=weights_path,
    )


def create_model(
    architecture: str,
    class_count: int,
    *,
    weights: WeightsPolicy | str | None = WeightsPolicy.NONE,
    device: str | torch.device | None = None,
    weights_path: str | Path | None = None,
) -> ResNet18Classifier:
    """Create a model through the internal architecture registry."""

    return ModelRegistry().create(
        architecture,
        class_count,
        weights=weights,
        device=device,
        weights_path=weights_path,
    )


def load_project_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = "cpu",
    expected_class_codes: tuple[str, ...] | None = None,
    expected_input_size: tuple[int, int] | None = None,
) -> tuple[ResNet18Classifier, dict[str, Any]]:
    """Load one validated project-produced ResNet18 checkpoint."""

    try:
        checkpoint = validate_project_checkpoint(
            checkpoint_path,
            expected_class_codes=expected_class_codes,
        )
    except TrainingRunError as error:
        raise ModelRegistryError(str(error)) from error
    input_size = checkpoint["input_size"]
    if expected_input_size is not None and (
        input_size["width"], input_size["height"]
    ) != tuple(expected_input_size):
        raise ModelRegistryError("project checkpoint input rectangle does not match the request")
    model = create_resnet18(
        checkpoint["class_count"],
        weights=WeightsPolicy.NONE,
        device=device,
    )
    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ModelRegistryError("project checkpoint state_dict does not match ResNet18") from error
    model.eval()
    return model, checkpoint


def resnet18_local_probabilities(
    model: ResNet18Classifier,
    inputs: Tensor,
) -> Tensor:
    """Project ResNet18 layer4 features to per-pixel sigmoid class maps."""

    if not isinstance(model, ResNet18Classifier):
        raise ModelRegistryError("model must be a ResNet18Classifier")
    inputs = duplicate_grayscale_channels(inputs)
    backbone = model.backbone
    features = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(inputs))))
    features = backbone.layer1(features)
    features = backbone.layer2(features)
    features = backbone.layer3(features)
    features = backbone.layer4(features)
    return _project_layer4_features(
        features,
        backbone.fc.weight,
        backbone.fc.bias,
        (inputs.shape[-2], inputs.shape[-1]),
    )


def _project_layer4_features(
    features: Tensor,
    classifier_weight: Tensor,
    classifier_bias: Tensor,
    output_size: tuple[int, int],
) -> Tensor:
    if not isinstance(features, Tensor) or features.ndim != 4:
        raise ModelRegistryError("layer4 features must have shape (N, F, H, W)")
    if classifier_weight.ndim != 2 or classifier_weight.shape[1] != features.shape[1]:
        raise ModelRegistryError("classifier weight feature axis must match layer4 features")
    if classifier_bias.ndim != 1 or classifier_bias.shape[0] != classifier_weight.shape[0]:
        raise ModelRegistryError("classifier bias must match classifier weight classes")
    if (
        len(output_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in output_size)
    ):
        raise ModelRegistryError("output_size must be a positive (height, width) pair")
    logits = F.conv2d(
        features,
        classifier_weight[:, :, None, None],
        classifier_bias,
    )
    logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
    return torch.sigmoid(logits).permute(0, 2, 3, 1).contiguous()


def _coerce_weights_policy(value: Any) -> WeightsPolicy:
    if value is None:
        return WeightsPolicy.NONE
    if isinstance(value, WeightsPolicy):
        return value
    if isinstance(value, Path):
        raise ModelRegistryError("External .pth loading is not supported")
    if isinstance(value, str):
        try:
            return WeightsPolicy(value)
        except ValueError as error:
            raise ModelRegistryError(
                "weights must be WeightsPolicy.NONE or the explicit IMAGENET policy; "
                "external .pth loading is not supported"
            ) from error
    raise ModelRegistryError("weights must be an explicit WeightsPolicy value")


__all__ = [
    "ModelRegistry",
    "ModelRegistryError",
    "ResNet18Classifier",
    "SUPPORTED_ARCHITECTURES",
    "WeightsPolicy",
    "create_model",
    "create_resnet18",
    "duplicate_grayscale_channels",
    "load_project_checkpoint",
    "resnet18_local_probabilities",
    "resolve_device",
]
