"""Value-only Training Request configuration controls."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QSpinBox, QWidget

from .training_inputs import TrainingInputControls, TrainingInputOption
from .training_protocol import TrainingConfig, TrainingRequest, TrainingProtocolError


class TrainingConfigurationError(ValueError):
    """Raised when selected immutable inputs cannot build a Training Request."""


class TrainingConfigurationControls(QWidget):
    """Configure the minimal supported ResNet18 Training Request values."""

    def __init__(
        self,
        input_controls: TrainingInputControls,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._input_controls = input_controls
        self.epochs_spin = QSpinBox(self)
        self.epochs_spin.setObjectName("trainingEpochsSpinBox")
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(10)
        self.batch_size_spin = QSpinBox(self)
        self.batch_size_spin.setObjectName("trainingBatchSizeSpinBox")
        self.batch_size_spin.setRange(1, 4096)
        self.batch_size_spin.setValue(8)
        self.learning_rate_spin = QDoubleSpinBox(self)
        self.learning_rate_spin.setObjectName("trainingLearningRateSpinBox")
        self.learning_rate_spin.setDecimals(6)
        self.learning_rate_spin.setRange(0.000001, 10.0)
        self.learning_rate_spin.setValue(0.001)
        self.seed_spin = QSpinBox(self)
        self.seed_spin.setObjectName("trainingSeedSpinBox")
        self.seed_spin.setRange(-2147483648, 2147483647)
        self.seed_spin.setValue(0)
        self.device_combo = QComboBox(self)
        self.device_combo.setObjectName("trainingDeviceComboBox")
        self.device_combo.addItems(("auto", "cpu", "cuda"))
        self.weights_combo = QComboBox(self)
        self.weights_combo.setObjectName("trainingWeightsPolicyComboBox")
        self.weights_combo.addItems(("none", "imagenet"))
        self.model_mode_combo = QComboBox(self)
        self.model_mode_combo.setObjectName("trainingModelModeComboBox")
        self.model_mode_combo.addItem("CAM v2", "cam_v2")
        self.model_mode_combo.addItem("Patch Classification v3", "patch_v3")
        self.model_mode_combo.addItem("Spatial MIL v4", "spatial_mil_v4")
        self.patch_size_spin = QSpinBox(self)
        self.patch_size_spin.setObjectName("trainingPatchSizeSpinBox")
        self.patch_size_spin.setRange(1, 100000)
        self.patch_size_spin.setValue(128)
        self.patch_stride_spin = QSpinBox(self)
        self.patch_stride_spin.setObjectName("trainingPatchStrideSpinBox")
        self.patch_stride_spin.setRange(1, 100000)
        self.patch_stride_spin.setValue(64)
        self.bag_pooling_label = QLabel("max", self)
        self.bag_pooling_label.setObjectName("trainingBagPoolingLabel")
        self.request_summary_label = QLabel(self)
        self.request_summary_label.setObjectName("trainingRequestSummaryLabel")
        self.request_summary_label.setWordWrap(True)
        self.context_status_label = QLabel("Training context ready.", self)
        self.context_status_label.setObjectName("trainingContextStatusLabel")
        self.context_status_label.setWordWrap(True)

        layout = QFormLayout(self)
        layout.addRow("Epochs", self.epochs_spin)
        layout.addRow("Batch size", self.batch_size_spin)
        layout.addRow("Learning rate", self.learning_rate_spin)
        layout.addRow("Seed", self.seed_spin)
        layout.addRow("Device", self.device_combo)
        layout.addRow("Weights policy", self.weights_combo)
        layout.addRow("Model mode", self.model_mode_combo)
        layout.addRow("Model Patch size", self.patch_size_spin)
        layout.addRow("Model Patch stride", self.patch_stride_spin)
        layout.addRow("Bag pooling", self.bag_pooling_label)
        layout.addRow("Request summary", self.request_summary_label)
        layout.addRow(self.context_status_label)
        self.model_mode_combo.currentIndexChanged.connect(self._sync_patch_controls)
        self.patch_size_spin.valueChanged.connect(self._update_request_summary)
        self.patch_stride_spin.valueChanged.connect(self._update_request_summary)
        self._sync_patch_controls()

    def _sync_patch_controls(self) -> None:
        mode = self.model_mode_combo.currentData()
        visible = mode in {"patch_v3", "spatial_mil_v4"}
        self.bag_pooling_label.setText(
            "spatial logits (no scalar pooling)" if mode == "spatial_mil_v4" else "max"
        )
        layout = self.layout()
        for widget in (self.patch_size_spin, self.patch_stride_spin, self.bag_pooling_label):
            widget.setVisible(visible)
            label = layout.labelForField(widget)
            if label is not None:
                label.setVisible(visible)
        self._update_request_summary()

    def _update_request_summary(self) -> None:
        mode = self.model_mode_combo.currentData()
        if mode == "patch_v3":
            self.request_summary_label.setText(
                "Patch Classification v3 | "
                f"Model Patch: {self.patch_size_spin.value()}x{self.patch_size_spin.value()} | "
                f"stride: {self.patch_stride_spin.value()} | bag pooling: max"
            )
        elif mode == "spatial_mil_v4":
            self.request_summary_label.setText(
                "Spatial MIL v4 | "
                f"Model Patch: {self.patch_size_spin.value()}x{self.patch_size_spin.value()} | "
                f"stride: {self.patch_stride_spin.value()} | spatial logits; no scalar pooling"
            )
        else:
            self.request_summary_label.setText("CAM v2 | legacy Grid sample training")

    def set_context_status(self, message: str, ready: bool) -> None:
        self.context_status_label.setText(message)
        self.context_status_label.setProperty("contextReady", ready)

    def context_error(self) -> str | None:
        try:
            self.build_request("context-check", "context-staging")
        except TrainingConfigurationError as error:
            return str(error)
        return None

    def build_request(self, request_id: str, artifact_staging_path: str | Path) -> TrainingRequest:
        snapshot = self._selected_option(self._input_controls.snapshot_combo, "Snapshot")
        split = self._selected_option(self._input_controls.split_combo, "Split")
        if snapshot.class_count is None or snapshot.class_count <= 0:
            raise TrainingConfigurationError("Snapshot class count is unavailable")
        if split.snapshot_id != snapshot.identifier:
            raise TrainingConfigurationError("Split does not belong to selected Snapshot")
        try:
            mode = self.model_mode_combo.currentData()
            patch_mode = mode in {"patch_v3", "spatial_mil_v4"}
            config = TrainingConfig(
                snapshot_id=snapshot.identifier,
                split_id=split.identifier,
                class_count=snapshot.class_count,
                epochs=self.epochs_spin.value(),
                batch_size=self.batch_size_spin.value(),
                architecture="resnet18",
                device=self.device_combo.currentText(),
                seed=self.seed_spin.value(),
                learning_rate=self.learning_rate_spin.value(),
                weights_policy=self.weights_combo.currentText(),
                patch_size=self.patch_size_spin.value() if patch_mode else None,
                patch_stride=self.patch_stride_spin.value() if patch_mode else None,
                training_policy=(
                    "spatial_mil_v4" if mode == "spatial_mil_v4" else "legacy"
                ),
            )
            return TrainingRequest(request_id, config, artifact_staging_path)
        except (TrainingProtocolError, TypeError, ValueError) as error:
            raise TrainingConfigurationError(str(error)) from error

    @staticmethod
    def _selected_option(combo: QComboBox, label: str) -> TrainingInputOption:
        index = combo.currentIndex()
        option = combo.itemData(index, Qt.UserRole + 1) if index >= 0 else None
        if not isinstance(option, TrainingInputOption) or not option.available:
            raise TrainingConfigurationError(f"{label} is unavailable")
        return option


__all__ = ["TrainingConfigurationControls", "TrainingConfigurationError"]
