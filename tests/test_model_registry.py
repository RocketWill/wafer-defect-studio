import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from wafer_defect_studio.model_registry import (
    ModelRegistryError,
    WeightsPolicy,
    create_resnet18,
    create_resnet18_spatial_logits,
    set_spatial_transfer_training_mode,
    load_project_checkpoint,
    resnet18_local_probabilities,
    resolve_device,
    spatial_logits_to_probabilities,
)


class ModelRegistryTest(unittest.TestCase):
    def test_spatial_transfer_training_mode_freezes_batchnorm_running_stats(self):
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, kernel_size=1),
            torch.nn.BatchNorm2d(2),
            torch.nn.ReLU(),
        )
        model.eval()
        batch_norm = model[1]
        running_mean = batch_norm.running_mean.detach().clone()
        running_var = batch_norm.running_var.detach().clone()

        set_spatial_transfer_training_mode(model)

        self.assertTrue(model.training)
        self.assertTrue(model[0].training)
        self.assertFalse(batch_norm.training)
        self.assertTrue(batch_norm.weight.requires_grad)
        self.assertTrue(batch_norm.bias.requires_grad)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.zero_grad()
        model(torch.ones((2, 1, 4, 4))).sum().backward()
        optimizer.step()
        self.assertIsNotNone(batch_norm.weight.grad)
        self.assertIsNotNone(batch_norm.bias.grad)
        self.assertTrue(torch.equal(batch_norm.running_mean, running_mean))
        self.assertTrue(torch.equal(batch_norm.running_var, running_var))

    def test_v4_checkpoint_loads_strict_spatial_model_and_preserves_absolute_probabilities(self):
        source = create_resnet18_spatial_logits(2, device="cpu")
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "v4.pt"
            torch.save(
                {
                    "checkpoint_format": "wafer_defect_studio.resnet18.v4",
                    "architecture": "resnet18_spatial_logits",
                    "class_count": 2,
                    "class_codes": ["scratch", "particle"],
                    "normalization_bounds": [{"dtype": "uint8", "source_min": 0, "source_max": 255, "low": 0.0, "high": 255.0, "low_percentile": 1.0, "high_percentile": 99.0}],
                    "input_size": {"width": 32, "height": 32},
                    "feature_stride": 4,
                    "patch_size": 32,
                    "patch_stride": 16,
                    "training_policy": "spatial_mil_v4",
                    "loss_weights": {"positive_spatial_mil": 1.0, "absent_class_hard_negative": 1.0, "overlap_consistency": 1.0},
                    "positive_class_weighting": {"formula": "negative_bag_count / positive_bag_count", "minimum": 1.0, "maximum": 10.0},
                    "augmentation_policy": {"name": "spatial_mil_v4_defect_preserving_affine", "contrast": [0.9, 1.1], "brightness": [-0.03, 0.03], "seed": 7, "seed_formula": "run_seed + epoch * 1_000_003 + bag_index"},
                    "state_dict": source.state_dict(),
                }, path,
            )
            loaded, checkpoint = load_project_checkpoint(path, device="cpu")
            self.assertEqual(loaded.architecture, "resnet18_spatial_logits")
            self.assertEqual(checkpoint["checkpoint_format"], "wafer_defect_studio.resnet18.v4")
            broken = torch.load(path, weights_only=False)
            broken["state_dict"] = {**broken["state_dict"], "spatial_head.weight": torch.zeros((1, 960, 1, 1))}
            torch.save(broken, path)
            with self.assertRaisesRegex(ModelRegistryError, "spatial logits"):
                load_project_checkpoint(path, device="cpu")

        logits = torch.tensor([[[[-2.0, 0.0], [2.0, 4.0]]]])
        actual = spatial_logits_to_probabilities(logits, (4, 6))
        expected = torch.nn.functional.interpolate(torch.sigmoid(logits), size=(4, 6), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        wrong_order = torch.sigmoid(torch.nn.functional.interpolate(logits, size=(4, 6), mode="bilinear", align_corners=False)).permute(0, 2, 3, 1)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(torch.allclose(actual, wrong_order))
        self.assertEqual(actual.shape, (1, 4, 6, 1))

    def test_default_stride16_preserves_legacy_state_dict_shapes(self):
        current = create_resnet18(2, device="cpu")
        legacy = create_resnet18(2, device="cpu", feature_stride=32)
        current.eval()
        legacy.eval()
        current_features = []
        legacy_features = []
        current_hook = current.backbone.layer4.register_forward_hook(
            lambda _module, _inputs, output: current_features.append(output.shape[-2:])
        )
        legacy_hook = legacy.backbone.layer4.register_forward_hook(
            lambda _module, _inputs, output: legacy_features.append(output.shape[-2:])
        )
        try:
            with torch.no_grad():
                current(torch.zeros((1, 1, 64, 64)))
                legacy(torch.zeros((1, 1, 64, 64)))
        finally:
            current_hook.remove()
            legacy_hook.remove()
        self.assertEqual(current_features, [torch.Size((4, 4))])
        self.assertEqual(legacy_features, [torch.Size((2, 2))])
        self.assertEqual(
            {name: value.shape for name, value in current.state_dict().items()},
            {name: value.shape for name, value in legacy.state_dict().items()},
        )

    def test_project_checkpoint_load_preserves_logits_and_contract(self):
        source = create_resnet18(
            2,
            weights=WeightsPolicy.NONE,
            device="cpu",
            feature_stride=32,
        )
        source.eval()
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "model.pt"
            torch.save(
                {
                    "checkpoint_format": "wafer_defect_studio.resnet18.v1",
                    "architecture": "resnet18",
                    "class_count": 2,
                    "class_codes": ["scratch", "particle"],
                    "normalization_bounds": [
                        {
                            "dtype": "uint8",
                            "source_min": 0,
                            "source_max": 255,
                            "low": 0.0,
                            "high": 255.0,
                            "low_percentile": 1.0,
                            "high_percentile": 99.0,
                        }
                    ],
                    "input_size": {"width": 32, "height": 32},
                    "state_dict": source.state_dict(),
                },
                path,
            )
            loaded, checkpoint = load_project_checkpoint(
                path,
                device="cpu",
                expected_class_codes=("scratch", "particle"),
                expected_input_size=(32, 32),
            )
            inputs = torch.rand((1, 3, 32, 32), generator=torch.Generator().manual_seed(4))
            with torch.no_grad():
                self.assertTrue(torch.equal(source(inputs), loaded(inputs)))
            self.assertEqual(checkpoint["class_codes"], ["scratch", "particle"])
            legacy_features = []
            legacy_hook = loaded.backbone.layer4.register_forward_hook(
                lambda _module, _inputs, output: legacy_features.append(output.shape[-2:])
            )
            try:
                with torch.no_grad():
                    loaded(torch.zeros((1, 1, 64, 64)))
            finally:
                legacy_hook.remove()
            self.assertEqual(legacy_features, [torch.Size((2, 2))])

            v2_path = Path(temporary_directory) / "model-v2.pt"
            torch.save(
                {
                    **checkpoint,
                    "checkpoint_format": "wafer_defect_studio.resnet18.v2",
                    "feature_stride": 16,
                },
                v2_path,
            )
            loaded_v2, _ = load_project_checkpoint(v2_path, device="cpu")
            features = []
            hook = loaded_v2.backbone.layer4.register_forward_hook(
                lambda _module, _inputs, output: features.append(output.shape[-2:])
            )
            try:
                with torch.no_grad():
                    loaded_v2(torch.zeros((1, 1, 64, 64)))
            finally:
                hook.remove()
            self.assertEqual(features, [torch.Size((4, 4))])

            v3_path = Path(temporary_directory) / "model-v3.pt"
            torch.save(
                {
                    **checkpoint,
                    "checkpoint_format": "wafer_defect_studio.resnet18.v3",
                    "feature_stride": 16,
                    "patch_size": 128,
                    "patch_stride": 64,
                    "bag_pooling": "max",
                },
                v3_path,
            )
            loaded_v3, loaded_v3_checkpoint = load_project_checkpoint(v3_path, device="cpu")
            self.assertEqual(loaded_v3.feature_stride, 16)
            self.assertEqual(loaded_v3_checkpoint["patch_size"], 128)
            self.assertTrue(
                all(
                    torch.equal(value, loaded_v3.state_dict()[name])
                    for name, value in checkpoint["state_dict"].items()
                )
            )

            for field, value, message in (
                ("patch_size", None, "patch_size"),
                ("patch_stride", 0, "patch_stride"),
                ("patch_stride", 129, "patch_stride cannot exceed patch_size"),
                ("bag_pooling", "mean", "bag_pooling must be max"),
            ):
                invalid_v3 = {
                    **checkpoint,
                    "checkpoint_format": "wafer_defect_studio.resnet18.v3",
                    "feature_stride": 16,
                    "patch_size": 128,
                    "patch_stride": 64,
                    "bag_pooling": "max",
                }
                if value is None:
                    invalid_v3.pop(field)
                else:
                    invalid_v3[field] = value
                invalid_v3_path = Path(temporary_directory) / f"invalid-v3-{field}-{value}.pt"
                torch.save(invalid_v3, invalid_v3_path)
                with self.assertRaisesRegex(ModelRegistryError, message):
                    load_project_checkpoint(invalid_v3_path, device="cpu")

            for feature_stride in (None, 32):
                invalid_path = Path(temporary_directory) / f"invalid-{feature_stride}.pt"
                invalid = {
                    **checkpoint,
                    "checkpoint_format": "wafer_defect_studio.resnet18.v2",
                }
                if feature_stride is not None:
                    invalid["feature_stride"] = feature_stride
                torch.save(invalid, invalid_path)
                with self.assertRaisesRegex(ModelRegistryError, "feature_stride must be 16"):
                    load_project_checkpoint(invalid_path, device="cpu")

    def test_resnet18_contract_is_multilabel_and_duplicates_grayscale(self):
        model = create_resnet18(3, weights=WeightsPolicy.NONE, device="cpu")
        model.eval()
        captured = []

        def capture(module, inputs):
            captured.append(inputs[0].detach().clone())

        hook = model.backbone.conv1.register_forward_pre_hook(capture)
        try:
            logits = model(torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]]))
        finally:
            hook.remove()

        self.assertEqual(logits.shape, (1, 3))
        self.assertEqual(captured[0].shape[1], 3)
        self.assertTrue(torch.equal(captured[0][:, 0], captured[0][:, 1]))
        self.assertTrue(torch.equal(captured[0][:, 1], captured[0][:, 2]))

    def test_registry_rejects_unsupported_architecture_and_external_weights(self):
        with self.assertRaises(ModelRegistryError):
            create_resnet18(2, weights="external.pth", device="cpu")
        with self.assertRaises(ModelRegistryError):
            create_resnet18(2, weights=WeightsPolicy.NONE, device="cpu", weights_path="x.pth")

    def test_all_convolutional_projection_matches_direct_sigmoid_conv(self):
        model = create_resnet18(2, device="cpu")
        model.backbone.fc.weight.data.zero_()
        model.backbone.fc.bias.data[:] = torch.tensor([0.25, -0.5])
        model.backbone.fc.weight.data[0, 0] = 1.0
        model.backbone.fc.weight.data[1, 1] = 2.0
        features = torch.zeros((1, 512, 2, 2), dtype=torch.float32)
        features[0, 0] = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        features[0, 1] = torch.tensor([[3.0, 2.0], [1.0, 0.0]])
        from wafer_defect_studio.model_registry import _project_layer4_features

        actual = _project_layer4_features(
            features,
            model.backbone.fc.weight,
            model.backbone.fc.bias,
            (2, 2),
        )
        expected_logits = torch.nn.functional.conv2d(
            features,
            model.backbone.fc.weight[:, :, None, None],
            model.backbone.fc.bias,
        )
        expected = torch.sigmoid(expected_logits).permute(0, 2, 3, 1)
        self.assertTrue(torch.allclose(actual, expected))
        self.assertEqual(actual.shape, (1, 2, 2, 2))

    def test_resnet18_local_probabilities_returns_window_maps(self):
        model = create_resnet18(2, device="cpu")
        model.eval()
        maps = resnet18_local_probabilities(model, torch.zeros((1, 1, 32, 32)))
        self.assertEqual(maps.shape, (1, 32, 32, 2))
        self.assertTrue(torch.all((maps >= 0.0) & (maps <= 1.0)))

    def test_device_policy_prefers_cuda_when_available_and_falls_back_to_cpu(self):
        resolved = resolve_device()
        self.assertEqual(resolved.type, "cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    unittest.main()
