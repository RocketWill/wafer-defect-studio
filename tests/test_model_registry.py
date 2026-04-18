import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from wafer_defect_studio.model_registry import (
    ModelRegistryError,
    WeightsPolicy,
    create_resnet18,
    load_project_checkpoint,
    resnet18_local_probabilities,
    resolve_device,
)


class ModelRegistryTest(unittest.TestCase):
    def test_project_checkpoint_load_preserves_logits_and_contract(self):
        source = create_resnet18(2, weights=WeightsPolicy.NONE, device="cpu")
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
