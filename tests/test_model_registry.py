import unittest

import torch

from wafer_defect_studio.model_registry import (
    ModelRegistryError,
    WeightsPolicy,
    create_resnet18,
    resolve_device,
)


class ModelRegistryTest(unittest.TestCase):
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

    def test_device_policy_prefers_cuda_when_available_and_falls_back_to_cpu(self):
        resolved = resolve_device()
        self.assertEqual(resolved.type, "cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    unittest.main()
