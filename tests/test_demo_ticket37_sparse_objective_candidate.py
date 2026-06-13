import tempfile
import unittest
import copy
from pathlib import Path

import numpy as np
import torch

from docs.demo.ticket37_sparse_objective_candidate import (
    CANDIDATE_ARTIFACT_ROOT,
    DETERMINISTIC_POLICY,
    WEIGHTS_SHA256,
    WEIGHTS_URL,
    compute_ticket37_objective,
    evaluate_ticket37_response,
    seal_ticket37_artifacts,
    validate_ticket37_maps_artifact,
    verify_ticket37_artifact_seal,
)
from docs.demo.ticket37_sparse_objective_candidate import (
    _establish_deterministic_policy,
    _validate_cached_imagenet_weights,
    _validate_output_targets,
    _validate_runtime_contract,
    _verify_ticket37_artifact_bytes,
    _write_json,
    validate_ticket37_confidence_map,
)
from docs.demo.ticket37_sparse_objective_contract import build_ticket37_sparse_objective_contract
from wafer_defect_studio.detection_windows import Rect
from wafer_defect_studio.spatial_mil import (
    dense_absent_class_loss,
    overlap_consistency_loss,
    same_image_grid_ranking_loss,
    sparse_instance_localization_loss,
)


class Ticket37SparseObjectiveCandidateTest(unittest.TestCase):
    def test_public_objective_combines_sparse_and_context_terms(self) -> None:
        logits = torch.zeros((2, 2, 2, 2), requires_grad=True)
        targets = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        instance_masks = torch.tensor([[[True, False], [False, False]]])
        batch_indices = torch.tensor([0], dtype=torch.long)
        class_indices = torch.tensor([0], dtype=torch.long)
        far_masks = torch.ones((2, 2, 2), dtype=torch.bool)
        rects = [[Rect(0, 0, 4, 4)], [Rect(0, 0, 4, 4)]]

        kwargs = dict(
            logits=logits,
            targets=targets,
            image_group_ids=("image", "image"),
            instance_masks=instance_masks,
            instance_batch_indices=batch_indices,
            instance_class_indices=class_indices,
            far_negative_masks=far_masks,
            far_batch_indices=torch.tensor([0, 1], dtype=torch.long),
            far_class_indices=torch.tensor([0, 1], dtype=torch.long),
            patch_rect_groups=rects,
        )
        loss = compute_ticket37_objective(
            **kwargs,
        )
        sparse = sparse_instance_localization_loss(
            logits,
            instance_masks,
            batch_indices,
            class_indices,
            far_masks,
            kwargs["far_batch_indices"],
            kwargs["far_class_indices"],
        )
        bag_logits = logits.reshape(2, 1, 2, 2, 2)
        expected = (
            sparse
            + 0.5 * dense_absent_class_loss(bag_logits, targets)
            + 0.25 * same_image_grid_ranking_loss(bag_logits, targets, ("image", "image"))
            + 0.10 * torch.stack(
                [overlap_consistency_loss(logits[index : index + 1], rects[index], feature_stride=2) for index in range(2)]
            ).mean()
        )
        self.assertTrue(torch.allclose(loss, expected))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_remote_response_is_diagnostic_and_cross_component_merge_fails(self) -> None:
        rendered = {
            "schema": "ticket37-proposal-evaluation.v1",
            "response_shape": [32, 32],
            "unique_truth_count": 2,
            "truth_component_count": 2,
            "unique_truths": [
                {"truth_id": 0, "instance_ids": ["a"], "pixels": [[1, 1]], "area": 1},
                {"truth_id": 1, "instance_ids": ["b"], "pixels": [[20, 20]], "area": 1},
            ],
            "truth_components": [
                {"component_id": 0, "truth_ids": [0], "instance_ids": ["a"], "pixels": [[1, 1]], "area": 1},
                {"component_id": 1, "truth_ids": [1], "instance_ids": ["b"], "pixels": [[20, 20]], "area": 1},
            ],
        }
        response = np.zeros((32, 32), dtype=np.float32)
        response[1, 1] = 1.0
        response[20, 20] = 1.0
        row, _matching = evaluate_ticket37_response(
            response,
            rendered,
            remote_response={"numerator": 100, "denominator": 100, "ratio": 1.0},
        )
        self.assertEqual(row["overall"], "PASS")
        self.assertEqual(row["remote_response"]["ratio"], 1.0)

        merged = np.zeros((32, 32), dtype=np.float32)
        merged[1:21, 1:21] = 1.0
        merged_row, _ = evaluate_ticket37_response(merged, rendered)
        self.assertEqual(merged_row["overall"], "FAIL")
        self.assertEqual(merged_row["cross_component_merge_proposal_ids"], [0])

    def test_seal_re_read_and_tamper_fail_closed_without_materializing_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            root = repo / CANDIDATE_ARTIFACT_ROOT
            root.mkdir(parents=True)
            for name in ("checkpoint.pt", "confidence-maps.npz", "proposal-matching.json", "configuration.json"):
                (root / name).write_bytes(name.encode("ascii"))
            seal = seal_ticket37_artifacts(root, repo_root=repo)
            _verify_ticket37_artifact_bytes(seal, repo)
            target = root / "configuration.json"
            target.write_bytes(target.read_bytes() + b"tamper")
            with self.assertRaises(ValueError):
                _verify_ticket37_artifact_bytes(seal, repo)

    def test_preflight_rejects_existing_root_before_git_or_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            (repo / CANDIDATE_ARTIFACT_ROOT).mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                _validate_output_targets(repo, repo / "docs/demo/ticket37-sparse-objective-candidate.json")
            (repo / CANDIDATE_ARTIFACT_ROOT).rmdir()
            report = repo / "docs/demo/ticket37-sparse-objective-candidate.json"
            report.parent.mkdir(parents=True)
            report.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _validate_output_targets(repo, report)

    def test_stored_maps_require_float32_finite_unit_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "maps.npz"
            shape = (1536, 1536, 2)
            valid = {case_id: np.zeros(shape, dtype=np.float32) for case_id in (
                "ticket35-density-101-train-scratch-001-000-01",
                "ticket35-density-101-train-scratch-150-000-05",
                "ticket35-density-101-train-particle-000-001-06",
                "ticket35-density-101-train-particle-000-150-10",
            )}
            np.savez_compressed(path, **valid)
            validate_ticket37_maps_artifact(path)
            for bad in (
                {key: value.astype(np.float64) for key, value in valid.items()},
                {**valid, next(iter(valid)): np.full(shape, np.nan, dtype=np.float32)},
                {**valid, next(iter(valid)): np.full(shape, 1.1, dtype=np.float32)},
            ):
                np.savez_compressed(path, **bad)
                with self.assertRaises(ValueError):
                    validate_ticket37_maps_artifact(path)

    def test_json_artifacts_are_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            _write_json(path, {"value": 1})
            self.assertEqual(path.read_text(encoding="utf-8"), '{"value":1}\n')
            with self.assertRaises(FileExistsError):
                _write_json(path, {"value": 2})

    def test_runtime_contract_rejects_recipe_or_gate_drift(self) -> None:
        base = build_ticket37_sparse_objective_contract()
        _validate_runtime_contract(base)
        for mutate in (
            lambda value: value["recipe"].update(epochs=29),
            lambda value: value["recipe"].update(threshold=0.6),
            lambda value: value["evaluation"]["gates"].update(proposal_recall_min=0.9),
        ):
            changed = copy.deepcopy(base)
            mutate(changed)
            with self.assertRaises(ValueError):
                _validate_runtime_contract(changed)

    def test_model_score_output_cannot_silently_cast_float64(self) -> None:
        with self.assertRaises(ValueError):
            validate_ticket37_confidence_map(np.zeros((1536, 1536, 2), dtype=np.float64))

    def test_contract_freezes_official_imagenet_url_and_full_sha256(self) -> None:
        contract = build_ticket37_sparse_objective_contract()
        recipe = contract["recipe"]
        self.assertEqual(recipe["weights_url"], "https://download.pytorch.org/models/resnet18-f37072fd.pth")
        self.assertEqual(recipe["weights_sha256"], "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec")
        self.assertEqual(WEIGHTS_URL, recipe["weights_url"])
        self.assertEqual(WEIGHTS_SHA256, recipe["weights_sha256"])

    def test_deterministic_policy_is_exact_and_applied(self) -> None:
        self.assertEqual(
            DETERMINISTIC_POLICY,
            {
                "cublas_workspace_config": ":4096:8",
                "torch_deterministic_algorithms": True,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
            },
        )
        previous_env = __import__("os").environ.get("CUBLAS_WORKSPACE_CONFIG")
        previous_enabled = torch.are_deterministic_algorithms_enabled()
        previous_deterministic = torch.backends.cudnn.deterministic
        previous_benchmark = torch.backends.cudnn.benchmark
        try:
            applied = _establish_deterministic_policy(DETERMINISTIC_POLICY)
            self.assertEqual(applied, DETERMINISTIC_POLICY)
            self.assertEqual(__import__("os").environ.get("CUBLAS_WORKSPACE_CONFIG"), ":4096:8")
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cudnn.benchmark)
        finally:
            if previous_env is None:
                __import__("os").environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                __import__("os").environ["CUBLAS_WORKSPACE_CONFIG"] = previous_env
            torch.use_deterministic_algorithms(previous_enabled)
            torch.backends.cudnn.deterministic = previous_deterministic
            torch.backends.cudnn.benchmark = previous_benchmark

    def test_cached_imagenet_metadata_is_checked_against_contract(self) -> None:
        contract = build_ticket37_sparse_objective_contract()
        verified = _validate_cached_imagenet_weights(contract["recipe"])
        self.assertEqual(verified["url"], contract["recipe"]["weights_url"])
        self.assertEqual(verified["sha256"], contract["recipe"]["weights_sha256"])


if __name__ == "__main__":
    unittest.main()
