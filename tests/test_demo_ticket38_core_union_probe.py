import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from docs.demo.ticket38_core_union_probe import (
    CANDIDATE_ARTIFACT_ROOT,
    DETERMINISTIC_POLICY,
    WEIGHTS_SHA256,
    WEIGHTS_URL,
    _validate_output_targets,
    _validate_matching_artifact,
    _validate_runtime_contract,
    _verify_ticket38_artifact_bytes,
    _build_configuration,
    _build_report,
    _write_json,
    compute_ticket38_objective,
    evaluate_ticket38_response,
    seal_ticket38_artifacts,
    validate_ticket38_confidence_map,
    validate_ticket38_maps_artifact,
)
from docs.demo.ticket38_core_union_probe_contract import (
    CASE_IDS,
    build_ticket38_core_union_probe_contract,
)
from wafer_defect_studio.detection_windows import Rect
from wafer_defect_studio.spatial_mil import (
    core_union_coverage_loss,
    dense_absent_class_loss,
    far_negative_suppression_loss,
    overlap_consistency_loss,
    same_image_grid_ranking_loss,
)


class Ticket38CoreUnionProbeTest(unittest.TestCase):
    def test_objective_uses_core_union_and_context_terms(self) -> None:
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
        loss = compute_ticket38_objective(**kwargs)

        core = core_union_coverage_loss(logits, instance_masks, batch_indices, class_indices)
        far = far_negative_suppression_loss(
            logits,
            far_masks,
            kwargs["far_batch_indices"],
            kwargs["far_class_indices"],
            hardest_fraction=0.01,
        )
        local = far_negative_suppression_loss(
            logits,
            far_masks[~far_masks.flatten(start_dim=1).all(dim=1)],
            kwargs["far_batch_indices"][~far_masks.flatten(start_dim=1).all(dim=1)],
            kwargs["far_class_indices"][~far_masks.flatten(start_dim=1).all(dim=1)],
            hardest_fraction=1.0,
        )
        bag_logits = logits.reshape(2, 1, 2, 2, 2)
        expected = (
            core
            + far
            + local
            + 0.5 * dense_absent_class_loss(bag_logits, targets)
            + 0.25 * same_image_grid_ranking_loss(bag_logits, targets, ("image", "image"))
            + 0.10 * torch.stack(
                [
                    overlap_consistency_loss(logits[index : index + 1], rects[index], feature_stride=2)
                    for index in range(2)
                ]
            ).mean()
        )
        self.assertTrue(torch.allclose(loss, expected))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_objective_duplicate_core_mask_preserves_loss_and_gradient(self) -> None:
        base = torch.zeros((2, 2, 2, 2), requires_grad=True)
        mask = torch.tensor([[[True, False], [False, False]]])
        common = dict(
            targets=torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
            image_group_ids=("image", "image"),
            far_negative_masks=torch.ones((2, 2, 2), dtype=torch.bool),
            far_batch_indices=torch.tensor([0, 1], dtype=torch.long),
            far_class_indices=torch.tensor([0, 1], dtype=torch.long),
            patch_rect_groups=[[Rect(0, 0, 4, 4)], [Rect(0, 0, 4, 4)]],
        )
        one = compute_ticket38_objective(
            logits=base,
            instance_masks=mask,
            instance_batch_indices=torch.tensor([0]),
            instance_class_indices=torch.tensor([0]),
            **common,
        )
        one.backward()
        one_grad = base.grad.detach().clone()

        duplicate_logits = base.detach().clone().requires_grad_()
        duplicate = compute_ticket38_objective(
            logits=duplicate_logits,
            instance_masks=mask.repeat(2, 1, 1),
            instance_batch_indices=torch.tensor([0, 0]),
            instance_class_indices=torch.tensor([0, 0]),
            **common,
        )
        duplicate.backward()

        torch.testing.assert_close(duplicate, one)
        torch.testing.assert_close(duplicate_logits.grad, one_grad)

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
        row, _matching = evaluate_ticket38_response(
            response,
            rendered,
            remote_response={"numerator": 100, "denominator": 100, "ratio": 1.0},
        )
        self.assertEqual(row["overall"], "PASS")
        self.assertEqual(row["remote_response"]["ratio"], 1.0)

        merged = np.zeros((32, 32), dtype=np.float32)
        merged[1:21, 1:21] = 1.0
        merged_row, _ = evaluate_ticket38_response(merged, rendered)
        self.assertEqual(merged_row["overall"], "FAIL")
        self.assertEqual(merged_row["cross_component_merge_proposal_ids"], [0])

    def test_seal_re_read_and_tamper_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            root = repo / CANDIDATE_ARTIFACT_ROOT
            root.mkdir(parents=True)
            for name in ("checkpoint.pt", "confidence-maps.npz", "proposal-matching.json", "configuration.json"):
                (root / name).write_bytes(name.encode("ascii"))
            seal = seal_ticket38_artifacts(root, repo_root=repo)
            _verify_ticket38_artifact_bytes(seal, repo)
            target = root / "configuration.json"
            target.write_bytes(target.read_bytes() + b"tamper")
            with self.assertRaises(ValueError):
                _verify_ticket38_artifact_bytes(seal, repo)

    def test_preflight_output_targets_reject_existing_root_or_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            (repo / CANDIDATE_ARTIFACT_ROOT).mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                _validate_output_targets(repo, repo / "docs/demo/ticket38-core-union-probe.json")
            (repo / CANDIDATE_ARTIFACT_ROOT).rmdir()
            report = repo / "docs/demo/ticket38-core-union-probe.json"
            report.parent.mkdir(parents=True)
            report.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _validate_output_targets(repo, report)

    def test_stored_maps_require_float32_finite_unit_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "maps.npz"
            shape = (1536, 1536, 2)
            valid = {case_id: np.zeros(shape, dtype=np.float32) for case_id in CASE_IDS}
            np.savez_compressed(path, **valid)
            validate_ticket38_maps_artifact(path)
            for bad in (
                {key: value.astype(np.float64) for key, value in valid.items()},
                {**valid, CASE_IDS[0]: np.full(shape, np.nan, dtype=np.float32)},
                {**valid, CASE_IDS[0]: np.full(shape, 1.1, dtype=np.float32)},
            ):
                np.savez_compressed(path, **bad)
                with self.assertRaises(ValueError):
                    validate_ticket38_maps_artifact(path)

    def test_json_artifacts_are_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            _write_json(path, {"value": 1})
            self.assertEqual(path.read_text(encoding="utf-8"), '{"value":1}\n')
            with self.assertRaises(FileExistsError):
                _write_json(path, {"value": 2})

    def test_canonical_json_case_mapping_order_is_not_semantic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matching.json"
            per_case = {
                CASE_IDS[0]: {
                    "scratch": {
                        "proposal_precision": 1.0,
                        "proposal_recall": 1.0,
                        "unique_truth_touch_recall": 1.0,
                    }
                },
                CASE_IDS[1]: {
                    "particle": {
                        "proposal_precision": 1.0,
                        "proposal_recall": 1.0,
                        "unique_truth_touch_recall": 1.0,
                    }
                },
            }
            _write_json(
                path,
                {
                    "schema": "ticket38-core-union-probe.matching.v1",
                    "case_order": list(CASE_IDS),
                    "per_case": per_case,
                },
            )

            self.assertNotEqual(tuple(json.loads(path.read_text())["per_case"]), CASE_IDS)
            _validate_matching_artifact(path)

    def test_runtime_contract_rejects_recipe_or_gate_drift(self) -> None:
        base = build_ticket38_core_union_probe_contract()
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
            validate_ticket38_confidence_map(np.zeros((1536, 1536, 2), dtype=np.float64))

    def test_contract_freezes_official_imagenet_url_and_full_sha256(self) -> None:
        recipe = build_ticket38_core_union_probe_contract()["recipe"]
        self.assertEqual(recipe["weights_url"], WEIGHTS_URL)
        self.assertEqual(recipe["weights_sha256"], WEIGHTS_SHA256)

    def test_deterministic_policy_is_exact(self) -> None:
        self.assertEqual(
            DETERMINISTIC_POLICY,
            {
                "cublas_workspace_config": ":4096:8",
                "torch_deterministic_algorithms": True,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
            },
        )

    def test_probe_artifacts_record_probe_run_count(self) -> None:
        contract = build_ticket38_core_union_probe_contract()
        report = _build_report(
            contract,
            ({"overall": "PASS"},),
            {},
            {"root": CANDIDATE_ARTIFACT_ROOT},
            deterministic_policy=DETERMINISTIC_POLICY,
        )
        self.assertEqual(report["execution"]["probe_run_count"], 1)
        self.assertNotIn("formal_run_count", report["execution"])

        with patch("docs.demo.ticket38_core_union_probe.torch.cuda.get_device_name", return_value="cpu-test"):
            configuration = _build_configuration(
                contract,
                "0" * 40,
                {case_id: [0.0] * 30 for case_id in CASE_IDS},
                imagenet_weights={
                    "enum": "ResNet18_Weights.IMAGENET1K_V1",
                    "url": WEIGHTS_URL,
                    "path": "cached-resnet18.pth",
                    "sha256": WEIGHTS_SHA256,
                },
                deterministic_policy=DETERMINISTIC_POLICY,
            )
        self.assertEqual(configuration["probe_run_count"], 1)
        self.assertNotIn("formal_run_count", configuration)


if __name__ == "__main__":
    unittest.main()
