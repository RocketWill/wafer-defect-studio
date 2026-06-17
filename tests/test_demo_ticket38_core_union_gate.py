import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from docs.demo.ticket38_core_union_gate import (
    CANDIDATE_ARTIFACT_ROOT,
    DETERMINISTIC_POLICY,
    WEIGHTS_SHA256,
    WEIGHTS_URL,
    _build_configuration,
    _build_report,
    _row_passes,
    _validate_matching_artifact,
    _validate_output_targets,
    _validate_runtime_contract,
    _verify_ticket38_artifact_bytes,
    compute_ticket38_objective,
    evaluate_ticket38_response,
    seal_ticket38_artifacts,
    validate_ticket38_maps_artifact,
    validate_ticket38_confidence_map,
)
from docs.demo.ticket38_core_union_gate_contract import (
    CASE_IDS,
    build_ticket38_core_union_gate_contract,
)
from wafer_defect_studio.detection_windows import Rect
from wafer_defect_studio.spatial_mil import (
    core_union_coverage_loss,
    dense_absent_class_loss,
    far_negative_suppression_loss,
    overlap_consistency_loss,
    same_image_grid_ranking_loss,
)


class Ticket38CoreUnionGateTest(unittest.TestCase):
    def test_objective_reuses_core_union_and_context_terms(self) -> None:
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
            logits, far_masks, kwargs["far_batch_indices"], kwargs["far_class_indices"], hardest_fraction=0.01
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
                [overlap_consistency_loss(logits[index : index + 1], rects[index], feature_stride=2) for index in range(2)]
            ).mean()
        )
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_response_records_exact_truth_proposal_and_match_counts(self) -> None:
        for count in (1, 8, 42):
            side = 64
            rendered = _rendered_truth(count, side)
            response = np.zeros((side, side), dtype=np.float32)
            for index in range(count):
                x = 2 + (index % 7) * 10
                y = 2 + (index // 7) * 10
                response[y, x] = 1.0
            row, matching = evaluate_ticket38_response(response, rendered)
            self.assertEqual(
                (row["truth_component_count"], row["proposal_count"], row["matched_count"]),
                (count, count, count),
            )
            self.assertEqual(
                (matching["truth_component_count"], matching["proposal_count"], matching["matched_count"]),
                (count, count, count),
            )

    def test_row_gate_rejects_count_drift(self) -> None:
        row = {
            "case_id": CASE_IDS[0],
            "truth_component_count": 2,
            "proposal_count": 1,
            "matched_count": 1,
            "proposal_precision": 1.0,
            "proposal_recall": 1.0,
            "unique_truth_touch_recall": 1.0,
            "unmatched_truth_component_ids": [],
            "unmatched_proposal_ids": [],
            "cross_component_merge_proposal_ids": [],
            "normal_grid_leaks": 0,
            "asserted_grid_occupancy_p95": 0.01,
        }
        self.assertFalse(_row_passes(row))
        row.update(truth_component_count=1, normal_grid_leak_rate=0.01)
        self.assertFalse(_row_passes(row))

    def test_runtime_contract_freezes_four_cases(self) -> None:
        contract = build_ticket38_core_union_gate_contract()
        _validate_runtime_contract(contract)
        changed = copy.deepcopy(contract)
        changed["membership"]["case_ids"].reverse()
        with self.assertRaises(ValueError):
            _validate_runtime_contract(changed)

    def test_output_targets_and_artifact_seal_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            root = repo / CANDIDATE_ARTIFACT_ROOT
            root.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                _validate_output_targets(repo, repo / "docs/demo/ticket38-core-union-gate.json")
            root.rmdir()
            report = repo / "docs/demo/ticket38-core-union-gate.json"
            report.parent.mkdir(parents=True)
            report.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _validate_output_targets(repo, report)

            root.mkdir(parents=True)
            for name in ("checkpoint.pt", "confidence-maps.npz", "proposal-matching.json", "configuration.json"):
                (root / name).write_bytes(name.encode("ascii"))
            seal = seal_ticket38_artifacts(root, repo_root=repo)
            _verify_ticket38_artifact_bytes(seal, repo)
            target = root / "configuration.json"
            target.write_bytes(target.read_bytes() + b"tamper")
            with self.assertRaises(ValueError):
                _verify_ticket38_artifact_bytes(seal, repo)

    def test_maps_require_exact_four_float32_case_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "maps.npz"
            shape = (1536, 1536, 2)
            valid = {case_id: np.zeros(shape, dtype=np.float32) for case_id in CASE_IDS}
            np.savez_compressed(path, **valid)
            validate_ticket38_maps_artifact(path)
            bad = dict(valid)
            bad.pop(CASE_IDS[-1])
            bad["unexpected"] = np.zeros(shape, dtype=np.float32)
            np.savez_compressed(path, **bad)
            with self.assertRaises(ValueError):
                validate_ticket38_maps_artifact(path)
            with self.assertRaises(ValueError):
                validate_ticket38_confidence_map(np.zeros((1536, 1536, 2), dtype=np.float64))

    def test_matching_membership_accepts_canonical_sort_key_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matching.json"
            expected_counts = dict(zip(CASE_IDS, (1, 8, 1, 42)))
            per_case = {
                case_id: {
                    "scratch" if "scratch" in case_id else "particle": {
                        "truth_component_count": expected_counts[case_id],
                        "proposal_count": expected_counts[case_id],
                        "matched_count": expected_counts[case_id],
                        "proposal_precision": 1.0,
                        "proposal_recall": 1.0,
                        "unique_truth_touch_recall": 1.0,
                    }
                }
                for case_id in reversed(CASE_IDS)
            }
            payload = {"schema": "ticket38-core-union-gate.matching.v1", "case_order": list(CASE_IDS), "per_case": per_case}
            path.write_text(json.dumps(payload), encoding="utf-8")
            _validate_matching_artifact(path)

    def test_gate_artifacts_record_gate_run_count(self) -> None:
        contract = build_ticket38_core_union_gate_contract()
        report = _build_report(
            contract,
            ({"overall": "PASS"},),
            {},
            {"root": CANDIDATE_ARTIFACT_ROOT},
            deterministic_policy=DETERMINISTIC_POLICY,
        )
        self.assertEqual(report["execution"]["gate_run_count"], 1)
        self.assertNotIn("probe_run_count", report["execution"])
        self.assertNotIn("formal_run_count", report["execution"])

        with patch("docs.demo.ticket38_core_union_gate.torch.cuda.get_device_name", return_value="cpu-test"):
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
        self.assertEqual(configuration["gate_run_count"], 1)
        self.assertNotIn("probe_run_count", configuration)


def _rendered_truth(count: int, side: int) -> dict[str, object]:
    unique = []
    components = []
    for index in range(count):
        x = 2 + (index % 7) * 10
        y = 2 + (index // 7) * 10
        pixel = [[x, y]]
        unique.append({"truth_id": index, "instance_ids": [str(index)], "pixels": pixel, "area": 1})
        components.append({"component_id": index, "truth_ids": [index], "instance_ids": [str(index)], "pixels": pixel, "area": 1})
    return {
        "schema": "ticket37-proposal-evaluation.v1",
        "response_shape": [side, side],
        "unique_truth_count": count,
        "truth_component_count": count,
        "unique_truths": unique,
        "truth_components": components,
    }


if __name__ == "__main__":
    unittest.main()
