import copy
import unittest

from docs.demo.ticket30_cuda_evidence import (
    EXPECTED_GPU,
    build_ticket30_cuda_evidence_plan,
    hash_ticket30_cuda_evidence_manifest,
    serialize_ticket30_cuda_evidence_manifest,
    validate_completed_ticket30_cuda_evidence_manifest,
)
from docs.demo.ticket30_evidence_corpus import FROZEN_CORPUS_SHA256


def _completed_manifest():
    manifest = build_ticket30_cuda_evidence_plan("a" * 40)
    for run in manifest["runs"]:
        names = ["checkpoint", "threshold", "map", "metrics"]
        if run["model"] == "spatial_mil_v4":
            names.append("hard_negative_selection")
        run["artifacts"] = {
            name: {"path": f"{run['seed']}/{run['model']}/{name}.json", "sha256": "b" * 64}
            for name in names
        }
    return manifest


class Ticket30CudaEvidenceTest(unittest.TestCase):
    def test_builds_frozen_matched_cuda_plan(self):
        manifest = _completed_manifest()

        self.assertEqual(manifest["corpus_sha256"], FROZEN_CORPUS_SHA256)
        self.assertEqual(manifest["device"], "cuda")
        self.assertEqual(manifest["gpu"], EXPECTED_GPU)
        self.assertEqual(manifest["stage_order"], ["train", "threshold", "map", "metrics"])
        self.assertEqual(
            [(run["seed"], run["model"]) for run in manifest["runs"]],
            [(seed, model) for seed in (17, 42, 91) for model in ("cam_v2", "patch_v3", "spatial_mil_v4")],
        )
        self.assertEqual(
            {run["model"]: run["score_domain"] for run in manifest["runs"]},
            {
                "cam_v2": "normalized_window_cam",
                "patch_v3": "patch_sigmoid",
                "spatial_mil_v4": "absolute_spatial_probability",
            },
        )
        self.assertTrue(all(run["epochs"] == 20 and run["weights"] == "imagenet" for run in manifest["runs"]))
        self.assertEqual(
            manifest["replay_config"],
            {
                "source_asset": {
                    "path": "docs/demo/assets/realistic-wafer-20mp.png",
                    "sha256": "6cc5df01674a94458355aa7993bc458edffc39636d22c497b7d7bb2eda2a1e6b",
                },
                "grid": {"width": 512, "height": 512},
                "training": {"batch_size": 4, "learning_rate": 0.001},
                "detection": {"reflect_padding": True, "center_weighting": "linear"},
            },
        )
        self.assertEqual(
            {run["model"]: run["detection_geometry"] for run in manifest["runs"]},
            {
                "cam_v2": {"window_size": 512, "window_stride": 512},
                "patch_v3": {"window_size": 128, "window_stride": 64},
                "spatial_mil_v4": {"window_size": 128, "window_stride": 64},
            },
        )
        spatial = next(run for run in manifest["runs"] if run["model"] == "spatial_mil_v4")
        self.assertEqual(spatial["hard_negative_max_bags"], 32)
        self.assertEqual(spatial["refinement_semantics"], "selection_probe_then_fresh_20_plus_5")
        self.assertEqual(validate_completed_ticket30_cuda_evidence_manifest(manifest), manifest)
        serialized = serialize_ticket30_cuda_evidence_manifest(manifest)
        self.assertEqual(serialized, serialize_ticket30_cuda_evidence_manifest(copy.deepcopy(manifest)))
        self.assertEqual(len(hash_ticket30_cuda_evidence_manifest(manifest)), 64)

    def test_rejects_frozen_contract_and_artifact_drift(self):
        valid = _completed_manifest()
        mutations = (
            (lambda value: value.update(device="cpu"), "CUDA"),
            (lambda value: value.update(gpu="other"), "GPU"),
            (lambda value: value.update(corpus_sha256="c" * 64), "corpus"),
            (lambda value: value.update(class_codes=["particle", "scratch"]), "class"),
            (lambda value: value["runs"].pop(), "run"),
            (lambda value: value["runs"].__setitem__(1, copy.deepcopy(value["runs"][0])), "run"),
            (lambda value: value["runs"][0].update(seed=18), "run"),
            (lambda value: value["runs"][0].update(model="other"), "run"),
            (lambda value: value["runs"][0].update(score_domain="absolute_spatial_probability"), "score domain"),
            (lambda value: value["replay_config"]["source_asset"].update(sha256="c" * 64), "replay config"),
            (lambda value: value["replay_config"]["training"].update(batch_size=8), "replay config"),
            (lambda value: value["replay_config"]["training"].update(learning_rate=0.01), "replay config"),
            (lambda value: value["runs"][0]["detection_geometry"].update(window_stride=256), "detection geometry"),
            (lambda value: value["runs"][-1].update(hard_negative_max_bags=16), "hard negative max bags"),
            (lambda value: value["runs"][-1].update(refinement_semantics="resume_20_plus_5"), "refinement semantics"),
            (lambda value: value["runs"][0]["artifacts"]["metrics"].update(sha256="bad"), "artifact"),
            (lambda value: value["runs"][0]["artifacts"].pop("metrics"), "artifact"),
            (lambda value: value["runs"][0]["artifacts"].update(hard_negative_selection={"path": "x", "sha256": "b" * 64}), "artifact"),
            (lambda value: value["runs"][-1]["artifacts"].pop("hard_negative_selection"), "artifact"),
            (lambda value: value.update(extra="unexpected"), "manifest fields"),
            (lambda value: value["runs"][0].update(extra="unexpected"), "run fields"),
            (
                lambda value: value["runs"][1]["artifacts"]["checkpoint"].update(
                    path=value["runs"][0]["artifacts"]["checkpoint"]["path"]
                ),
                "duplicate artifact path",
            ),
        )
        for mutate, message in mutations:
            manifest = copy.deepcopy(valid)
            mutate(manifest)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_completed_ticket30_cuda_evidence_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
