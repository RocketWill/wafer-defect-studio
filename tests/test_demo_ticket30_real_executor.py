import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from docs.demo.ticket30_evidence_corpus import build_ticket30_evidence_corpus
from docs.demo.ticket30_real_executor import _phase2_project_helpers, create_ticket30_real_executor
from docs.demo.wafer_quality_evidence import WaferEvidenceCase
from wafer_defect_studio.grid_geometry import annotation_grids


class _Backend:
    def __init__(self):
        self.calls = []

    def prepare(self, seed, source_image, work_dir):
        self.calls.append(("prepare", seed))
        return {"seed": seed}

    def train(self, context, run, destination, *, priority_ids=(), selection_sha256=None):
        self.calls.append(("train", tuple(priority_ids)))
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / ("refined.pt" if priority_ids else "probe.pt")
        path.write_bytes(b"checkpoint")
        return path

    def detect(self, context, checkpoint, split, cases, run, destination):
        self.calls.append(("detect", split))
        if cases is None:
            oracle = build_ticket30_evidence_corpus()[0].oracle
            case = type("Case", (), {"filename": f"{split}.png", "oracle": oracle})()
            case_split = split
        else:
            case = cases[0]
            case_split = "test"
        maps = np.zeros((1536, 1536, 2), dtype=np.float32)
        return (WaferEvidenceCase(case.filename, case_split, case.oracle,
                                  annotation_grids(1536, 1536, 512, 512), maps),)

    def training_bundle(self, context):
        return object()


class Ticket30RealExecutorTest(unittest.TestCase):
    def test_production_prepare_helpers_import_from_package_execution(self):
        seed_project, prepare_annotation, create_dataset = _phase2_project_helpers()

        self.assertEqual(seed_project.__name__, "_seed_project")
        self.assertEqual(prepare_annotation.__name__, "_prepare_annotation")
        self.assertEqual(create_dataset.__name__, "_create_dataset")

    def test_spatial_run_probes_mines_then_trains_fresh_and_holds_test_out(self):
        backend = _Backend()
        selection = {
            "selected": [{"bag_id": "bag-1"}],
        }
        with tempfile.TemporaryDirectory() as temporary, patch(
            "docs.demo.ticket30_real_executor.calibrate_map_thresholds",
            return_value={"selected": {"scratch": {"threshold": 0.5}, "particle": {"threshold": 0.5}}},
        ), patch(
            "docs.demo.ticket30_real_executor.select_hard_negative_bags", return_value=selection
        ), patch(
            "docs.demo.ticket30_real_executor.resolve_hard_negative_selection", return_value=("bag-1",)
        ):
            root = Path(temporary)
            source = root / "source.png"
            source.write_bytes(b"source")
            run = {"seed": 17, "model": "spatial_mil_v4", "score_domain": "absolute_spatial_probability",
                   "hard_negative_max_bags": 32, "detection_geometry": {"window_size": 128, "window_stride": 64}}
            executor = create_ticket30_real_executor(backend)
            for stage in ("train", "threshold", "map", "metrics"):
                executor(stage, run, root / "run", source)

        self.assertEqual(
            backend.calls,
            [("prepare", 17), ("train", ()), ("detect", "validation"),
             ("detect", "train"), ("train", ("bag-1",)),
             ("detect", "validation"), ("detect", "test")],
        )

    def test_baseline_trains_once_without_mining(self):
        backend = _Backend()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            source.write_bytes(b"source")
            run = {"seed": 17, "model": "cam_v2", "score_domain": "normalized_window_cam",
                   "detection_geometry": {"window_size": 512, "window_stride": 512}}
            executor = create_ticket30_real_executor(backend)
            executor("train", run, root / "run", source)

        self.assertEqual(backend.calls, [("prepare", 17), ("train", ())])


if __name__ == "__main__":
    unittest.main()
