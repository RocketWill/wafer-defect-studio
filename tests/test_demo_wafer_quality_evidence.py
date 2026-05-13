import unittest

import numpy

from docs.demo.defect_oracle import DefectOracle, Particle, Scratch
from docs.demo.wafer_quality_evidence import WaferEvidenceCase, compute_wafer_quality_evidence
from wafer_defect_studio.grid_geometry import annotation_grids


class WaferQualityEvidenceTest(unittest.TestCase):
    def test_two_class_two_image_metrics_include_closed_boundary_contact_and_nearest_rank_p95(self):
        grids = annotation_grids(4, 2, 2, 2)
        cases = (
            WaferEvidenceCase(
                "a.png",
                "validation",
                DefectOracle(4, 2, (Scratch("scratch", ((2, 0), (2, 1)), 0), Particle("particle", (0, 0), 0))),
                grids,
                numpy.array(
                    [
                        [[0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]],
                        [[0.1, 0.1], [0.1, 0.1], [0.1, 0.9], [0.1, 0.9]],
                    ]
                ),
            ),
            WaferEvidenceCase(
                "b.png",
                "validation",
                DefectOracle(4, 2, (Particle("particle", (3, 1), 0),)),
                grids,
                numpy.array(
                    [
                        [[numpy.nan, numpy.nan], [numpy.nan, numpy.nan], [0.1, 0.1], [0.1, 0.9]],
                        [[numpy.nan, numpy.nan], [numpy.nan, numpy.nan], [0.1, 0.1], [0.1, 0.9]],
                    ]
                ),
            ),
        )

        evidence = compute_wafer_quality_evidence(cases, ("scratch", "particle"), {"scratch": 0.5, "particle": 0.5})

        self.assertEqual(
            evidence["per_class"]["scratch"],
            {
                "defect_instances": 1,
                "covered_defect_instances": 1,
                "defect_coverage_recall": 1.0,
                "grid_tp": 1,
                "grid_fp": 0,
                "grid_fn": 1,
                "grid_precision": 1.0,
                "grid_recall": 0.5,
                "normal_grid_leaks": 0,
                "normal_grids": 2,
                "normal_grid_leak_rate": 0.0,
                "asserted_grid_occupancy_p95": 0.25,
            },
        )
        self.assertEqual(evidence["per_class"]["particle"]["defect_instances"], 2)
        self.assertEqual(evidence["per_class"]["particle"]["covered_defect_instances"], 2)
        self.assertEqual(evidence["per_class"]["particle"]["grid_tp"], 2)
        self.assertEqual(evidence["per_class"]["particle"]["grid_fp"], 1)
        self.assertEqual(evidence["per_class"]["particle"]["normal_grid_leak_rate"], 0.5)
        self.assertEqual(evidence["per_class"]["particle"]["asserted_grid_occupancy_p95"], 0.5)

    def test_shape_and_class_errors_include_case_context(self):
        case = WaferEvidenceCase(
            "bad.png",
            "validation",
            DefectOracle(4, 2, ()),
            annotation_grids(4, 2, 2, 2),
            numpy.zeros((2, 4, 1)),
        )

        with self.assertRaisesRegex(ValueError, "bad.png.*class.*2.*1"):
            compute_wafer_quality_evidence((case,), ("scratch", "particle"), {"scratch": 0.5, "particle": 0.5})

        wrong_shape = WaferEvidenceCase(
            "shape.png",
            "validation",
            case.oracle,
            case.grids,
            numpy.zeros((1, 4, 2)),
        )
        with self.assertRaisesRegex(ValueError, "shape.png.*map shape.*2, 4, 2.*1, 4, 2"):
            compute_wafer_quality_evidence(
                (wrong_shape,),
                ("scratch", "particle"),
                {"scratch": 0.5, "particle": 0.5},
            )

    def test_zero_denominators_are_not_reported_as_perfect_scores(self):
        case = WaferEvidenceCase(
            "empty.png",
            "validation",
            DefectOracle(2, 2, ()),
            annotation_grids(2, 2, 2, 2),
            numpy.zeros((2, 2, 1)),
        )

        metrics = compute_wafer_quality_evidence((case,), ("scratch",), {"scratch": 0.5})["per_class"]["scratch"]

        self.assertIsNone(metrics["defect_coverage_recall"])
        self.assertIsNone(metrics["grid_precision"])
        self.assertIsNone(metrics["grid_recall"])
        self.assertEqual(metrics["normal_grid_leak_rate"], 0.0)
        self.assertIsNone(metrics["asserted_grid_occupancy_p95"])

    def test_asserted_false_negative_contributes_zero_occupancy(self):
        case = WaferEvidenceCase(
            "miss.png",
            "validation",
            DefectOracle(2, 2, (Particle("particle", (1, 1), 0),)),
            annotation_grids(2, 2, 2, 2),
            numpy.zeros((2, 2, 1)),
        )

        metrics = compute_wafer_quality_evidence((case,), ("particle",), {"particle": 0.5})["per_class"]["particle"]

        self.assertEqual(metrics["grid_fn"], 1)
        self.assertEqual(metrics["asserted_grid_occupancy_p95"], 0.0)


if __name__ == "__main__":
    unittest.main()
