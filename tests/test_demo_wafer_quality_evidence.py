import unittest

import numpy

from docs.demo.defect_oracle import DefectOracle, Particle, Scratch, defect_intersects_rectangle
from docs.demo.wafer_quality_evidence import (
    WaferEvidenceCase,
    compute_wafer_quality_evidence,
    quality_relevant_threshold_candidates,
)
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

    def test_quality_candidates_are_only_grid_and_exact_defect_support_maxima(self):
        case = WaferEvidenceCase(
            "large.png",
            "validation",
            DefectOracle(
                1536,
                1536,
                (
                    Scratch("scratch", ((10, 10), (20, 10)), 0),
                    Particle("scratch", (700, 700), 1),
                ),
            ),
            annotation_grids(1536, 1536, 512, 512),
            numpy.linspace(0.0, 1.0, 1536 * 1536, dtype=numpy.float32).reshape(1536, 1536, 1),
        )

        candidates = quality_relevant_threshold_candidates((case,), 0)

        self.assertLessEqual(len(candidates), 2 + len(case.grids) + len(case.oracle.defects))
        self.assertEqual(candidates, tuple(sorted(set(candidates))))
        self.assertEqual(candidates[0], 0.0)
        self.assertEqual(candidates[-1], 1.0)

    def test_optimized_coverage_matches_naive_pixel_scan_for_each_defect_kind_and_boundary(self):
        grids = annotation_grids(8, 4, 4, 4)
        defects = (
            Scratch("scratch", ((0, 0), (3, 2)), 1),
            Particle("particle", (7, 3), 1),
            Particle("particle", (4, 2), 0),
        )
        values = numpy.zeros((4, 8, 2))
        values[0, 0, 0] = 0.9
        values[3, 7, 1] = 0.9
        values[2, 3, 1] = 0.9  # closed pixel rectangle contacts the point at x=4
        case = WaferEvidenceCase("boundary.png", "validation", DefectOracle(8, 4, defects), grids, values)

        result = compute_wafer_quality_evidence(
            (case,), ("scratch", "particle"), {"scratch": 0.5, "particle": 0.5}
        )
        for code, class_index in (("scratch", 0), ("particle", 1)):
            retained = numpy.isfinite(values[:, :, class_index]) & (values[:, :, class_index] >= 0.5)
            expected = sum(
                any(
                    retained[y, x] and defect_intersects_rectangle(defect, x, y, x + 1, y + 1)
                    for y in range(4)
                    for x in range(8)
                )
                for defect in defects
                if defect.class_code == code
            )
            self.assertEqual(result["per_class"][code]["covered_defect_instances"], expected)


if __name__ == "__main__":
    unittest.main()
