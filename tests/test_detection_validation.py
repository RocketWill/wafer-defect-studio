import unittest

from wafer_defect_studio.annotation import GridAnnotation
from wafer_defect_studio.detection_windows import Rect
from wafer_defect_studio.detection_validation import compare_detection_proposals
from wafer_defect_studio.grid_geometry import AnnotationGrid
from wafer_defect_studio.proposal_generation import DefectProposal


class DetectionValidationTest(unittest.TestCase):
    def test_compares_proposal_cells_with_multilabel_annotations(self):
        grids = tuple(
            AnnotationGrid(row, column, column * 4, row * 4, 4, 4)
            for row in range(2)
            for column in range(2)
        )
        annotations = (
            GridAnnotation("image-1", 0, 0, ("scratch",)),
            GridAnnotation("image-1", 0, 1, ("particle",)),
        )
        proposals = (
            DefectProposal("p-scratch", "scratch", Rect(0, 0, 2, 2), 4, 0.9, 0.8, {}),
            DefectProposal("p-particle", "particle", Rect(4, 0, 4, 4), 16, 0.9, 0.8, {}),
            DefectProposal("p-false-positive", "scratch", Rect(0, 4, 1, 1), 1, 0.8, 0.8, {}),
        )

        report = compare_detection_proposals(
            annotations,
            proposals,
            grids,
            class_names=("scratch", "particle"),
        )

        self.assertEqual(report.evaluated_cells, 4)
        self.assertEqual(report.exact_cell_matches, 3)
        self.assertEqual(report.for_class("scratch").true_positive, 1)
        self.assertEqual(report.for_class("scratch").false_positive, 1)
        self.assertEqual(report.for_class("scratch").false_negative, 0)
        self.assertEqual(report.for_class("particle").true_positive, 1)
        self.assertEqual(report.for_class("particle").false_positive, 0)
        self.assertEqual(report.for_class("particle").false_negative, 0)
        self.assertAlmostEqual(report.for_class("particle").f1, 1.0)


if __name__ == "__main__":
    unittest.main()
