import unittest

from wafer_defect_studio.detection_windows import Rect
from wafer_defect_studio.grid_geometry import AnnotationGrid
from wafer_defect_studio.proposal_generation import DefectProposal
from wafer_defect_studio.proposal_queue import ReviewQueueItem
from wafer_defect_studio.proposal_conversion import preview_proposal_conversion


class ProposalConversionTest(unittest.TestCase):
    def test_preview_maps_positive_intersections_and_preserves_provenance(self):
        grids = (
            AnnotationGrid(1, 1, 4, 4, 4, 4),
            AnnotationGrid(0, 1, 4, 0, 4, 4),
            AnnotationGrid(1, 0, 0, 4, 4, 4),
            AnnotationGrid(0, 0, 0, 0, 4, 4),
        )
        p2 = DefectProposal(
            "p-2",
            "particle",
            Rect(3, 3, 3, 3),
            9,
            0.90,
            0.80,
            {"run_id": "run-1", "profile_id": "profile-1"},
        )
        p1 = DefectProposal(
            "p-1",
            "scratch",
            Rect(0, 0, 2, 2),
            4,
            0.95,
            0.85,
            {"run_id": "run-1", "profile_id": "profile-1"},
        )
        corrected = ReviewQueueItem(
            proposal=DefectProposal(
                "p-3",
                "crack",
                Rect(20, 20, 2, 2),
                4,
                0.80,
                0.70,
                {"run_id": "run-1", "profile_id": "profile-1"},
            ),
            status="corrected",
            source_rect=Rect(4, 4, 1, 1),
        )

        preview = preview_proposal_conversion((p2, corrected, p1), grids)

        self.assertEqual(
            [(cell.row, cell.column) for cell in preview.cells],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
        )
        self.assertEqual(preview.cells[0].class_codes, ("particle", "scratch"))
        self.assertEqual(preview.cells[0].proposal_ids, ("p-1", "p-2"))
        self.assertEqual(preview.cells[1].class_codes, ("particle",))
        self.assertEqual(preview.cells[1].proposal_ids, ("p-2",))
        self.assertEqual(preview.cells[2].class_codes, ("particle",))
        self.assertEqual(preview.cells[2].proposal_ids, ("p-2",))
        self.assertEqual(preview.cells[3].class_codes, ("crack", "particle"))
        self.assertEqual(preview.cells[3].proposal_ids, ("p-2", "p-3"))
        self.assertEqual(preview.source_proposal_ids, ("p-1", "p-2", "p-3"))
        self.assertEqual(
            dict(preview.provenance),
            {
                "source_coordinate_system": "source-image-pixels",
                "run_id": "run-1",
                "profile_id": "profile-1",
            },
        )

    def test_preview_requires_selected_inputs_and_reviewed_queue_items(self):
        grid = AnnotationGrid(0, 0, 0, 0, 4, 4)
        proposal = DefectProposal("p-1", "scratch", Rect(0, 0, 1, 1), 1, 0.8, 0.8, {})
        rejected = ReviewQueueItem(proposal, "rejected", proposal.source_rect)

        with self.assertRaises(ValueError):
            preview_proposal_conversion((), (grid,))
        with self.assertRaises(ValueError):
            preview_proposal_conversion((proposal,), ())
        with self.assertRaises(ValueError):
            preview_proposal_conversion((rejected,), (grid,))
        with self.assertRaises(TypeError):
            preview_proposal_conversion((object(),), (grid,))


if __name__ == "__main__":
    unittest.main()
