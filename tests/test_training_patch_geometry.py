import unittest

from wafer_defect_studio.detection_windows import Rect
from wafer_defect_studio.training_dataset import enumerate_model_patch_rects
from wafer_defect_studio.training_protocol import TrainingConfig


class TrainingPatchGeometryTest(unittest.TestCase):
    def test_rectangles_are_absolute_row_major_and_cover_divisible_or_edge_aligned_grids(self):
        config = TrainingConfig(
            snapshot_id="snapshot-1",
            split_id="split-1",
            class_count=2,
            epochs=1,
            batch_size=1,
            patch_size=128,
            patch_stride=64,
        )

        divisible = enumerate_model_patch_rects(Rect(100, 200, 512, 512), config)
        self.assertEqual(len(divisible), 49)
        self.assertEqual(divisible[:8], tuple(
            Rect(x, y, 128, 128)
            for x, y in (
                (100, 200), (164, 200), (228, 200), (292, 200),
                (356, 200), (420, 200), (484, 200), (100, 264),
            )
        ))
        self.assertEqual(divisible[-1], Rect(484, 584, 128, 128))

        grid = Rect(11, 23, 500, 450)
        edge_aligned = enumerate_model_patch_rects(grid, config)
        x_starts = tuple(dict.fromkeys(rect.x for rect in edge_aligned))
        y_starts = tuple(dict.fromkeys(rect.y for rect in edge_aligned))
        self.assertEqual(x_starts, (11, 75, 139, 203, 267, 331, 383))
        self.assertEqual(y_starts, (23, 87, 151, 215, 279, 343, 345))
        self.assertEqual(len(edge_aligned), len(set(edge_aligned)))
        self.assertTrue(all(
            grid.x <= rect.x and rect.right <= grid.right
            and grid.y <= rect.y and rect.bottom <= grid.bottom
            for rect in edge_aligned
        ))
        self.assertTrue(all(right <= left + 128 for left, right in zip(x_starts, x_starts[1:])))
        self.assertTrue(all(bottom <= top + 128 for top, bottom in zip(y_starts, y_starts[1:])))
        self.assertEqual(edge_aligned[-1].right, grid.right)
        self.assertEqual(edge_aligned[-1].bottom, grid.bottom)


if __name__ == "__main__":
    unittest.main()
