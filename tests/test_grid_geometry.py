import unittest

from wafer_defect_studio.grid_geometry import AnnotationGrid, annotation_grids


class GridGeometryTest(unittest.TestCase):
    def test_full_intersecting_lattice_and_validation(self):
        grids = annotation_grids(23, 17, 10, 6, origin_x=3, origin_y=2)

        expected = (
            AnnotationGrid(-1, -1, -7, -4, 10, 6),
            AnnotationGrid(-1, 0, 3, -4, 10, 6),
            AnnotationGrid(-1, 1, 13, -4, 10, 6),
            AnnotationGrid(0, -1, -7, 2, 10, 6),
            AnnotationGrid(0, 0, 3, 2, 10, 6),
            AnnotationGrid(0, 1, 13, 2, 10, 6),
            AnnotationGrid(1, -1, -7, 8, 10, 6),
            AnnotationGrid(1, 0, 3, 8, 10, 6),
            AnnotationGrid(1, 1, 13, 8, 10, 6),
            AnnotationGrid(2, -1, -7, 14, 10, 6),
            AnnotationGrid(2, 0, 3, 14, 10, 6),
            AnnotationGrid(2, 1, 13, 14, 10, 6),
        )
        self.assertEqual(grids, expected)
        self.assertEqual(len(grids), 12)
        self.assertEqual((grids[0].center_x, grids[0].center_y), (-2.0, -1.0))
        self.assertEqual((grids[-1].center_x, grids[-1].center_y), (18.0, 17.0))
        for grid in grids:
            self.assertEqual((grid.width, grid.height), (10, 6))

        invalid = (
            (True, 17, 10, 6),
            (23, False, 10, 6),
            (23, 17, True, 6),
            (23, 17, 10, False),
            (0, 17, 10, 6),
            (23, -1, 10, 6),
            (23, 17, 0, 6),
            (23, 17, 10, 0),
            (23, 17, 10, 6, -1, 2),
            (23, 17, 10, 6, 10, 2),
            (23, 17, 10, 6, 1.5, 2),
            (23, 17, 10, 6, 3, True),
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    annotation_grids(*arguments)


if __name__ == "__main__":
    unittest.main()
