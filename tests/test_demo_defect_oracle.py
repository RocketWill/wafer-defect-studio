import unittest

from docs.demo.defect_oracle import DefectOracle, Particle, Scratch
from wafer_defect_studio.grid_geometry import annotation_grids


class DefectOracleTest(unittest.TestCase):
    def test_round_trip_and_closed_boundary_grid_truth(self):
        oracle = DefectOracle(
            image_width=20,
            image_height=20,
            defects=(
                Scratch("scratch", ((9, 2), (9, 8)), radius=1),
                Particle("particle", (5, 9), radius=1),
            ),
        )

        encoded = oracle.to_json()

        self.assertEqual(DefectOracle.from_json(encoded), oracle)
        self.assertEqual(encoded, oracle.to_json())
        self.assertEqual(
            oracle.grid_truth(annotation_grids(20, 20, 10, 10)),
            {
                (0, 0): ("particle", "scratch"),
                (0, 1): ("scratch",),
                (1, 0): ("particle",),
                (1, 1): (),
            },
        )


if __name__ == "__main__":
    unittest.main()
