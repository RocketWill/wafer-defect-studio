import hashlib
import unittest

import numpy as np

from docs.demo.ticket31_contract import DEVELOPMENT_SEEDS, FINAL_MEMBERS
from docs.demo.ticket31_development_corpus import (
    build_ticket31_development_corpus,
    development_grid_annotations,
    render_ticket31_development_pixels,
)


class Ticket31DevelopmentCorpusTest(unittest.TestCase):
    def test_freezes_position_diverse_grid_only_development_cases(self) -> None:
        corpus = build_ticket31_development_corpus()

        self.assertEqual(len(corpus), 90)
        self.assertEqual(tuple(dict.fromkeys(case.seed for case in corpus)), DEVELOPMENT_SEEDS)
        self.assertFalse({case.filename for case in corpus} & set(FINAL_MEMBERS))
        self.assertEqual(len({case.filename for case in corpus}), len(corpus))
        self.assertEqual(
            len({instance.instance_id for case in corpus for instance in case.instances}),
            sum(len(case.instances) for case in corpus),
        )

        expected_grids = {(row, column) for row in range(3) for column in range(3)}
        for seed in DEVELOPMENT_SEEDS:
            for split in ("train", "validation"):
                cases = tuple(
                    case for case in corpus if case.seed == seed and case.split == split
                )
                self.assertEqual(len(cases), 9)
                self.assertEqual(
                    {case.family for case in cases}, {f"ticket31-{split}-{seed}"}
                )
                for class_code in ("scratch", "particle"):
                    covered = {
                        grid
                        for case in cases
                        for grid, codes in development_grid_annotations(case).items()
                        if class_code in codes
                    }
                    self.assertEqual(covered, expected_grids)
                self.assertEqual({case.position_bin for case in cases}, {"near", "center", "far"})
                self.assertEqual({case.scratch_orientation for case in cases}, {"horizontal", "vertical", "diagonal"})
                self.assertEqual({case.scratch_length for case in cases}, {24, 48, 80})
                self.assertEqual({case.particle_radius for case in cases}, {2, 3, 5})
                self.assertEqual({case.contrast_bin for case in cases}, {"low", "medium", "high"})

        first = corpus[0]
        pixels = render_ticket31_development_pixels(first)
        self.assertEqual(pixels.shape, (1536, 1536))
        self.assertEqual(pixels.dtype, np.uint8)
        self.assertEqual(hashlib.sha256(pixels.tobytes()).hexdigest(), first.pixel_sha256)
        truth = development_grid_annotations(first)
        self.assertTrue(truth)
        self.assertTrue(all(isinstance(grid, tuple) and len(grid) == 2 for grid in truth))


if __name__ == "__main__":
    unittest.main()
