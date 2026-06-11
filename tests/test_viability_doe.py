import unittest
from unittest.mock import patch

import numpy as np

from src.viability.doe import _sample_unit_cube


class ViabilityDoeTest(unittest.TestCase):
    def test_sobol_requires_scipy(self):
        with patch.dict("sys.modules", {"scipy": None, "scipy.stats": None}):
            with self.assertRaisesRegex(RuntimeError, "requires scipy.stats.qmc"):
                _sample_unit_cube(n=4, dimension=3, method="sobol", random_seed=1)

    def test_random_sampling_does_not_require_scipy(self):
        samples = _sample_unit_cube(n=4, dimension=3, method="random", random_seed=1)
        self.assertEqual(samples.shape, (4, 3))
        self.assertTrue(np.all(samples >= 0.0))
        self.assertTrue(np.all(samples <= 1.0))


if __name__ == "__main__":
    unittest.main()
