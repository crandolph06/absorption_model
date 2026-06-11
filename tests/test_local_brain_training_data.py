import unittest

import numpy as np

from tools.generate_local_brain_training_data import (
    collect_valid_configs,
    single_phase_config_from_unit_values,
)


class LocalBrainTrainingDataTest(unittest.TestCase):
    def test_unit_values_map_to_single_phase_bounds(self):
        config = single_phase_config_from_unit_values(np.array([0, 0, 0, 0, 0, 0, 0, 0]))
        self.assertEqual(config, (6, 3, 0.0, 18, 0, 0, 0, 25))

        config = single_phase_config_from_unit_values(
            np.array(
                [
                    0.999999,
                    0.999999,
                    1.0,
                    0.999999,
                    0.999999,
                    0.999999,
                    0.999999,
                    0.999999,
                ]
            )
        )
        self.assertEqual(config, (20, 9, 1.0, 23, 14, 14, 14, 49))

    def test_sobol_valid_config_collection_uses_stable_indices(self):
        configs = collect_valid_configs(
            n=3,
            method="sobol",
            start_index=8,
            seed=7,
            scramble=True,
            chunk_size=8,
        )

        self.assertEqual(len(configs), 3)
        self.assertGreaterEqual(configs[0][0], 8)
        self.assertLess(configs[0][0], configs[-1][0])


if __name__ == "__main__":
    unittest.main()
