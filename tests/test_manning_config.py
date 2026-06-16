import unittest

from src.manning_config import get_initial_squadrons


class ManningConfigTest(unittest.TestCase):
    def test_empty_upgrade_seed_means_no_preseeded_upgrades(self):
        squadrons = get_initial_squadrons(2026, [(1, 10, 2, 10, 10.0, 0.5, ())])

        self.assertEqual(len(squadrons), 1)
        self.assertEqual(squadrons[0].mqt_students, 0)
        self.assertEqual(squadrons[0].flug_students, 0)
        self.assertEqual(squadrons[0].ipug_students, 0)

    def test_upgrade_seed_requires_three_values_when_present(self):
        with self.assertRaisesRegex(ValueError, "upgrade seed must be empty or"):
            get_initial_squadrons(2026, [(1, 10, 2, 10, 10.0, 0.5, (1, 2))])

    def test_squadron_seed_requires_seven_values(self):
        with self.assertRaisesRegex(ValueError, "Squadron seed must be"):
            get_initial_squadrons(2026, [(1, 10, 2, 10, ())])


if __name__ == "__main__":
    unittest.main()
