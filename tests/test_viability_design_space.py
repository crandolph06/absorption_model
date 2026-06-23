import unittest

from src.viability.config import load_config
from src.viability.design_space import DesignSpace
from src.viability.doe import dataframe_to_design_records, generate_doe


class ViabilityDesignSpaceTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/viability.example.yaml")
        self.space = DesignSpace(self.config.policy)

    def test_denormalize_rounds_integer_variables(self):
        design = self.space.denormalize([0.5] * self.space.dimension)

        self.assertEqual(design["annual_intake"], 180)
        self.assertEqual(design["paa"], 24)
        self.assertEqual(design["flug_quota_per_phase"], 5)
        self.assertEqual(design["ipug_quota_per_phase"], 5)
        self.assertAlmostEqual(design["retention_rate"], 0.375)

    def test_denormalize_with_raw_preserves_continuous_values(self):
        unit_values = [0.5137] * self.space.dimension
        raw, applied = self.space.denormalize_with_raw(unit_values)

        self.assertAlmostEqual(raw["annual_intake"], 10 + 0.5137 * 340)
        self.assertEqual(applied["annual_intake"], int(round(raw["annual_intake"])))
        self.assertNotAlmostEqual(raw["annual_intake"], float(applied["annual_intake"]))

    def test_validate_design_rejects_out_of_bounds_value(self):
        design = {
            "annual_intake": 351,
            "retention_rate": 0.5,
            "ute": 12,
            "paa": 24,
            "max_manning_pct": 150,
            "flug_quota_per_phase": 3,
            "ipug_quota_per_phase": 2,
            "upgrade_sortie_fraction": 0.5,
        }

        with self.assertRaisesRegex(ValueError, "annual_intake"):
            self.space.validate_design(design)

    def test_generate_doe_includes_design_ids_and_baseline(self):
        df = generate_doe(
            self.config,
            n=10,
            method="random",
            include_corners=False,
            include_baselines=True,
        )

        self.assertEqual(list(df.columns)[0], "design_id")
        self.assertIn("doe_source", df.columns)
        self.assertIn("sample_index", df.columns)
        self.assertEqual(df.iloc[0]["design_id"], "random_000000")
        self.assertIn("raw_annual_intake", df.columns)
        self.assertIn("applied_annual_intake", df.columns)
        self.assertGreaterEqual(len(df), 10)
        baseline = df[
            (df["annual_intake"] == 250)
            & (df["retention_rate"] == 0.5)
            & (df["ute"] == 12.0)
            & (df["paa"] == 24)
        ]
        self.assertEqual(len(baseline), 1)

    def test_generate_doe_resume_omits_corners_and_baseline_by_default(self):
        config_dict = self.config.to_dict()
        config_dict["doe"]["start_index"] = 8
        resumed_config = type(self.config).from_dict(config_dict)

        df = generate_doe(
            resumed_config,
            n=4,
            method="sobol",
        )

        self.assertEqual(list(df["doe_source"].unique()), ["sobol"])
        self.assertEqual(df.iloc[0]["design_id"], "sobol_000008")
        self.assertEqual(df.iloc[-1]["design_id"], "sobol_000011")

    def test_doe_records_convert_to_policy_design_dicts(self):
        df = generate_doe(
            self.config,
            n=3,
            method="random",
            include_corners=False,
            include_baselines=False,
        )

        records = dataframe_to_design_records(df, self.config)

        self.assertEqual(len(records), 3)
        self.assertEqual(set(records[0]), set(self.space.variable_names))


if __name__ == "__main__":
    unittest.main()
