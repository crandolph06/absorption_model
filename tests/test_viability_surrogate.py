import tempfile
import unittest

import joblib
import pandas as pd

from src.viability.config import load_config
from src.viability.doe import generate_doe
from src.viability.surrogate import fit_surrogates


class ViabilitySurrogateTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/viability.example.yaml")

    def test_fit_surrogates_writes_reloadable_artifacts_and_metrics(self):
        evaluations = _synthetic_evaluations(self.config, n=16, all_infeasible=False)
        with tempfile.TemporaryDirectory() as tmp:
            result = fit_surrogates(evaluations, self.config, tmp, fit_gpr=False)

            self.assertTrue(result.metrics_path.exists())
            self.assertIn("phi_ridge", result.model_paths)
            self.assertIn("constraint_total_pilots_final_ridge", result.model_paths)
            self.assertIn("false_feasible_rate", result.metrics)

            bundle = joblib.load(result.model_paths["phi_ridge"])
            self.assertEqual(bundle["target"], "phi")
            self.assertEqual(bundle["feature_names"], list(self.config.policy.variables))
            prediction = bundle["model"].predict([[0.5] * len(bundle["feature_names"])])
            self.assertEqual(prediction.shape, (1,))

    def test_fit_surrogates_handles_no_feasible_rows(self):
        evaluations = _synthetic_evaluations(self.config, n=16, all_infeasible=True)
        with tempfile.TemporaryDirectory() as tmp:
            result = fit_surrogates(evaluations, self.config, tmp, fit_gpr=False)

            self.assertGreaterEqual(result.metrics["false_feasible_rate"], 0.0)
            self.assertGreaterEqual(result.metrics["feasible_class_accuracy"], 0.0)


def _synthetic_evaluations(config, n: int, all_infeasible: bool) -> pd.DataFrame:
    frame = generate_doe(
        config,
        n=n,
        method="sobol",
        include_corners=False,
        include_baselines=False,
    )
    rows = []
    for _, row in frame.iterrows():
        annual = float(row["annual_intake"])
        retention = float(row["retention_rate"])
        ute = float(row["ute"])
        phi = 1.2 - annual / 220.0 - retention + (12.0 - ute) / 20.0
        if all_infeasible:
            phi = abs(phi) + 0.2
        rows.append(
            {
                **row.to_dict(),
                "status": "ok",
                "phi": phi,
                "feasible": phi <= 0.0,
                "constraint_total_pilots_final": phi * 100.0,
                "constraint_total_pilots_window": phi * 120.0,
                "constraint_wg_rap": phi + 0.05,
                "constraint_fl_rap": phi - 0.03,
                "constraint_ip_rap": phi + 0.02,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
