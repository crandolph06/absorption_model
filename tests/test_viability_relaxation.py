import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.viability.config import load_config
from src.viability.relaxation import (
    build_relaxation_study,
    run_dynamic_relaxation_study,
)


class ViabilityRelaxationTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/viability.example.yaml")

    def test_relaxation_study_selects_linf_and_pareto_rows(self):
        evaluations = _synthetic_evaluations(self.config)

        study = build_relaxation_study(evaluations, self.config, top_n=3)

        nearest = study["nearest"]
        pareto = study["pareto"]
        summary = study["summary"]
        self.assertEqual(nearest.iloc[0]["schedule_id"], "balanced_near")
        self.assertAlmostEqual(summary["best_linf_relaxation"], 0.2)
        self.assertIn("wg_zero_trade", set(pareto["schedule_id"]))
        self.assertGreaterEqual(len(study["relaxation_sets"]), 4)

    def test_relaxation_study_writes_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluations_path = root / "evaluations.csv"
            _synthetic_evaluations(self.config).to_csv(evaluations_path, index=False)

            result = run_dynamic_relaxation_study(
                config=self.config,
                evaluation_paths=[evaluations_path],
                output_dir=root / "relaxation",
                top_n=3,
            )

            self.assertTrue(result.nearest_path.exists())
            self.assertTrue(result.pareto_path.exists())
            self.assertTrue(result.relaxation_sets_path.exists())
            self.assertTrue(result.summary_path.exists())
            self.assertTrue(result.report_path.exists())
            self.assertEqual(result.evaluated_count, 4)
            self.assertEqual(result.feasible_count, 0)

    def test_relaxation_study_fails_clearly_when_constraints_are_missing(self):
        evaluations = _synthetic_evaluations(self.config).drop(columns=["constraint_wg_rap"])

        with self.assertRaisesRegex(ValueError, "missing relaxation constraint columns"):
            build_relaxation_study(evaluations, self.config)


def _synthetic_evaluations(config):
    scales = {
        name: config.constraint_scales.scale_for(name)
        for name in ["total_pilots_window", "wg_rap", "fl_rap", "ip_rap"]
    }

    def row(schedule_id, phi, total, wg, fl, ip):
        return {
            "schedule_id": schedule_id,
            "status": "ok",
            "phase_backend": config.model.phase_backend,
            "feasible": False,
            "phi": phi,
            "active_constraint": "wg_rap",
            "active_constraint_value": wg * scales["wg_rap"],
            "constraint_total_pilots_window": total * scales["total_pilots_window"],
            "constraint_wg_rap": wg * scales["wg_rap"],
            "constraint_fl_rap": fl * scales["fl_rap"],
            "constraint_ip_rap": ip * scales["ip_rap"],
        }

    return pd.DataFrame(
        [
            row("balanced_near", 0.6, 0.2, 0.2, 0.2, 0.2),
            row("wg_zero_trade", 1.0, 0.8, 0.0, 0.7, 0.7),
            row("total_zero_trade", 1.1, 0.0, 0.9, 0.9, 0.9),
            row("dominated", 2.0, 1.0, 1.0, 1.0, 1.0),
        ]
    )


if __name__ == "__main__":
    unittest.main()
