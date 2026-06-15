import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.viability.config import load_config
from src.viability.dynamic_search import (
    generate_dynamic_schedules,
    generate_local_perturbation_schedules,
    run_dynamic_policy_diagnostic,
    run_dynamic_policy_search,
)


class ViabilityDynamicSearchTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/viability.example.yaml")

    def test_dynamic_schedule_heuristics_are_named_in_artifacts(self):
        schedules = generate_dynamic_schedules(
            self.config,
            epoch_count=3,
            n=0,
            start_index=0,
            source="unused",
            include_heuristics=True,
        )

        self.assertIn("template_name", schedules.columns)
        self.assertIn("static_best_current_miss", set(schedules["template_name"]))
        self.assertEqual(
            schedules.loc[
                schedules["template_name"] == "static_best_current_miss",
                "schedule_id",
            ].iloc[0],
            "heuristic_0002",
        )

    def test_dynamic_search_writes_expected_artifacts_with_fake_evaluator(self):
        calls = []

        def fake_evaluator(
            schedules,
            config,
            *,
            epoch_count,
            workers,
            checkpoint_dir,
            checkpoint_every,
        ):
            calls.append(
                {
                    "rows": len(schedules),
                    "epoch_count": epoch_count,
                    "workers": workers,
                    "checkpoint_every": checkpoint_every,
                }
            )
            rows = schedules.copy(deep=True)
            total_window = rows["raw_epoch1_annual_intake"].to_numpy(dtype=float) / 100.0
            rows = rows.assign(
                phase_backend=config.model.phase_backend,
                status="ok",
                error=None,
                active_constraint="total_pilots_window",
                active_constraint_value=1.0,
                constraint_total_pilots_window=total_window,
                constraint_wg_rap=1.0,
                constraint_fl_rap=2.0,
                constraint_ip_rap=-1.0,
                phi=total_window,
                feasible=total_window <= 0.0,
            )
            return rows

        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.viability.dynamic_search._differential_evolution_candidate", return_value=None):
                result = run_dynamic_policy_search(
                    config=self.config,
                    output_dir=tmp,
                    epoch_count=3,
                    initial_samples=4,
                    optimizer_pool_size=16,
                    verify_top=3,
                    workers=2,
                    checkpoint_every=2,
                    evaluator=fake_evaluator,
                )

            root = Path(tmp)
            self.assertTrue((root / "initial_schedules.csv").exists())
            self.assertTrue((root / "initial_evaluations.parquet").exists())
            self.assertTrue((root / "optimizer_candidates.csv").exists())
            self.assertTrue((root / "optimizer_evaluations.parquet").exists())
            self.assertTrue((root / "all_evaluations.parquet").exists())
            self.assertTrue((root / "dynamic_search_summary.json").exists())
            self.assertEqual(result.evaluated_count, calls[0]["rows"] + calls[1]["rows"])
            self.assertEqual(calls[0]["workers"], 2)
            self.assertEqual(calls[0]["checkpoint_every"], 2)
            self.assertEqual(pd.read_parquet(result.all_evaluations_path).shape[0], result.evaluated_count)

    def test_dynamic_diagnostic_writes_sensitivity_and_report(self):
        def fake_evaluator(
            schedules,
            config,
            *,
            epoch_count,
            workers,
            checkpoint_dir,
            checkpoint_every,
        ):
            rows = schedules.copy(deep=True)
            phi = 10.0 - rows["raw_epoch1_annual_intake"].to_numpy(dtype=float) / 100.0
            rows = rows.assign(
                phase_backend=config.model.phase_backend,
                status="ok",
                error=None,
                active_constraint="wg_rap",
                active_constraint_value=phi,
                constraint_total_pilots_window=1.0,
                constraint_wg_rap=phi,
                constraint_fl_rap=2.0,
                constraint_ip_rap=-1.0,
                phi=phi,
                feasible=phi <= 0.0,
            )
            return rows

        with tempfile.TemporaryDirectory() as tmp:
            schedules = generate_local_perturbation_schedules(
                self.config,
                _seed_best_row(self.config),
                epoch_count=3,
                total_phases=60,
                perturbation_fraction=0.05,
            )
            evaluations = fake_evaluator(
                schedules,
                self.config,
                epoch_count=3,
                workers=1,
                checkpoint_dir=Path(tmp) / "unused",
                checkpoint_every=10,
            )
            evaluations_path = Path(tmp) / "search_evaluations.parquet"
            evaluations.to_parquet(evaluations_path, index=False)

            result = run_dynamic_policy_diagnostic(
                config=self.config,
                evaluations_path=evaluations_path,
                output_dir=Path(tmp) / "diagnostic",
                epoch_count=3,
                workers=1,
                evaluator=fake_evaluator,
            )

            self.assertTrue(result.sensitivity_path.exists())
            self.assertTrue(result.report_path.exists())
            sensitivity = pd.read_csv(result.sensitivity_path)
            self.assertIn("wg_rap", set(sensitivity["response"]))
            self.assertIn("Finite-Horizon Control", result.report_path.read_text())


def _seed_best_row(config):
    row = {
        "schedule_id": "seed",
        "status": "ok",
        "phase_backend": config.model.phase_backend,
        "phi": 1.0,
        "feasible": False,
        "active_constraint": "wg_rap",
        "constraint_total_pilots_window": 1.0,
        "constraint_wg_rap": 1.0,
        "constraint_fl_rap": 2.0,
        "constraint_ip_rap": -1.0,
    }
    for epoch in range(1, 4):
        row[f"raw_epoch{epoch}_annual_intake"] = 150
        row[f"raw_epoch{epoch}_retention_rate"] = 0.5
        row[f"raw_epoch{epoch}_ute"] = 18
        row[f"raw_epoch{epoch}_paa"] = 28
        row[f"raw_epoch{epoch}_max_manning_pct"] = 180
        row[f"raw_epoch{epoch}_flug_quota_per_phase"] = 2
        row[f"raw_epoch{epoch}_ipug_quota_per_phase"] = 0
        row[f"epoch{epoch}_annual_intake"] = 150
        row[f"epoch{epoch}_retention_rate"] = 0.5
        row[f"epoch{epoch}_ute"] = 18
        row[f"epoch{epoch}_paa"] = 28
        row[f"epoch{epoch}_max_manning_pct"] = 180
        row[f"epoch{epoch}_flug_quota_per_phase"] = 2
        row[f"epoch{epoch}_ipug_quota_per_phase"] = 0
    return pd.Series(row)


if __name__ == "__main__":
    unittest.main()
