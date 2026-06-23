import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.viability.config import load_config
from src.viability.dynamic_analysis_common import clone_config_with_policy_highs
from src.viability.dynamic_bound_relaxation import (
    generate_bound_relaxation_candidates,
    run_dynamic_bound_relaxation_study,
)
from src.viability.dynamic_ipug import (
    generate_ipug_counterfactual_candidates,
    run_dynamic_ipug_diagnostic,
)
from src.viability.dynamic_trajectory_artifacts import (
    run_dynamic_trajectory_artifacts,
)
from src.viability.dynamic_policy import dynamic_feature_names


class ViabilityDynamicAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/viability.example.yaml")

    def test_clone_config_with_policy_highs_preserves_baseline(self):
        widened = clone_config_with_policy_highs(
            self.config,
            {"retention_rate": 0.95, "ipug_quota_per_phase": 20},
        )

        self.assertEqual(self.config.policy.variables["retention_rate"].high, 0.65)
        self.assertEqual(widened.policy.variables["retention_rate"].high, 0.95)
        self.assertEqual(widened.policy.variables["ipug_quota_per_phase"].high, 20.0)

    def test_bound_relaxation_candidates_are_fixed_shape_and_bounded(self):
        widened = clone_config_with_policy_highs(self.config, {"retention_rate": 0.95})
        row = _best_row(self.config)

        candidates = generate_bound_relaxation_candidates(
            widened,
            row,
            epoch_count=3,
            bound_extensions={"retention_rate": [0.75, 0.95]},
            sweep_points=3,
        )

        self.assertIn("baseline", set(candidates["experiment_id"]))
        self.assertIn("retention_rate", set(candidates["relaxed_variable"].dropna()))
        retention_columns = [f"epoch{epoch}_retention_rate" for epoch in range(1, 4)]
        varied = candidates[candidates["experiment_id"] == "retention_rate_high_0p95"]
        for _, candidate in varied.iterrows():
            self.assertEqual(len(set(candidate[column] for column in retention_columns)), 1)
        self.assertTrue(candidates["epoch1_retention_rate"].between(0.10, 0.95).all())

    def test_ipug_counterfactual_candidates_set_all_epochs(self):
        widened = clone_config_with_policy_highs(self.config, {"ipug_quota_per_phase": 20})

        candidates = generate_ipug_counterfactual_candidates(
            widened,
            _best_row(self.config),
            epoch_count=3,
            ipug_values=[0, 10, 20],
        )

        self.assertEqual(candidates["sweep_value"].tolist(), [0.0, 10.0, 20.0])
        for epoch in range(1, 4):
            self.assertEqual(candidates[f"epoch{epoch}_ipug_quota_per_phase"].tolist(), [0, 10, 20])

    def test_bound_relaxation_study_writes_artifacts_with_fake_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous_path = root / "previous.csv"
            _evaluations(self.config).to_csv(previous_path, index=False)

            result = run_dynamic_bound_relaxation_study(
                config=self.config,
                evaluations_path=previous_path,
                output_dir=root / "bound",
                epoch_count=3,
                sweep_points=2,
                bound_extensions={"retention_rate": [0.75]},
                workers=1,
                evaluator=_fake_evaluator,
            )

            self.assertTrue(result.candidates_path.exists())
            self.assertTrue(result.evaluations_path.exists())
            self.assertTrue(result.best_by_experiment_path.exists())
            self.assertTrue(result.summary_path.exists())
            report = result.report_path.read_text(encoding="utf-8")
            self.assertIn("Dynamic Input-Bound Relaxation Study", report)
            evaluated = pd.read_parquet(result.evaluations_path)
            self.assertIn("relaxed_variable", evaluated.columns)

    def test_ipug_diagnostic_writes_artifacts_with_fake_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous_path = root / "previous.csv"
            _evaluations(self.config).to_csv(previous_path, index=False)

            result = run_dynamic_ipug_diagnostic(
                config=self.config,
                evaluations_path=previous_path,
                output_dir=root / "ipug",
                epoch_count=3,
                ipug_values=[0, 10],
                workers=1,
                evaluator=_fake_evaluator,
            )

            self.assertTrue(result.candidates_path.exists())
            self.assertTrue(result.evaluations_path.exists())
            self.assertTrue(result.summary_path.exists())
            self.assertIn("IPUG", result.report_path.read_text(encoding="utf-8"))

    def test_trajectory_artifacts_write_figures_with_synthetic_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previous_path = root / "previous.csv"
            _evaluations(self.config).to_csv(previous_path, index=False)

            result = run_dynamic_trajectory_artifacts(
                config=self.config,
                evaluation_specs=[(previous_path, 3, "three_epoch")],
                output_dir=root / "figures",
                history_runner=lambda schedule, config: _history_rows(),
            )

            self.assertTrue(result.summary_path.exists())
            self.assertTrue(result.selected_policies_path.exists())
            self.assertIn("inventory", result.figure_paths)
            self.assertIn("constraint_trade_space", result.figure_paths)
            self.assertTrue(result.figure_paths["inventory"].exists())
            self.assertTrue(result.figure_paths["constraint_trade_space"].exists())
            self.assertTrue(result.trajectory_paths["three_epoch"].exists())


def _best_row(config):
    return _evaluations(config).sort_values("phi").iloc[0]


def _evaluations(config):
    rows = []
    feature_names = dynamic_feature_names(config.policy, 3)
    for schedule_id, phi in [("best", 0.3), ("other", 1.2)]:
        row = {
            "schedule_id": schedule_id,
            "schedule_source": "test",
            "sample_index": 0,
            "phase_backend": config.model.phase_backend,
            "phi": phi,
            "feasible": False,
            "active_constraint": "wg_rap",
            "active_constraint_value": phi,
            "status": "ok",
            "error": None,
            "constraint_total_pilots_window": 10.0,
            "constraint_wg_rap": phi,
            "constraint_fl_rap": 0.2,
            "constraint_ip_rap": -1.0,
        }
        for feature_name in feature_names:
            if feature_name.endswith("annual_intake"):
                value = 150
            elif feature_name.endswith("retention_rate"):
                value = 0.5
            elif feature_name.endswith("ute"):
                value = 18.0
            elif feature_name.endswith("paa"):
                value = 28
            elif feature_name.endswith("max_manning_pct"):
                value = 180.0
            elif feature_name.endswith("flug_quota_per_phase"):
                value = 2
            elif feature_name.endswith("ipug_quota_per_phase"):
                value = 0
            elif feature_name.endswith("upgrade_sortie_fraction"):
                value = 0.5
            elif feature_name.endswith("flug_window_start"):
                value = 250
            elif feature_name.endswith("ipug_window_start"):
                value = 400
            else:
                raise AssertionError(feature_name)
            row[feature_name] = value
            row[f"raw_{feature_name}"] = float(value)
            row[f"applied_{feature_name}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _fake_evaluator(
    schedules,
    config,
    *,
    epoch_count,
    workers,
    checkpoint_dir,
    checkpoint_every,
):
    rows = schedules.copy(deep=True)
    retention = rows.get("epoch1_retention_rate", pd.Series([0.5] * len(rows))).astype(float)
    ipug = rows.get("epoch1_ipug_quota_per_phase", pd.Series([0] * len(rows))).astype(float)
    phi = 1.0 - retention + 0.02 * ipug
    return rows.assign(
        phase_backend=config.model.phase_backend,
        status="ok",
        error=None,
        active_constraint="wg_rap",
        active_constraint_value=phi,
        constraint_total_pilots_window=10.0,
        constraint_wg_rap=phi,
        constraint_fl_rap=0.2,
        constraint_ip_rap=-1.0,
        metric_min_staff_ips_after_assessment_start=100.0 + ipug,
        metric_max_ip_rap_shortfall_after_assessment_start=-1.0,
        phi=phi,
        feasible=phi <= 0.0,
    )


def _history_rows():
    rows = []
    for index in range(6):
        rows.append(
            {
                "year": 2026 + index // 3,
                "phase": index % 3 + 1,
                "total_pilots": 3000 + index * 20,
                "line_pilots": 2500 + index * 10,
                "staff_ips": 100 + index,
                "staff_fls": 120 + index,
                "wg_rap_shortfall": 6.0 - index * 0.2,
                "fl_rap_shortfall": 3.0 - index * 0.1,
                "ip_rap_shortfall": -1.0,
                "fl_qty": 900,
                "ip_qty": 400,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
