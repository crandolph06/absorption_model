import json
import tempfile
import unittest

import joblib
import pandas as pd

from src.viability.config import load_config
from src.viability.doe import generate_doe
from src.viability.io import write_config_resolved
from src.viability.surrogate import (
    fit_surrogates,
    fit_constraint_gpr_bundle,
    normalized_constraint_frame,
    predict_constraint_surrogate,
    run_gpr_convergence,
    write_holdout_selection_from_file,
    write_gpr_prediction_overlay_plot,
)


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

    def test_fit_surrogates_reports_gpr_metrics_when_enabled(self):
        evaluations = _synthetic_evaluations(self.config, n=16, all_infeasible=False)
        with tempfile.TemporaryDirectory() as tmp:
            result = fit_surrogates(evaluations, self.config, tmp, fit_gpr=True)

            self.assertEqual(result.metrics["gpr_status"], "fit")
            self.assertIn("gpr_phi", result.metrics)
            self.assertIn("MAE_phi", result.metrics["gpr_phi"])
            self.assertIn("constraint_sign_accuracy", result.metrics["gpr_phi"])
            self.assertIn("constraints_gpr", result.model_paths)

            bundle = joblib.load(result.model_paths["constraints_gpr"])
            self.assertEqual(bundle["target"], "normalized_constraints")
            self.assertIn("models_by_constraint", bundle)

    def test_normalized_constraints_reconstruct_phi(self):
        evaluations = _synthetic_evaluations(self.config, n=8, all_infeasible=False)
        constraint_columns = [
            column for column in evaluations.columns if column.startswith("constraint_")
        ]

        normalized = normalized_constraint_frame(
            evaluations,
            constraint_columns,
            self.config,
        )

        reconstructed = normalized.max(axis=1)
        pd.testing.assert_series_equal(
            reconstructed.reset_index(drop=True),
            evaluations["phi"].reset_index(drop=True),
            check_names=False,
        )

    def test_normalized_constraints_fail_when_phi_does_not_match(self):
        evaluations = _synthetic_evaluations(self.config, n=8, all_infeasible=False)
        evaluations.loc[0, "constraint_wg_rap"] = evaluations.loc[0, "phi"] + 10.0
        constraint_columns = [
            column for column in evaluations.columns if column.startswith("constraint_")
        ]

        with self.assertRaisesRegex(ValueError, "Stored phi does not match"):
            normalized_constraint_frame(evaluations, constraint_columns, self.config)

    def test_constraint_gpr_bundle_predicts_reconstructed_phi(self):
        evaluations = _synthetic_evaluations(self.config, n=16, all_infeasible=False)
        constraint_columns = [
            column for column in evaluations.columns if column.startswith("constraint_")
        ]
        normalized = normalized_constraint_frame(evaluations, constraint_columns, self.config)
        x = evaluations[list(self.config.policy.variables)].to_numpy(dtype=float)
        bundle = fit_constraint_gpr_bundle(
            x=x,
            normalized_constraints=normalized,
            feature_names=list(self.config.policy.variables),
            config=self.config,
            max_rows=len(evaluations),
        )

        prediction = predict_constraint_surrogate(bundle, x[:3], conservative_sigma=2.0)

        self.assertEqual(prediction.mu.shape, (3, len(normalized.columns)))
        self.assertEqual(prediction.sigma.shape, (3, len(normalized.columns)))
        self.assertEqual(prediction.predicted_phi.shape, (3,))
        self.assertTrue((prediction.conservative_phi >= prediction.predicted_phi).all())
        self.assertEqual(len(prediction.active_constraint), 3)

    def test_run_gpr_convergence_writes_table_model_and_plots(self):
        evaluations = _synthetic_evaluations(self.config, n=32, all_infeasible=False)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_gpr_convergence(
                evaluations,
                self.config,
                tmp,
                train_sizes=[4, 8],
                holdout_fraction=0.25,
                target_r2=0.0,
                target_normalized_mae=1.0,
                target_normalized_rmse=1.0,
            )

            self.assertTrue(result.metrics_path.exists())
            self.assertTrue(result.metrics_table_path.exists())
            self.assertTrue(result.model_path.exists())
            self.assertEqual(result.model_path.name, "surrogate_constraints_gpr.joblib")
            self.assertIn("R2_phi", result.metrics_table.columns)
            self.assertIn("MAE_phi_normalized", result.metrics_table.columns)
            self.assertIn("MSE_phi_normalized", result.metrics_table.columns)
            self.assertIn("constraint_sign_accuracy", result.metrics_table.columns)
            for path in result.plot_paths.values():
                self.assertTrue(path.exists())

    def test_run_gpr_convergence_uses_external_holdout_without_leakage(self):
        evaluations = _synthetic_evaluations(self.config, n=24, all_infeasible=False)
        holdout = evaluations.iloc[:6].copy()
        with tempfile.TemporaryDirectory() as tmp:
            result = run_gpr_convergence(
                evaluations,
                self.config,
                tmp,
                train_sizes=[4],
                holdout_evaluations=holdout,
            )

            summary = json.loads(result.metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["holdout_source"], "external_table")
            self.assertEqual(summary["holdout_size"], 6)
            self.assertEqual(summary["n_rows_used"], 18)
            self.assertEqual(summary["holdout_id_columns"], ["design_id"])
            self.assertEqual(result.metrics_table.iloc[-1]["train_size"], 18)

    def test_write_holdout_selection_from_file_writes_reusable_table(self):
        evaluations = _synthetic_evaluations(self.config, n=24, all_infeasible=False)
        with tempfile.TemporaryDirectory() as tmp:
            evaluations_path = f"{tmp}/evaluations.csv"
            holdout_path = f"{tmp}/holdout.csv"
            evaluations.to_csv(evaluations_path, index=False)

            result = write_holdout_selection_from_file(
                evaluations_path,
                self.config,
                holdout_path,
                holdout_fraction=0.25,
            )

            holdout = pd.read_csv(result.holdout_path)
            self.assertEqual(result.n_rows_total, 24)
            self.assertEqual(result.holdout_size, 6)
            self.assertEqual(len(holdout), 6)
            self.assertIn("design_id", holdout.columns)

    def test_write_gpr_prediction_overlay_plot_from_convergence_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            holdout = _synthetic_evaluations(self.config, n=10, all_infeasible=False)
            holdout_path = f"{tmp}/fixed_holdout.csv"
            holdout.to_csv(holdout_path, index=False)
            run_dirs = []
            for name, n_rows in [("gpr_convergence_1024", 32), ("gpr_convergence_2048", 40)]:
                evaluations = _synthetic_evaluations(
                    self.config,
                    n=n_rows,
                    all_infeasible=False,
                )
                run_dir = f"{tmp}/{name}"
                write_config_resolved(self.config, run_dir)
                run_gpr_convergence(
                    evaluations,
                    self.config,
                    run_dir,
                    train_sizes=[4, 8],
                    holdout_fraction=0.25,
                    target_r2=0.0,
                    target_normalized_mae=1.0,
                    target_normalized_rmse=1.0,
                    holdout_evaluations=holdout,
                    holdout_path=holdout_path,
                )
                run_dirs.append(run_dir)

            result = write_gpr_prediction_overlay_plot(
                run_dirs,
                f"{tmp}/overlay.png",
                labels=["1024", "2048"],
                colors=["black", "firebrick"],
                alphas=[0.25, 0.75],
                zorders=[1, 2],
            )

            self.assertTrue(result.plot_path.exists())
            self.assertEqual(result.point_counts["1024"], 10)
            self.assertEqual(result.point_counts["2048"], 10)


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
                "constraint_total_pilots_window": (phi - 0.2) * 100.0,
                "constraint_wg_rap": phi - 0.3,
                "constraint_fl_rap": phi - 0.4,
                "constraint_ip_rap": phi - 0.5,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
