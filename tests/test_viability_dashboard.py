import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.viability.config import load_config
from src.viability.dashboard import (
    DashboardArtifactPaths,
    DynamicDashboardArtifactPaths,
    aggregate_history_trajectory,
    constraint_relaxation_table,
    direct_verification_caveat,
    direct_verification_label,
    dynamic_epoch_table,
    feasible_intervals,
    load_dynamic_dashboard_artifacts,
    load_dashboard_artifacts,
    local_feasible_sweep,
    nearest_dynamic_misses,
    one_lever_sweep,
    policy_values_from_row,
    select_dashboard_candidate,
    select_dynamic_schedule,
)


class _LinearUncertaintyModel:
    def __init__(self, weights, bias=0.0, sigma=0.0):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)
        self.sigma = float(sigma)

    def predict(self, x_values, return_std=False):
        mu = x_values @ self.weights + self.bias
        sigma = np.full(len(x_values), self.sigma, dtype=float)
        if return_std:
            return mu, sigma
        return mu


class ViabilityDashboardTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/viability.example.yaml")

    def test_near_boundary_default_selects_minimum_abs_phi(self):
        candidate = select_dashboard_candidate(
            _verified_candidates(self.config),
            mode="near_boundary_feasible",
        )

        self.assertEqual(candidate["candidate_id"], "candidate_near")
        self.assertAlmostEqual(float(candidate["phi"]), -0.05)

    def test_best_margin_and_specific_candidate_selection(self):
        verified = _verified_candidates(self.config)

        best = select_dashboard_candidate(verified, mode="best_margin_feasible")
        specific = select_dashboard_candidate(
            verified,
            mode="candidate_id",
            candidate_id="candidate_near",
        )

        self.assertEqual(best["candidate_id"], "candidate_best")
        self.assertEqual(specific["candidate_id"], "candidate_near")

    def test_one_lever_sweep_respects_bounds_and_integer_rounding(self):
        base_values = _policy_values()

        sweep = one_lever_sweep(self.config, base_values, "paa", max_points=5)

        self.assertTrue(sweep["raw_paa"].between(18, 30).all())
        self.assertTrue(sweep["paa"].between(18, 30).all())
        self.assertTrue(pd.api.types.is_integer_dtype(sweep["paa"]))
        self.assertEqual(sweep["paa"].tolist(), [18, 21, 24, 27, 30])

    def test_feasible_interval_extraction_returns_contiguous_sections(self):
        scored = pd.DataFrame(
            {
                "raw_ute": [6.0, 8.0, 10.0, 12.0, 14.0],
                "conservative_phi": [0.2, -0.1, -0.2, 0.1, -0.3],
            }
        )

        intervals = feasible_intervals(
            scored,
            "ute",
            feasible_column="conservative_phi",
            threshold=0.0,
        )

        self.assertEqual(len(intervals), 2)
        self.assertEqual((intervals[0].low, intervals[0].high), (8.0, 10.0))
        self.assertEqual((intervals[1].low, intervals[1].high), (14.0, 14.0))

    def test_missing_feasible_interval_returns_empty_result(self):
        scored = pd.DataFrame(
            {
                "raw_retention_rate": [0.1, 0.2, 0.3],
                "conservative_phi": [0.2, 0.1, 0.05],
            }
        )

        intervals = feasible_intervals(
            scored,
            "retention_rate",
            feasible_column="conservative_phi",
            threshold=0.0,
        )

        self.assertEqual(intervals, [])

    def test_synthetic_surrogate_produces_local_slider_interval(self):
        weights = np.zeros(len(self.config.policy.variables))
        weights[list(self.config.policy.variables).index("annual_intake")] = -1.0
        bundle = _linear_bundle(self.config, weights, bias=0.4, sigma=0.0)

        result = local_feasible_sweep(
            bundle,
            self.config,
            _policy_values(),
            "annual_intake",
            conservative_sigma=1.0,
            max_points=5,
        )

        self.assertGreater(len(result.intervals), 0)
        self.assertTrue((result.sweep["conservative_phi"] <= 0.4).all())
        self.assertGreaterEqual(result.intervals[0].low, 10.0)
        self.assertLessEqual(result.intervals[-1].high, 350.0)

    def test_policy_values_load_from_selected_verified_candidate(self):
        verified = _verified_candidates(self.config)
        row = select_dashboard_candidate(
            verified,
            mode="candidate_id",
            candidate_id="candidate_best",
        )

        values = policy_values_from_row(row, self.config)

        self.assertEqual(values["annual_intake"], 345)
        self.assertEqual(values["paa"], 24)
        self.assertAlmostEqual(float(row["phi"]), -1.0)

    def test_history_aggregation_produces_required_trajectory_columns(self):
        trajectory = aggregate_history_trajectory(
            _history_rows(),
            self.config,
        )

        for column in [
            "total_pilots",
            "line_pilots",
            "experience_ratio",
            "wg_rap_margin",
            "fl_rap_margin",
            "ip_rap_margin",
            "staff_ips",
            "staff_fls",
            "active_constraint",
            "phi",
            "timeline",
        ]:
            self.assertIn(column, trajectory.columns)
        self.assertEqual(len(trajectory), 2)
        self.assertTrue(np.isfinite(trajectory["phi"]).all())

    def test_artifact_loading_fails_clearly_when_required_columns_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_artifacts(
                self.config,
                root,
                verified_candidates=_verified_candidates(self.config).drop(
                    columns=["feasible"]
                ),
            )

            with self.assertRaisesRegex(
                ValueError,
                "verified candidates is missing required columns",
            ):
                load_dashboard_artifacts(paths)

    def test_artifact_loading_fails_when_envelope_link_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_artifacts(
                self.config,
                root,
                missing_envelope_plot=True,
            )

            with self.assertRaisesRegex(
                FileNotFoundError,
                "Envelope summary references missing plot files",
            ):
                load_dashboard_artifacts(paths)

    def test_artifact_loading_resolves_report_and_envelope_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_artifacts(self.config, root)

            artifacts = load_dashboard_artifacts(paths)

            self.assertEqual(len(artifacts.verified_candidates), 3)
            self.assertTrue(paths.report.exists())
            self.assertEqual(
                artifacts.envelope_summary["slices"][0]["x"],
                "annual_intake",
            )

    def test_direct_verification_labels_include_backend(self):
        self.assertIn("brain backend", direct_verification_label(self.config))
        self.assertIn("internal sortie brain", direct_verification_caveat(self.config))

    def test_dynamic_artifact_loading_and_nearest_miss_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_dynamic_artifacts(self.config, root, include_relaxation=True)

            artifacts = load_dynamic_dashboard_artifacts(paths)
            selected = select_dynamic_schedule(
                artifacts.evaluations,
                mode="best_phi",
            )
            epoch_table = dynamic_epoch_table(
                selected,
                artifacts.config,
                epoch_count=artifacts.epoch_count,
            )
            relaxations = constraint_relaxation_table(selected)
            misses = nearest_dynamic_misses(artifacts.evaluations, top_n=1)

            self.assertEqual(artifacts.epoch_count, 3)
            self.assertIsNotNone(artifacts.relaxation_summary)
            self.assertIsNotNone(artifacts.relaxation_nearest)
            self.assertIsNotNone(artifacts.relaxation_pareto)
            self.assertIsNotNone(artifacts.relaxation_sets)
            self.assertEqual(selected["schedule_id"], "schedule_best")
            self.assertEqual(epoch_table.shape, (3, 8))
            self.assertEqual(relaxations.iloc[0]["constraint"], "wg_rap")
            self.assertEqual(misses.iloc[0]["schedule_id"], "schedule_best")

    def test_dynamic_loading_fails_clearly_when_relaxation_artifacts_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_paths = _write_dynamic_artifacts(self.config, root)
            paths = DynamicDashboardArtifactPaths(
                config=base_paths.config,
                evaluations=base_paths.evaluations,
                summary=base_paths.summary,
                sensitivity=base_paths.sensitivity,
                report=base_paths.report,
                relaxation_dir=root / "missing_relaxation",
            )

            with self.assertRaisesRegex(
                FileNotFoundError,
                "Dynamic relaxation artifact directory does not exist",
            ):
                load_dynamic_dashboard_artifacts(paths)

    def test_dynamic_loading_fails_clearly_when_schedule_columns_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _write_dynamic_artifacts(
                self.config,
                root,
                evaluations=_dynamic_evaluations(self.config).drop(
                    columns=["epoch3_ipug_quota_per_phase"]
                ),
            )

            with self.assertRaisesRegex(
                ValueError,
                "dynamic evaluations is missing required columns",
            ):
                load_dynamic_dashboard_artifacts(paths)


def _policy_values(**overrides):
    values = {
        "annual_intake": 312,
        "retention_rate": 0.39,
        "ute": 16.0,
        "paa": 24,
        "max_manning_pct": 150.0,
        "flug_quota_per_phase": 5,
        "ipug_quota_per_phase": 5,
    }
    values.update(overrides)
    return values


def _policy_row(config, design_id, *, annual_intake, retention_rate, phi, feasible):
    row = {
        "design_id": design_id,
        "phi": phi,
        "feasible": feasible,
        "status": "ok",
        "constraint_total_pilots_window": phi * 100.0,
        "active_constraint": "total_pilots_window",
        "active_constraint_value": phi * 100.0,
        **_policy_values(
            annual_intake=annual_intake,
            retention_rate=retention_rate,
        ),
    }
    for name in config.policy.variables:
        row[f"raw_{name}"] = float(row[name])
        row[f"applied_{name}"] = row[name]
    return row


def _verified_candidates(config):
    return pd.DataFrame(
        [
            {
                **_policy_row(
                    config,
                    "design_best",
                    annual_intake=345,
                    retention_rate=0.62,
                    phi=-1.0,
                    feasible=True,
                ),
                "candidate_id": "candidate_best",
            },
            {
                **_policy_row(
                    config,
                    "design_near",
                    annual_intake=312,
                    retention_rate=0.39,
                    phi=-0.05,
                    feasible=True,
                ),
                "candidate_id": "candidate_near",
            },
            {
                **_policy_row(
                    config,
                    "design_bad",
                    annual_intake=280,
                    retention_rate=0.2,
                    phi=0.5,
                    feasible=False,
                ),
                "candidate_id": "candidate_bad",
            },
        ]
    )


def _evaluations(config):
    return pd.DataFrame(
        [
            _policy_row(
                config,
                "eval_0",
                annual_intake=300,
                retention_rate=0.4,
                phi=0.2,
                feasible=False,
            ),
            _policy_row(
                config,
                "eval_1",
                annual_intake=340,
                retention_rate=0.6,
                phi=-1.0,
                feasible=True,
            ),
        ]
    )


def _history_rows():
    return pd.DataFrame(
        [
            {
                "year": 2040,
                "phase": 1,
                "total_pilots": 1800,
                "line_pilots": 1500,
                "staff_ips": 120,
                "staff_fls": 130,
                "wg_rap_shortfall": -0.1,
                "fl_rap_shortfall": -0.2,
                "ip_rap_shortfall": -0.3,
                "fl_qty": 500,
                "ip_qty": 400,
            },
            {
                "year": 2040,
                "phase": 1,
                "total_pilots": 1900,
                "line_pilots": 1600,
                "staff_ips": 125,
                "staff_fls": 135,
                "wg_rap_shortfall": -0.2,
                "fl_rap_shortfall": -0.1,
                "ip_rap_shortfall": -0.4,
                "fl_qty": 600,
                "ip_qty": 500,
            },
            {
                "year": 2040,
                "phase": 2,
                "total_pilots": 1750,
                "line_pilots": 1450,
                "staff_ips": 115,
                "staff_fls": 128,
                "wg_rap_shortfall": -0.1,
                "fl_rap_shortfall": -0.3,
                "ip_rap_shortfall": -0.2,
                "fl_qty": 500,
                "ip_qty": 410,
            },
        ]
    )


def _linear_bundle(config, weights, bias=0.0, sigma=0.0):
    return {
        "models_by_constraint": {
            "total_pilots_window": _LinearUncertaintyModel(
                weights,
                bias=bias,
                sigma=sigma,
            )
        },
        "constraint_names": ["total_pilots_window"],
        "target": "normalized_constraints",
        "feature_names": list(config.policy.variables),
    }


def _write_artifacts(
    config,
    root,
    *,
    verified_candidates=None,
    missing_envelope_plot=False,
):
    surrogate_path = root / "surrogate_constraints_gpr.joblib"
    weights = np.zeros(len(config.policy.variables))
    joblib.dump(_linear_bundle(config, weights), surrogate_path)

    evaluations_path = root / "evaluations.csv"
    _evaluations(config).to_csv(evaluations_path, index=False)

    verified_path = root / "verified_candidates.csv"
    if verified_candidates is None:
        verified_candidates = _verified_candidates(config)
    verified_candidates.to_csv(verified_path, index=False)

    fixed_path = root / "fixed.png"
    projected_path = root / "projected.png"
    fixed_path.write_bytes(b"fake-png")
    if not missing_envelope_plot:
        projected_path.write_bytes(b"fake-png")

    search_summary_path = root / "search_summary.json"
    verification_summary_path = root / "verification_summary.json"
    envelope_summary_path = root / "envelope_summary.json"
    report_path = root / "report.md"

    search_summary_path.write_text(json.dumps({"scored_count": 2}), encoding="utf-8")
    verification_summary_path.write_text(
        json.dumps({"verified_count": 3}),
        encoding="utf-8",
    )
    envelope_summary_path.write_text(
        json.dumps(
            {
                "slices": [
                    {
                        "x": "annual_intake",
                        "y": "retention_rate",
                        "fixed_plot_path": str(fixed_path),
                        "projected_plot_path": str(projected_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    report_path.write_text("# Report\n", encoding="utf-8")

    return DashboardArtifactPaths(
        config=Path("configs/viability.example.yaml"),
        surrogate=surrogate_path,
        evaluations=evaluations_path,
        verified_candidates=verified_path,
        search_summary=search_summary_path,
        verification_summary=verification_summary_path,
        envelope_summary=envelope_summary_path,
        report=report_path,
    )


def _dynamic_evaluations(config):
    rows = []
    for schedule_id, phi, wg_rap in [
        ("schedule_best", 0.2, 0.2),
        ("schedule_other", 1.0, 1.0),
    ]:
        row = {
            "schedule_id": schedule_id,
            "schedule_source": "heuristic",
            "template_name": "static_best_current_miss",
            "sample_index": 0,
            "phase_backend": config.model.phase_backend,
            "phi": phi,
            "feasible": False,
            "active_constraint": "wg_rap",
            "active_constraint_value": wg_rap,
            "status": "ok",
            "error": None,
            "constraint_total_pilots_window": -10.0,
            "constraint_wg_rap": wg_rap,
            "constraint_fl_rap": 0.1,
            "constraint_ip_rap": -1.0,
        }
        for epoch in range(1, 4):
            for name, value in _policy_values().items():
                column = f"epoch{epoch}_{name}"
                row[column] = value
                row[f"raw_{column}"] = float(value)
                row[f"applied_{column}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _write_dynamic_artifacts(config, root, *, evaluations=None, include_relaxation=False):
    evaluations_path = root / "all_evaluations.csv"
    if evaluations is None:
        evaluations = _dynamic_evaluations(config)
    evaluations.to_csv(evaluations_path, index=False)

    summary_path = root / "dynamic_search_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "epoch_count": 3,
                "evaluated_count": int(len(evaluations)),
                "feasible_count": 0,
                "best_phi": float(evaluations["phi"].min()),
                "best_schedule_id": "schedule_best",
                "best_active_constraint": "wg_rap",
            }
        ),
        encoding="utf-8",
    )

    sensitivity_path = root / "local_sensitivity.csv"
    pd.DataFrame(
        [
            {
                "epoch": 1,
                "control": "retention_rate",
                "response": "wg_rap",
                "sensitivity": -1.0,
                "abs_sensitivity": 1.0,
            }
        ]
    ).to_csv(sensitivity_path, index=False)
    report_path = root / "dynamic_control_report.md"
    report_path.write_text("# Dynamic Control Report\n", encoding="utf-8")
    relaxation_dir = None
    if include_relaxation:
        relaxation_dir = root / "relaxation_study_v1"
        relaxation_dir.mkdir()
        (relaxation_dir / "relaxation_summary.json").write_text(
            json.dumps(
                {
                    "evaluated_count": int(len(evaluations)),
                    "ok_count": int(len(evaluations)),
                    "feasible_count": 0,
                    "best_phi": float(evaluations["phi"].min()),
                    "best_phi_schedule_id": "schedule_best",
                    "best_linf_relaxation": 0.2,
                    "best_linf_schedule_id": "schedule_best",
                }
            ),
            encoding="utf-8",
        )
        relaxation_table = evaluations.assign(
            max_normalized_relaxation=0.2,
            positive_normalized_sum=0.3,
        )
        relaxation_table.to_csv(relaxation_dir / "nearest_under_relaxation.csv", index=False)
        relaxation_table.to_csv(relaxation_dir / "pareto_frontier.csv", index=False)
        pd.DataFrame(
            [
                {
                    "constraint_set": "wg_rap",
                    "schedule_id": "schedule_best",
                    "max_normalized_relaxation": 0.2,
                    "sum_normalized_relaxation": 0.2,
                }
            ]
        ).to_csv(relaxation_dir / "relaxation_sets.csv", index=False)
        (relaxation_dir / "relaxation_report.md").write_text("# Relaxation\n", encoding="utf-8")

    return DynamicDashboardArtifactPaths(
        config=Path("configs/viability.example.yaml"),
        evaluations=evaluations_path,
        summary=summary_path,
        sensitivity=sensitivity_path,
        report=report_path,
        relaxation_dir=relaxation_dir,
    )


if __name__ == "__main__":
    unittest.main()
