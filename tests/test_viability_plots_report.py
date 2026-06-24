from dataclasses import replace
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.viability.config import EnvelopeSliceConfig, ViabilityConfig, load_config
from src.viability.plots import (
    differential_evolution_comparison,
    fixed_slice_grid,
    projected_sobol_grid,
    require_envelope_config,
    run_envelope_plots,
    select_anchor_policy,
)
from src.viability.report import require_report_config, write_viability_report


class _LinearUncertaintyModel:
    def __init__(self, weights, bias=0.0, sigma=0.05):
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)
        self.sigma = float(sigma)

    def predict(self, x_values, return_std=False):
        mu = x_values @ self.weights + self.bias
        sigma = np.full(len(x_values), self.sigma, dtype=float)
        if return_std:
            return mu, sigma
        return mu


def _linear_bundle(config, weights, bias=0.0, sigma=0.05):
    return {
        "models_by_constraint": {
            "total_pilots_final": _LinearUncertaintyModel(weights, bias=bias, sigma=sigma)
        },
        "constraint_names": ["total_pilots_final"],
        "target": "normalized_constraints",
        "feature_names": list(config.policy.variables),
    }


class ViabilityPlotsReportTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/viability.example.yaml")

    def test_envelope_config_requires_all_fields_when_section_present(self):
        data = self.config.to_dict()
        del data["envelope"]["grid_size"]

        with self.assertRaisesRegex(ValueError, "envelope.grid_size"):
            ViabilityConfig.from_dict(data)

    def test_report_config_requires_all_fields_when_section_present(self):
        data = self.config.to_dict()
        del data["report"]["top_candidate_count"]

        with self.assertRaisesRegex(ValueError, "report.top_candidate_count"):
            ViabilityConfig.from_dict(data)

    def test_envelope_and_report_config_are_required_by_commands(self):
        with self.assertRaisesRegex(ValueError, "envelope section"):
            require_envelope_config(replace(self.config, envelope=None))
        with self.assertRaisesRegex(ValueError, "report section"):
            require_report_config(replace(self.config, report=None))

    def test_near_boundary_feasible_anchor_selects_smallest_abs_phi(self):
        verified = _verified_candidates(self.config)
        envelope_config = require_envelope_config(self.config)

        anchor = select_anchor_policy(verified, envelope_config)

        self.assertEqual(anchor["candidate_id"], "candidate_near")
        self.assertAlmostEqual(float(anchor["phi"]), -0.05)

    def test_anchor_selection_fails_without_verified_feasible_rows(self):
        verified = _verified_candidates(self.config)
        verified["feasible"] = False
        envelope_config = require_envelope_config(self.config)

        with self.assertRaisesRegex(ValueError, "no verified candidates are feasible"):
            select_anchor_policy(verified, envelope_config)

    def test_run_envelope_plots_skips_slices_without_verified_feasible_rows(self):
        config = _small_envelope_config(
            self.config,
            slices=[EnvelopeSliceConfig(x="annual_intake", y="retention_rate")],
            grid_size=3,
            sobol_hidden_samples=4,
            de_points=1,
            de_maxiter=1,
            de_popsize=2,
        )
        weights = np.zeros(len(config.policy.variables))
        bundle = _linear_bundle(config, weights, bias=0.4)
        verified = _verified_candidates(config)
        verified["feasible"] = False

        with tempfile.TemporaryDirectory() as tmp:
            result = run_envelope_plots(
                surrogate=bundle,
                surrogate_path=f"{tmp}/surrogate_constraints_gpr.joblib",
                evaluations=_evaluations(config),
                verified_candidates=verified,
                config=config,
                output_dir=tmp,
            )

            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            self.assertTrue(result.plots_skipped)
            self.assertEqual(result.plot_paths, {})
            self.assertEqual(summary["slices"], [])
            self.assertEqual(summary["best_verified_candidate_id"], "candidate_best")
            self.assertIsNone(summary["anchor_design_id"])

    def test_report_generation_notes_skipped_envelope_plots(self):
        config = replace(
            self.config,
            report=replace(
                require_report_config(self.config),
                top_candidate_count=2,
                near_boundary_count=2,
            ),
        )
        verified = _verified_candidates(config)
        verified["feasible"] = False
        envelope_summary = {
            "plots_skipped": True,
            "plots_skipped_reason": "No verified feasible candidates.",
            "best_verified_candidate_id": "candidate_best",
            "best_verified_phi": 0.5,
            "slices": [],
        }
        search_summary = {
            "scored_count": 16,
            "selected_count": 2,
        }
        verification_summary = {
            "verified_count": 3,
            "verified_feasible_count": 0,
            "predicted_feasible_count": 0,
            "conservative_predicted_feasible_count": 0,
            "false_feasible_count": 0,
            "false_conservative_feasible_count": 0,
            "best_verified_phi": 0.5,
        }

        with tempfile.TemporaryDirectory() as tmp:
            result = write_viability_report(
                config=config,
                evaluations=_evaluations(config),
                verified_candidates=verified,
                search_summary=search_summary,
                verification_summary=verification_summary,
                envelope_summary=envelope_summary,
                output_path=f"{tmp}/report.md",
            )

            text = result.report_path.read_text(encoding="utf-8")
            self.assertIn("Verified feasible candidates: `0`", text)
            self.assertIn("No verified feasible policies were found.", text)
            self.assertIn("No verified feasible candidates.", text)
            self.assertIn("candidate_best", text)

    def test_fixed_slice_grid_respects_integer_rounding(self):
        config = _small_envelope_config(
            self.config,
            slices=[EnvelopeSliceConfig(x="paa", y="ute")],
            grid_size=5,
            sobol_hidden_samples=4,
        )
        weights = np.zeros(len(config.policy.variables))
        bundle = _linear_bundle(config, weights)
        anchor = select_anchor_policy(_verified_candidates(config), require_envelope_config(config))

        grid = fixed_slice_grid(
            bundle,
            anchor,
            require_envelope_config(config).slices[0],
            config,
            require_envelope_config(config),
        )

        self.assertEqual(len(grid), 25)
        self.assertTrue(pd.api.types.is_integer_dtype(grid["paa"]))
        self.assertTrue(grid["paa"].between(18, 30).all())
        self.assertTrue(grid["ute"].between(6.0, 20.0).all())

    def test_projected_sobol_grid_returns_minimum_over_hidden_candidates(self):
        config = _small_envelope_config(
            self.config,
            slices=[EnvelopeSliceConfig(x="annual_intake", y="retention_rate")],
            grid_size=2,
            sobol_hidden_samples=4,
            scramble=False,
        )
        weights = np.zeros(len(config.policy.variables))
        weights[list(config.policy.variables).index("ute")] = 1.0
        bundle = _linear_bundle(config, weights)

        grid = projected_sobol_grid(
            bundle,
            require_envelope_config(config).slices[0],
            config,
            require_envelope_config(config),
        )

        self.assertEqual(len(grid), 4)
        self.assertTrue((grid["projected_phi"] >= 0.0).all())
        self.assertTrue((grid["projected_phi"] <= 0.5).all())
        self.assertIn("best_ute", grid.columns)

    def test_differential_evolution_comparison_writes_expected_columns(self):
        config = _small_envelope_config(
            self.config,
            slices=[EnvelopeSliceConfig(x="annual_intake", y="retention_rate")],
            grid_size=2,
            sobol_hidden_samples=4,
            de_points=2,
            de_maxiter=1,
            de_popsize=2,
        )
        weights = np.zeros(len(config.policy.variables))
        weights[list(config.policy.variables).index("ute")] = 1.0
        bundle = _linear_bundle(config, weights)
        projected = projected_sobol_grid(
            bundle,
            require_envelope_config(config).slices[0],
            config,
            require_envelope_config(config),
        )

        comparison = differential_evolution_comparison(
            bundle,
            projected,
            require_envelope_config(config).slices[0],
            config,
            require_envelope_config(config),
        )

        self.assertEqual(len(comparison), 2)
        self.assertIn("de_predicted_phi", comparison.columns)
        self.assertIn("de_nfev", comparison.columns)

    def test_run_envelope_plots_writes_artifacts(self):
        config = _small_envelope_config(
            self.config,
            slices=[EnvelopeSliceConfig(x="annual_intake", y="retention_rate")],
            grid_size=3,
            sobol_hidden_samples=4,
            de_points=1,
            de_maxiter=1,
            de_popsize=2,
        )
        weights = np.zeros(len(config.policy.variables))
        weights[list(config.policy.variables).index("annual_intake")] = -1.0
        weights[list(config.policy.variables).index("retention_rate")] = -1.0
        bundle = _linear_bundle(config, weights, bias=0.4)

        with tempfile.TemporaryDirectory() as tmp:
            result = run_envelope_plots(
                surrogate=bundle,
                surrogate_path=f"{tmp}/surrogate_constraints_gpr.joblib",
                evaluations=_evaluations(config),
                verified_candidates=_verified_candidates(config),
                config=config,
                output_dir=tmp,
            )

            self.assertTrue(result.summary_path.exists())
            self.assertEqual(result.anchor_design_id, "design_near")
            self.assertEqual(len(result.plot_paths), 2)
            self.assertEqual(len(result.grid_paths), 2)
            self.assertEqual(len(result.de_comparison_paths), 1)

    def test_report_generation_includes_key_results_and_plot_links(self):
        config = replace(
            self.config,
            report=replace(
                require_report_config(self.config),
                top_candidate_count=2,
                near_boundary_count=2,
            ),
        )
        envelope_summary = {
            "slices": [
                {
                    "x": "annual_intake",
                    "y": "retention_rate",
                    "fixed_plot_path": "/tmp/fixed.png",
                    "projected_plot_path": "/tmp/projected.png",
                }
            ]
        }
        search_summary = {
            "scored_count": 16,
            "selected_count": 2,
        }
        verification_summary = {
            "verified_count": 3,
            "verified_feasible_count": 2,
            "predicted_feasible_count": 2,
            "conservative_predicted_feasible_count": 1,
            "false_feasible_count": 1,
            "false_conservative_feasible_count": 0,
            "best_verified_phi": -1.0,
        }

        with tempfile.TemporaryDirectory() as tmp:
            result = write_viability_report(
                config=config,
                evaluations=_evaluations(config),
                verified_candidates=_verified_candidates(config),
                search_summary=search_summary,
                verification_summary=verification_summary,
                envelope_summary=envelope_summary,
                output_path=f"{tmp}/report.md",
            )

            text = result.report_path.read_text(encoding="utf-8")
            self.assertIn("Verified feasible candidates: `2`", text)
            self.assertIn("False feasible count", text)
            self.assertIn("candidate_best", text)
            self.assertIn("annual_intake vs retention_rate", text)


def _small_envelope_config(
    config,
    *,
    slices,
    grid_size,
    sobol_hidden_samples,
    scramble=True,
    de_points=1,
    de_maxiter=1,
    de_popsize=2,
):
    doe = replace(config.doe, scramble=scramble)
    envelope = replace(
        require_envelope_config(config),
        grid_size=grid_size,
        prediction_chunk_size=64,
        sobol_hidden_samples=sobol_hidden_samples,
        de_compare_points_per_slice=de_points,
        de_maxiter=de_maxiter,
        de_popsize=de_popsize,
        slices=slices,
    )
    return replace(config, doe=doe, envelope=envelope)


def _evaluations(config):
    rows = [
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
    return pd.DataFrame(rows)


def _verified_candidates(config):
    rows = [
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
            "predicted_phi": -0.9,
            "conservative_phi": -0.8,
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
            "predicted_phi": -0.01,
            "conservative_phi": 0.10,
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
            "predicted_phi": -0.1,
            "conservative_phi": 0.2,
        },
    ]
    return pd.DataFrame(rows)


def _policy_row(
    config,
    design_id,
    *,
    annual_intake,
    retention_rate,
    phi,
    feasible,
):
    row = {
        "design_id": design_id,
        "phi": phi,
        "feasible": feasible,
        "active_constraint": "total_pilots_window",
        "active_constraint_value": phi * 100.0,
        "annual_intake": annual_intake,
        "retention_rate": retention_rate,
        "ute": 16.0,
        "paa": 24,
        "max_manning_pct": 150.0,
        "flug_quota_per_phase": 5,
        "ipug_quota_per_phase": 5,
        "upgrade_sortie_fraction": 0.5,
        "flug_window_start": 250,
        "ipug_window_start": 400,
    }
    for name in config.policy.variables:
        row[f"raw_{name}"] = float(row[name])
        row[f"applied_{name}"] = row[name]
    return row


if __name__ == "__main__":
    unittest.main()
