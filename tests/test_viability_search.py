from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.viability.config import ViabilityConfig, load_config
from src.viability.doe import generate_doe
from src.viability.search import (
    filter_candidates_for_verify,
    filter_verified_constraints,
    generate_search_candidate_pool,
    require_search_config,
    run_surrogate_search,
    score_search_candidates,
    select_candidates_to_verify,
    verify_candidates,
)


class _FakeUncertaintyModel:
    def __init__(self, mu_phi, sigma_phi):
        self.mu_phi = np.asarray(mu_phi, dtype=float)
        self.sigma_phi = np.asarray(sigma_phi, dtype=float)

    def predict(self, x_values, return_std=False):
        if not return_std:
            return self.mu_phi[: len(x_values)]
        return self.mu_phi[: len(x_values)], self.sigma_phi[: len(x_values)]


def _fake_constraint_bundle(mu_by_constraint, sigma_by_constraint):
    constraint_names = list(mu_by_constraint)
    return {
        "models_by_constraint": {
            name: _FakeUncertaintyModel(mu_by_constraint[name], sigma_by_constraint[name])
            for name in constraint_names
        },
        "constraint_names": constraint_names,
        "target": "normalized_constraints",
        "feature_names": [
            "annual_intake",
            "retention_rate",
            "ute",
            "paa",
            "max_manning_pct",
            "flug_quota_per_phase",
            "ipug_quota_per_phase",
            "upgrade_sortie_fraction",
            "flug_window_start",
            "ipug_window_start",
        ],
    }


class ViabilitySearchTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/viability.example.yaml")

    def test_search_config_requires_all_fields_when_section_present(self):
        data = self.config.to_dict()
        del data["search"]["conservative_sigma"]

        with self.assertRaisesRegex(ValueError, "search.conservative_sigma"):
            ViabilityConfig.from_dict(data)

    def test_search_config_is_required_by_search_commands(self):
        config = replace(self.config, search=None)

        with self.assertRaisesRegex(ValueError, "search section"):
            require_search_config(config)

    def test_sobol_candidate_pool_uses_explicit_start_index(self):
        search_config = replace(
            require_search_config(self.config),
            candidate_start_index=512,
            candidate_pool_size=4,
        )

        candidates = generate_search_candidate_pool(self.config, search_config)

        self.assertEqual(candidates["sample_index"].tolist(), [512, 513, 514, 515])

    def test_scoring_uses_reconstructed_constraint_phi_and_conservative_phi(self):
        candidates = generate_doe(
            self.config,
            n=3,
            method="sobol",
            start_index=1024,
            include_corners=False,
            include_baselines=False,
        )
        bundle = _fake_constraint_bundle(
            {
                "total_pilots_final": [-0.50, 0.20, 0.01],
                "wg_rap": [-0.10, -0.30, 0.04],
            },
            {
                "total_pilots_final": [0.05, 0.10, 0.20],
                "wg_rap": [0.30, 0.02, 0.01],
            },
        )

        scored = score_search_candidates(
            bundle,
            candidates,
            self.config,
            conservative_sigma=2.0,
        )

        self.assertEqual(scored["predicted_phi"].round(6).tolist(), [-0.1, 0.2, 0.04])
        self.assertEqual(scored["conservative_phi"].round(6).tolist(), [0.5, 0.4, 0.41])
        self.assertEqual(scored["predicted_feasible"].tolist(), [True, False, False])
        self.assertEqual(scored["conservative_predicted_feasible"].tolist(), [False, False, False])
        self.assertEqual(scored.loc[0, "predicted_active_constraint"], "wg_rap")
        self.assertIn("mu_constraint_total_pilots_final", scored.columns)
        self.assertIn("sigma_constraint_wg_rap", scored.columns)

    def test_selection_dedupes_and_backfills_when_no_feasible_candidates_exist(self):
        candidates = pd.DataFrame(
            [
                _manual_design_row(self.config, "near_0", annual_intake=10),
                _manual_design_row(self.config, "near_1", annual_intake=20),
                _manual_design_row(self.config, "far_0", annual_intake=180),
                _manual_design_row(self.config, "far_1", annual_intake=350),
            ]
        )
        bundle = _fake_constraint_bundle(
            {
                "total_pilots_final": [1.0, 0.9, 0.4, 0.2],
                "wg_rap": [0.8, 0.7, 0.3, 0.1],
            },
            {
                "total_pilots_final": [0.05, 0.04, 0.03, 0.02],
                "wg_rap": [0.05, 0.04, 0.03, 0.02],
            },
        )
        scored = score_search_candidates(bundle, candidates, self.config, conservative_sigma=1.0)
        search_config = replace(
            require_search_config(self.config),
            n_candidates_to_verify=2,
            min_normalized_distance=0.05,
        )

        selected = select_candidates_to_verify(scored, self.config, search_config)

        self.assertEqual(len(selected), 2)
        self.assertEqual(selected["candidate_id"].tolist(), ["candidate_0000", "candidate_0001"])
        self.assertEqual(selected["design_id"].tolist(), ["far_1", "far_0"])
        expected_sources = {"minimum_predicted_violation", "backfill"}
        self.assertTrue(set(selected["selection_source"]).issubset(expected_sources))

    def test_run_surrogate_search_writes_artifacts(self):
        search_config = replace(
            require_search_config(self.config),
            candidate_start_index=2048,
            candidate_pool_size=16,
            n_candidates_to_verify=4,
            min_normalized_distance=0.0,
            candidate_report_rows=6,
        )
        config = replace(self.config, search=search_config)
        bundle = _fake_constraint_bundle(
            {
                "total_pilots_final": np.linspace(-0.4, 0.6, 16),
                "wg_rap": np.linspace(-0.6, 0.2, 16),
            },
            {
                "total_pilots_final": np.linspace(0.02, 0.20, 16),
                "wg_rap": np.linspace(0.20, 0.02, 16),
            },
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = run_surrogate_search(
                surrogate=bundle,
                surrogate_path=Path(tmp) / "surrogate_constraints_gpr.joblib",
                config=config,
                output_dir=tmp,
            )

            self.assertTrue(result.candidates_path.exists())
            self.assertTrue(result.scored_path.exists())
            self.assertTrue(result.summary_path.exists())
            self.assertEqual(result.candidate_count, 4)
            self.assertEqual(result.scored_count, 16)
            candidates = pd.read_csv(result.candidates_path)
            self.assertEqual(len(candidates), 4)
            self.assertIn("selection_source", candidates.columns)
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["selected_count"], 4)
            self.assertIn("mu_sigma", result.plot_paths)
            self.assertIn("selected_phi", result.plot_paths)

    def test_search_config_defaults_required_unallocated_iron_when_requirement_enabled(self):
        config = load_config("configs/viability/a_current_unit.yaml")
        search_config = require_search_config(config)
        self.assertEqual(search_config.required_constraints_for_verify, ("unallocated_iron",))

    def test_search_config_leaves_required_constraints_empty_when_iron_disabled(self):
        config = load_config("configs/viability.example.yaml")
        search_config = require_search_config(config)
        self.assertEqual(search_config.required_constraints_for_verify, ())

    def test_selection_only_uses_predicted_unallocated_iron_feasible_candidates(self):
        candidates = pd.DataFrame(
            [
                _manual_design_row(self.config, "iron_ok", annual_intake=10),
                _manual_design_row(self.config, "iron_bad", annual_intake=20),
                _manual_design_row(self.config, "iron_ok_2", annual_intake=30),
            ]
        )
        bundle = _fake_constraint_bundle(
            {
                "total_pilots_final": [0.4, 0.2, 0.3],
                "unallocated_iron": [-0.1, 0.2, -0.05],
            },
            {
                "total_pilots_final": [0.05, 0.05, 0.05],
                "unallocated_iron": [0.05, 0.05, 0.05],
            },
        )
        scored = score_search_candidates(bundle, candidates, self.config, conservative_sigma=1.0)
        search_config = replace(
            require_search_config(self.config),
            n_candidates_to_verify=2,
            min_normalized_distance=0.0,
            required_constraints_for_verify=("unallocated_iron",),
        )

        selected = select_candidates_to_verify(scored, self.config, search_config)

        self.assertEqual(len(selected), 2)
        self.assertEqual(set(selected["design_id"]), {"iron_ok", "iron_ok_2"})

    def test_verify_candidates_drops_rows_with_positive_unallocated_iron(self):
        candidates = pd.DataFrame(
            [
                {
                    **_manual_design_row(self.config, "iron_ok", annual_intake=350),
                    "candidate_id": "candidate_0000",
                    "selection_rank": 1,
                    "selection_source": "conservative_feasible",
                    "predicted_phi": -0.2,
                    "predicted_sigma_phi": 0.05,
                    "conservative_phi": -0.1,
                    "predicted_feasible": True,
                    "conservative_predicted_feasible": True,
                    "predicted_active_constraint": "total_pilots_final",
                    "mu_constraint_unallocated_iron": -0.1,
                },
                {
                    **_manual_design_row(self.config, "iron_bad", annual_intake=10),
                    "candidate_id": "candidate_0001",
                    "selection_rank": 2,
                    "selection_source": "near_boundary",
                    "predicted_phi": -0.1,
                    "predicted_sigma_phi": 0.2,
                    "conservative_phi": 0.1,
                    "predicted_feasible": True,
                    "conservative_predicted_feasible": False,
                    "predicted_active_constraint": "wg_rap",
                    "mu_constraint_unallocated_iron": 0.2,
                },
            ]
        )
        search_config = replace(
            require_search_config(self.config),
            n_candidates_to_verify=2,
            required_constraints_for_verify=("unallocated_iron",),
        )
        config = replace(self.config, search=search_config)

        with tempfile.TemporaryDirectory() as tmp:
            result = verify_candidates(
                candidates=candidates,
                config=config,
                output_dir=tmp,
                evaluator=_fake_evaluator,
            )

            self.assertTrue(result.verified_path.exists())
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["verified_count"], 1)
            self.assertEqual(summary["submitted_candidate_count"], 2)
            self.assertEqual(summary["skipped_predicted_infeasible_count"], 1)
            self.assertEqual(summary["dropped_after_verify_count"], 0)
            self.assertEqual(summary["best_candidate_id"], "candidate_0000")

    def test_verify_candidates_writes_verified_results_and_summary(self):
        candidates = pd.DataFrame(
            [
                {
                    **_manual_design_row(self.config, "feasible_predicted", annual_intake=350),
                    "candidate_id": "candidate_0000",
                    "selection_rank": 1,
                    "selection_source": "conservative_feasible",
                    "predicted_phi": -0.2,
                    "predicted_sigma_phi": 0.05,
                    "conservative_phi": -0.1,
                    "predicted_feasible": True,
                    "conservative_predicted_feasible": True,
                    "predicted_active_constraint": "total_pilots_final",
                },
                {
                    **_manual_design_row(self.config, "false_feasible", annual_intake=10),
                    "candidate_id": "candidate_0001",
                    "selection_rank": 2,
                    "selection_source": "near_boundary",
                    "predicted_phi": -0.1,
                    "predicted_sigma_phi": 0.2,
                    "conservative_phi": 0.1,
                    "predicted_feasible": True,
                    "conservative_predicted_feasible": False,
                    "predicted_active_constraint": "wg_rap",
                },
            ]
        )
        search_config = replace(require_search_config(self.config), n_candidates_to_verify=2)
        config = replace(self.config, search=search_config)

        with tempfile.TemporaryDirectory() as tmp:
            result = verify_candidates(
                candidates=candidates,
                config=config,
                output_dir=tmp,
                evaluator=_fake_evaluator,
            )

            self.assertTrue(result.verified_path.exists())
            self.assertTrue(result.summary_path.exists())
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["verified_count"], 2)
            self.assertEqual(summary["verified_feasible_count"], 1)
            self.assertEqual(summary["false_feasible_count"], 1)
            self.assertEqual(summary["false_conservative_feasible_count"], 0)
            self.assertEqual(summary["best_candidate_id"], "candidate_0000")
            self.assertIn("predicted_vs_verified_phi", result.plot_paths)

    def test_a_current_config_defaults_to_unallocated_iron_verify_requirement(self):
        config = load_config("configs/viability/a_current_unit.yaml")
        search_config = require_search_config(config)
        self.assertEqual(search_config.required_constraints_for_verify, ("unallocated_iron",))

    def test_selection_skips_predicted_unallocated_iron_violations(self):
        candidates = pd.DataFrame(
            [
                _manual_design_row(self.config, "iron_ok", annual_intake=10),
                _manual_design_row(self.config, "iron_bad", annual_intake=20),
                _manual_design_row(self.config, "iron_ok_2", annual_intake=30),
                _manual_design_row(self.config, "iron_bad_2", annual_intake=40),
            ]
        )
        bundle = _fake_constraint_bundle(
            {
                "total_pilots_final": [0.4, 0.2, 0.3, 0.1],
                "unallocated_iron": [-0.05, 0.20, -0.01, 0.15],
            },
            {
                "total_pilots_final": [0.05, 0.04, 0.03, 0.02],
                "unallocated_iron": [0.01, 0.02, 0.01, 0.02],
            },
        )
        scored = score_search_candidates(bundle, candidates, self.config, conservative_sigma=1.0)
        search_config = replace(
            require_search_config(self.config),
            n_candidates_to_verify=2,
            min_normalized_distance=0.0,
            required_constraints_for_verify=("unallocated_iron",),
        )

        selected = select_candidates_to_verify(scored, self.config, search_config)

        self.assertEqual(len(selected), 2)
        self.assertTrue((selected["mu_constraint_unallocated_iron"] <= 0.0).all())
        self.assertEqual(set(selected["design_id"]), {"iron_ok", "iron_ok_2"})

    def test_verify_keeps_only_direct_evaluations_with_zero_unallocated_iron(self):
        candidates = pd.DataFrame(
            [
                {
                    **_manual_design_row(self.config, "iron_ok", annual_intake=350),
                    "candidate_id": "candidate_0000",
                    "selection_rank": 1,
                    "selection_source": "conservative_feasible",
                    "predicted_phi": -0.2,
                    "predicted_sigma_phi": 0.05,
                    "conservative_phi": -0.1,
                    "predicted_feasible": True,
                    "conservative_predicted_feasible": True,
                    "predicted_active_constraint": "total_pilots_final",
                    "mu_constraint_unallocated_iron": -0.05,
                },
                {
                    **_manual_design_row(self.config, "iron_bad", annual_intake=10),
                    "candidate_id": "candidate_0001",
                    "selection_rank": 2,
                    "selection_source": "near_boundary",
                    "predicted_phi": -0.1,
                    "predicted_sigma_phi": 0.2,
                    "conservative_phi": 0.1,
                    "predicted_feasible": True,
                    "conservative_predicted_feasible": False,
                    "predicted_active_constraint": "wg_rap",
                    "mu_constraint_unallocated_iron": 0.20,
                },
            ]
        )
        search_config = replace(
            require_search_config(self.config),
            n_candidates_to_verify=2,
            required_constraints_for_verify=("unallocated_iron",),
        )
        config = replace(self.config, search=search_config)

        filtered, skipped = filter_candidates_for_verify(
            candidates,
            search_config.required_constraints_for_verify,
        )
        self.assertEqual(skipped, 1)
        self.assertEqual(len(filtered), 1)

        with tempfile.TemporaryDirectory() as tmp:
            result = verify_candidates(
                candidates=candidates,
                config=config,
                output_dir=tmp,
                evaluator=_fake_evaluator,
            )

            self.assertEqual(result.verified_count, 1)
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["skipped_predicted_infeasible_count"], 1)
            self.assertEqual(summary["required_constraints_for_verify"], ["unallocated_iron"])
            verified = pd.read_parquet(result.verified_path)
            self.assertEqual(verified["design_id"].tolist(), ["iron_ok"])
            self.assertTrue((verified["constraint_unallocated_iron"] <= 0.0).all())


def _fake_evaluator(
    designs,
    config,
    workers=None,
    checkpoint_dir=None,
    checkpoint_every=50,
):
    rows = []
    for _, row in designs.iterrows():
        phi = _phi(row)
        output = {
            "design_id": row["design_id"],
            "status": "ok",
            "phi": phi,
            "feasible": phi <= 0.0,
            "active_constraint": "total_pilots_final",
            "active_constraint_value": phi * 100.0,
            "constraint_total_pilots_final": phi * 100.0,
            "constraint_wg_rap": phi - 0.3,
            "constraint_unallocated_iron": -0.1 if row["design_id"] == "iron_ok" else 0.2,
        }
        for name in config.policy.variables:
            output[name] = row[name]
            output[f"applied_{name}"] = row[f"applied_{name}"]
            output[f"raw_{name}"] = row[f"raw_{name}"]
        rows.append(output)
    return pd.DataFrame(rows)


def _phi(row) -> float:
    annual = float(row["annual_intake"])
    return 0.4 - annual / 350.0


def _manual_design_row(config, design_id: str, *, annual_intake: int) -> dict:
    row = {
        "design_id": design_id,
        "doe_source": "manual",
        "sample_index": 0,
        "annual_intake": annual_intake,
        "retention_rate": 0.3,
        "ute": 12.0,
        "paa": 24,
        "max_manning_pct": 150.0,
        "flug_quota_per_phase": 3,
        "ipug_quota_per_phase": 2,
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
