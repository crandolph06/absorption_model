from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.viability.active_learning import (
    exclude_holdout_rows,
    generate_candidate_pool,
    remove_existing_designs,
    require_active_learning_config,
    run_active_learning,
    score_candidates,
    select_candidate_batch,
    sort_boundary_candidates,
    sort_scored_candidates,
)
from src.viability.config import ViabilityConfig, load_config
from src.viability.doe import generate_doe


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


class ViabilityActiveLearningTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/viability.example.yaml")

    def test_active_learning_config_requires_all_fields_when_section_present(self):
        data = self.config.to_dict()
        del data["active_learning"]["boundary_batch_fraction"]

        with self.assertRaisesRegex(ValueError, "active_learning.boundary_batch_fraction"):
            ViabilityConfig.from_dict(data)

    def test_active_learning_config_is_required_by_active_learn(self):
        config = replace(self.config, active_learning=None)

        with self.assertRaisesRegex(ValueError, "active_learning section"):
            require_active_learning_config(config)

    def test_sobol_candidate_pool_continues_from_explicit_start_index(self):
        active_config = replace(
            require_active_learning_config(self.config),
            candidate_start_index=8,
            candidate_pool_size=4,
        )

        first = generate_candidate_pool(
            config=self.config,
            active_config=active_config,
            start_index=active_config.candidate_start_index,
        )
        second = generate_candidate_pool(
            config=self.config,
            active_config=active_config,
            start_index=active_config.candidate_start_index + active_config.candidate_pool_size,
        )

        self.assertEqual(first["sample_index"].tolist(), [8, 9, 10, 11])
        self.assertEqual(second["sample_index"].tolist(), [12, 13, 14, 15])

    def test_dedupe_removes_existing_and_holdout_policy_tuples(self):
        candidates = generate_doe(
            self.config,
            n=8,
            method="sobol",
            start_index=16,
            include_corners=False,
            include_baselines=False,
        )
        existing = candidates.iloc[[0, 3]].copy()
        holdout = candidates.iloc[[4]].copy()

        filtered = remove_existing_designs(candidates, [existing, holdout], self.config)

        self.assertEqual(len(filtered), 5)
        removed_ids = set(existing["design_id"].tolist() + holdout["design_id"].tolist())
        self.assertFalse(removed_ids.intersection(set(filtered["design_id"].tolist())))

    def test_exclude_holdout_keeps_fixed_holdout_out_of_training(self):
        evaluations = _synthetic_evaluations(self.config, n=16, start_index=0)
        holdout = evaluations.iloc[[1, 7]].copy()

        training = exclude_holdout_rows(evaluations, holdout, self.config)

        self.assertEqual(len(training), 14)
        self.assertFalse(set(holdout["design_id"].tolist()).intersection(set(training["design_id"].tolist())))

    def test_uncertainty_sort_uses_boundary_distance_as_tie_breaker(self):
        candidates = generate_doe(
            self.config,
            n=4,
            method="sobol",
            start_index=32,
            include_corners=False,
            include_baselines=False,
        )
        model = _fake_constraint_bundle(
            {
                "total_pilots_final": [0.50, -0.20, 0.05, 0.10],
                "wg_rap": [0.20, -0.40, -0.10, -0.20],
            },
            {
                "total_pilots_final": [0.10, 0.80, 0.80, 0.20],
                "wg_rap": [0.05, 0.30, 0.40, 0.10],
            },
        )

        scored = score_candidates(model, candidates, self.config)
        ordered = sort_scored_candidates(scored)

        self.assertEqual(ordered.iloc[0]["design_id"], candidates.iloc[2]["design_id"])
        self.assertEqual(ordered.iloc[1]["design_id"], candidates.iloc[1]["design_id"])
        self.assertIn("predicted_active_constraint", scored.columns)
        self.assertIn("mu_constraint_total_pilots_final", scored.columns)
        self.assertIn("sigma_constraint_wg_rap", scored.columns)

    def test_boundary_sort_uses_abs_mu_phi_first(self):
        candidates = generate_doe(
            self.config,
            n=4,
            method="sobol",
            start_index=40,
            include_corners=False,
            include_baselines=False,
        )
        model = _FakeUncertaintyModel(
            mu_phi=[3.0, 0.03, -0.02, 1.0],
            sigma_phi=[5.0, 0.1, 0.2, 0.3],
        )

        scored = score_candidates(model, candidates, self.config)
        ordered = sort_boundary_candidates(scored)

        self.assertEqual(ordered.iloc[0]["design_id"], candidates.iloc[2]["design_id"])
        self.assertEqual(ordered.iloc[1]["design_id"], candidates.iloc[1]["design_id"])

    def test_diversity_filter_skips_near_duplicate_batch_candidate(self):
        candidates = pd.DataFrame(
            [
                _manual_design_row(self.config, "near_0", annual_intake=10),
                _manual_design_row(self.config, "near_1", annual_intake=20),
                _manual_design_row(self.config, "far_0", annual_intake=350),
            ]
        )
        model = _FakeUncertaintyModel(
            mu_phi=[0.4, 0.3, 0.2],
            sigma_phi=[3.0, 2.0, 1.0],
        )
        scored = score_candidates(model, candidates, self.config)
        active_config = replace(
            require_active_learning_config(self.config),
            acquisition="uncertainty",
            boundary_batch_fraction=0.0,
            batch_size=2,
            min_normalized_distance=0.05,
        )

        selected = select_candidate_batch(
            scored,
            self.config,
            active_config,
        )

        self.assertEqual(selected["design_id"].tolist(), ["near_0", "far_0"])
        self.assertEqual(selected["selection_source"].tolist(), ["uncertainty", "uncertainty"])

    def test_boundary_stratified_selection_splits_boundary_and_uncertainty_rows(self):
        candidates = pd.DataFrame(
            [
                _manual_design_row(self.config, "boundary_0", annual_intake=10),
                _manual_design_row(self.config, "boundary_1", annual_intake=60),
                _manual_design_row(self.config, "uncertain_0", annual_intake=300),
                _manual_design_row(self.config, "uncertain_1", annual_intake=350),
            ]
        )
        model = _FakeUncertaintyModel(
            mu_phi=[0.01, -0.02, 5.0, 6.0],
            sigma_phi=[0.1, 0.2, 4.0, 3.0],
        )
        scored = score_candidates(model, candidates, self.config)
        active_config = replace(
            require_active_learning_config(self.config),
            acquisition="boundary_stratified_uncertainty",
            boundary_batch_fraction=0.5,
            batch_size=4,
            min_normalized_distance=0.0,
        )

        selected = select_candidate_batch(scored, self.config, active_config)

        self.assertEqual(
            selected["design_id"].tolist(),
            ["boundary_0", "boundary_1", "uncertain_0", "uncertain_1"],
        )
        self.assertEqual(
            selected["selection_source"].tolist(),
            ["boundary", "boundary", "uncertainty", "uncertainty"],
        )

    def test_run_active_learning_writes_state_metrics_and_artifacts(self):
        active_config = replace(
            require_active_learning_config(self.config),
            candidate_start_index=64,
            candidate_pool_size=16,
            iterations=1,
            batch_size=2,
            min_normalized_distance=0.0,
            candidate_report_rows=5,
        )
        config = replace(self.config, active_learning=active_config)
        evaluations = _synthetic_evaluations(config, n=32, start_index=0)
        holdout = _synthetic_evaluations(config, n=4, start_index=128)

        with tempfile.TemporaryDirectory() as tmp:
            result = run_active_learning(
                evaluations=evaluations,
                holdout=holdout,
                config=config,
                output_dir=tmp,
                evaluator=_fake_evaluator,
            )

            self.assertTrue(result.state_path.exists())
            self.assertTrue(result.metrics_path.exists())
            self.assertTrue(result.latest_training_path.exists())
            self.assertTrue(result.latest_model_path.exists())
            self.assertEqual(result.latest_model_path.name, "surrogate_constraints_gpr.joblib")
            self.assertIn("holdout_metrics", result.plot_paths)
            self.assertIn("predict_vs_truth_normalized", result.plot_paths)
            self.assertIn("selected_mu_sigma", result.plot_paths)
            self.assertEqual(result.metrics_table["iteration"].tolist(), [0, 1])

            selected = pd.read_csv(f"{tmp}/iteration_001/selected_candidates.csv")
            self.assertEqual(len(selected), 2)
            self.assertEqual(len(set(selected["design_id"].tolist())), 2)
            selected_path = Path(tmp) / "iteration_001" / "selected_evaluations.parquet"
            if selected_path.exists():
                selected_evaluations = pd.read_parquet(selected_path)
            else:
                selected_evaluations = pd.read_csv(selected_path.with_suffix(".csv"))
            self.assertEqual(len(selected_evaluations), 2)

            with self.assertRaisesRegex(FileExistsError, "--resume"):
                run_active_learning(
                    evaluations=evaluations,
                    holdout=holdout,
                    config=config,
                    output_dir=tmp,
                    evaluator=_fake_evaluator,
                )

    def test_resume_rejects_changed_candidate_sequence_config(self):
        active_config = replace(
            require_active_learning_config(self.config),
            candidate_start_index=96,
            candidate_pool_size=8,
            iterations=0,
            batch_size=2,
            min_normalized_distance=0.0,
            candidate_report_rows=5,
        )
        config = replace(self.config, active_learning=active_config)
        evaluations = _synthetic_evaluations(config, n=16, start_index=0)
        holdout = _synthetic_evaluations(config, n=4, start_index=192)

        with tempfile.TemporaryDirectory() as tmp:
            run_active_learning(
                evaluations=evaluations,
                holdout=holdout,
                config=config,
                output_dir=tmp,
                evaluator=_fake_evaluator,
            )
            changed_config = replace(
                config,
                active_learning=replace(active_config, candidate_pool_size=16),
            )

            with self.assertRaisesRegex(ValueError, "config_hash"):
                run_active_learning(
                    evaluations=evaluations,
                    holdout=holdout,
                    config=changed_config,
                    output_dir=tmp,
                    resume=True,
                    evaluator=_fake_evaluator,
                )


def _synthetic_evaluations(config, n: int, start_index: int) -> pd.DataFrame:
    frame = generate_doe(
        config,
        n=n,
        method="sobol",
        start_index=start_index,
        include_corners=False,
        include_baselines=False,
    )
    return _evaluation_rows(frame, config)


def _fake_evaluator(
    designs,
    config,
    workers=None,
    checkpoint_dir=None,
    checkpoint_every=50,
):
    return _evaluation_rows(designs, config)


def _evaluation_rows(frame: pd.DataFrame, config) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        phi = _phi(row)
        output = {
            **row.to_dict(),
            "status": "ok",
            "phi": phi,
            "feasible": phi <= 0.0,
            "active_constraint": "total_pilots_final",
            "active_constraint_value": phi * 100.0,
            "constraint_total_pilots_final": phi * 100.0,
            "constraint_total_pilots_window": (phi - 0.2) * 100.0,
            "constraint_wg_rap": phi - 0.3,
            "constraint_fl_rap": phi - 0.4,
            "constraint_ip_rap": phi - 0.5,
        }
        for name in config.policy.variables:
            output[f"applied_{name}"] = row[name]
            if f"raw_{name}" in row:
                output[f"raw_{name}"] = row[f"raw_{name}"]
        rows.append(output)
    return pd.DataFrame(rows)


def _phi(row) -> float:
    annual = float(row["annual_intake"])
    retention = float(row["retention_rate"])
    ute = float(row["ute"])
    max_manning = float(row["max_manning_pct"])
    return 1.3 - annual / 240.0 - retention + (12.0 - ute) / 24.0 - (max_manning - 100.0) / 300.0


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
