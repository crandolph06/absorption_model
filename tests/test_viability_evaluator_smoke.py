from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.viability.config import load_config
from src.viability.evaluator import (
    EvaluationResult,
    _flatten_result,
    _load_brain,
    _validate_brain_output,
    evaluate_design,
    evaluate_designs_parallel,
    simulate_design_history,
)
from src.viability.policy import PolicyDesign


class _FakeBrain:
    def __init__(self, outputs):
        self.outputs = outputs

    def predict(self, x):
        return np.zeros((len(x), self.outputs))


class ViabilityEvaluatorSmokeTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/viability.example.yaml")
        self.variable_names = list(self.config.policy.variables)

    def test_brain_validation_rejects_legacy_12_output_layout(self):
        with self.assertRaisesRegex(ValueError, "expected 16 outputs, got 12"):
            _validate_brain_output(_FakeBrain(outputs=12), expected_outputs=16)

    def test_flatten_result_keeps_design_metrics_and_constraints_readable(self):
        design = PolicyDesign.from_mapping(
            {
                "annual_intake": 250,
                "retention_rate": 0.5,
                "ute": 12,
                "paa": 24,
                "max_manning_pct": 150,
                "flug_quota_per_phase": 3,
                "ipug_quota_per_phase": 2,
            },
            self.config.policy,
            raw_values={
                "annual_intake": 250.4,
                "retention_rate": 0.5,
                "ute": 12.0,
                "paa": 24.2,
                "max_manning_pct": 150.0,
                "flug_quota_per_phase": 3.1,
                "ipug_quota_per_phase": 2.1,
            },
        )
        result = evaluate_design(design, self.config)
        row = _flatten_result(7, result, self.variable_names)

        self.assertEqual(row["design_id"], 7)
        self.assertEqual(row["applied_annual_intake"], 250)
        self.assertAlmostEqual(row["raw_annual_intake"], 250.4)
        self.assertIn("active_constraint", row)
        self.assertIn("active_constraint_value", row)
        self.assertIn("status", row)
        self.assertIn("error", row)
        self.assertEqual(row["phase_backend"], self.config.model.phase_backend)

    def test_flatten_result_records_phase_backend(self):
        result = EvaluationResult(
            design={},
            raw_design={},
            applied_design={},
            raw_metrics={},
            constraints={},
            phi=0.0,
            feasible=True,
            active_constraint=None,
            active_constraint_value=None,
            status="ok",
            phase_backend="physics",
        )

        row = _flatten_result(3, result, self.variable_names)

        self.assertEqual(row["phase_backend"], "physics")

    def test_physics_backend_does_not_load_brain_and_enables_allocator(self):
        captured = {}

        class FakeSimulation:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.current_year = 2026
                self.current_phase = 1
                self.sq_phase_flug_intake = None
                self.sq_phase_ipug_intake = None

            def run_simulation(self, years_to_run, squadron_configs, ute):
                return pd.DataFrame(
                    [
                        {
                            "year": 2026,
                            "phase": 1,
                            "total_pilots": 100,
                            "line_pilots": 80,
                            "staff_ips": 1,
                            "staff_fls": 2,
                            "wg_rap_shortfall": 0.0,
                            "fl_rap_shortfall": 0.0,
                            "ip_rap_shortfall": 0.0,
                        }
                    ]
                )

        physics_config = replace(
            self.config,
            model=replace(
                self.config.model,
                phase_backend="physics",
                brain_path=None,
                expected_brain_outputs=None,
                years_to_run=1,
                assessment_start_year=self.config.model.start_year,
                target_year=self.config.model.start_year,
            ),
        )
        design = PolicyDesign.from_mapping(
            {
                "annual_intake": 250,
                "retention_rate": 0.5,
                "ute": 12,
                "paa": 24,
                "max_manning_pct": 150,
                "flug_quota_per_phase": 3,
                "ipug_quota_per_phase": 2,
            },
            physics_config.policy,
        )

        with (
            patch("src.viability.evaluator._load_brain") as load_brain,
            patch("src.viability.evaluator.CAFSimulation", FakeSimulation),
        ):
            history = simulate_design_history(design, physics_config)

        load_brain.assert_not_called()
        self.assertFalse(history.empty)
        self.assertIsNone(captured["brain"])
        self.assertTrue(captured["use_physics_allocator"])
        self.assertIs(captured["sim_config"], physics_config.model.simulation)

    def test_parallel_evaluation_keeps_doe_metadata_and_sample_seed(self):
        values = {
            "annual_intake": 250,
            "retention_rate": 0.5,
            "ute": 12,
            "paa": 24,
            "max_manning_pct": 150,
            "flug_quota_per_phase": 3,
            "ipug_quota_per_phase": 2,
        }
        designs = pd.DataFrame(
            [
                {
                    "design_id": "sobol_001024",
                    "doe_source": "sobol",
                    "sample_index": 1024,
                    **values,
                }
            ]
        )
        captured_jobs = []

        def fake_job(job):
            captured_jobs.append(job)
            design_id, _values, raw_values, metadata, _config, _seed = job
            return (
                design_id,
                metadata,
                EvaluationResult(
                    design=_values,
                    raw_design=raw_values or _values,
                    applied_design=_values,
                    raw_metrics={},
                    constraints={"total_pilots_final": -1.0},
                    phi=-0.01,
                    feasible=True,
                    active_constraint="total_pilots_final",
                    active_constraint_value=-1.0,
                    status="ok",
                ),
            )

        with patch("src.viability.evaluator._evaluate_design_job", side_effect=fake_job):
            result = evaluate_designs_parallel(designs, self.config, workers=1)

        self.assertEqual(result.loc[0, "sample_index"], 1024)
        self.assertEqual(result.loc[0, "doe_source"], "sobol")
        self.assertEqual(captured_jobs[0][-1], self.config.run.random_seed + 1024)

    def test_evaluate_design_one_year_when_compatible_brain_is_available(self):
        brain_path = Path(self.config.model.brain_path)
        if not brain_path.exists():
            self.skipTest(f"Configured brain is not available: {brain_path}")
        try:
            _validate_brain_output(_load_brain(self.config.model.brain_path), 16)
        except ValueError as exc:
            self.skipTest(str(exc))

        smoke_config = replace(
            self.config,
            model=replace(
                self.config.model,
                years_to_run=1,
                assessment_start_year=self.config.model.start_year,
                target_year=self.config.model.start_year,
            ),
        )
        design = PolicyDesign.from_mapping(
            {
                "annual_intake": 250,
                "retention_rate": 0.5,
                "ute": 12,
                "paa": 24,
                "max_manning_pct": 150,
                "flug_quota_per_phase": 3,
                "ipug_quota_per_phase": 2,
            },
            smoke_config.policy,
        )

        result = evaluate_design(design, smoke_config)

        self.assertEqual(result.status, "ok", result.error)
        self.assertIn("final_total_pilots", result.raw_metrics)
        self.assertIn("total_pilots_final", result.constraints)
        self.assertIsNotNone(result.active_constraint)


if __name__ == "__main__":
    unittest.main()
