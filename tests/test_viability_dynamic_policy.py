from dataclasses import replace
import unittest
from unittest.mock import patch

import pandas as pd

from src.viability.config import load_config
from src.viability.dynamic_policy import (
    EpochPolicySchedule,
    dynamic_feature_names,
    epoch_for_phase_index,
    schedule_from_unit_vector,
)
from src.viability.evaluator import (
    EvaluationResult,
    evaluate_design,
    evaluate_policy_schedule,
    evaluate_schedules_parallel,
    simulate_policy_schedule_history,
)
from src.viability.policy import PolicyDesign


class ViabilityDynamicPolicyTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/viability.example.yaml")

    def test_epoch_mapping_partitions_horizon_evenly(self):
        self.assertEqual(epoch_for_phase_index(0, 60, 3), 0)
        self.assertEqual(epoch_for_phase_index(19, 60, 3), 0)
        self.assertEqual(epoch_for_phase_index(20, 60, 3), 1)
        self.assertEqual(epoch_for_phase_index(39, 60, 3), 1)
        self.assertEqual(epoch_for_phase_index(40, 60, 3), 2)
        self.assertEqual(epoch_for_phase_index(59, 60, 3), 2)

    def test_schedule_from_unit_vector_rounds_integer_policy_values(self):
        schedule = schedule_from_unit_vector(
            [0.0] * (len(self.config.policy.variables) * 3),
            self.config.policy,
            epoch_count=3,
            total_phases=60,
        )

        self.assertEqual(schedule.epoch_count, 3)
        self.assertEqual(schedule.epoch_designs[0].annual_intake, 10)
        self.assertEqual(schedule.epoch_designs[0].paa, 18)
        self.assertEqual(schedule.epoch_designs[0].flug_quota_per_phase, 0)

    def test_flat_schedule_mapping_round_trips_to_epoch_designs(self):
        values = {}
        for name in dynamic_feature_names(self.config.policy, 3):
            if name.endswith("annual_intake"):
                values[name] = 250
            elif name.endswith("retention_rate"):
                values[name] = 0.5
            elif name.endswith("ute"):
                values[name] = 12
            elif name.endswith("paa"):
                values[name] = 24
            elif name.endswith("max_manning_pct"):
                values[name] = 150
            elif name.endswith("flug_quota_per_phase"):
                values[name] = 3
            elif name.endswith("ipug_quota_per_phase"):
                values[name] = 2
            elif name.endswith("upgrade_sortie_fraction"):
                values[name] = 0.5
            elif name.endswith("flug_window_start"):
                values[name] = 250
            elif name.endswith("ipug_window_start"):
                values[name] = 400

        schedule = EpochPolicySchedule.from_flat_mapping(
            values,
            self.config.policy,
            epoch_count=3,
            total_phases=60,
        )

        self.assertEqual(schedule.policy_for_phase_index(42).annual_intake, 250)
        self.assertEqual(schedule.to_flat_dict()["epoch2_paa"], 24)

    def test_dynamic_simulation_applies_epoch_policy_values(self):
        captured = []

        class FakeSimulation:
            def __init__(self, **kwargs):
                self.history = []
                self.squadrons = []
                self.current_year = 2026
                self.current_phase = 1
                self.annual_intake = kwargs["annual_intake"]
                self.retention_rate = kwargs["retention_rate"]
                self.max_manning = kwargs["max_manning_pct"] / 100
                self.sq_phase_flug_intake = None
                self.sq_phase_ipug_intake = None
                self.sim_config = kwargs["sim_config"]

            def run_phase(self, phase_num, year):
                captured.append(
                    (
                        year,
                        phase_num,
                        self.annual_intake,
                        self.retention_rate,
                        self.max_manning,
                        self.sq_phase_flug_intake,
                        self.sq_phase_ipug_intake,
                    )
                )
                self.history.append(
                    {
                        "year": year,
                        "phase": phase_num,
                        "total_pilots": 5000,
                        "line_pilots": 4000,
                        "staff_ips": 0,
                        "staff_fls": 0,
                        "wg_rap_shortfall": 0.0,
                        "fl_rap_shortfall": 0.0,
                        "ip_rap_shortfall": 0.0,
                    }
                )

        class FakeSquadron:
            paa = 0
            ute = 0

            def update_stats(self):
                return None

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
        values = {}
        for epoch, intake in enumerate([100, 200, 300], start=1):
            values[f"epoch{epoch}_annual_intake"] = intake
            values[f"epoch{epoch}_retention_rate"] = 0.5
            values[f"epoch{epoch}_ute"] = 12
            values[f"epoch{epoch}_paa"] = 24
            values[f"epoch{epoch}_max_manning_pct"] = 150
            values[f"epoch{epoch}_flug_quota_per_phase"] = epoch
            values[f"epoch{epoch}_ipug_quota_per_phase"] = 0
            values[f"epoch{epoch}_upgrade_sortie_fraction"] = 0.5
            values[f"epoch{epoch}_flug_window_start"] = 250
            values[f"epoch{epoch}_ipug_window_start"] = 400
        schedule = EpochPolicySchedule.from_flat_mapping(
            values,
            physics_config.policy,
            epoch_count=3,
            total_phases=3,
        )

        with (
            patch("src.viability.evaluator.CAFSimulation", FakeSimulation),
            patch("src.viability.evaluator.get_initial_squadrons", return_value=[FakeSquadron()]),
        ):
            history = simulate_policy_schedule_history(schedule, physics_config)

        self.assertEqual(len(history), 3)
        self.assertEqual([row[2] for row in captured], [100, 200, 300])
        self.assertEqual([row[5] for row in captured], [1, 2, 3])

    def test_constant_epoch_schedule_matches_constant_physics_evaluator(self):
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
        values = {
            "annual_intake": 150,
            "retention_rate": 0.55,
            "ute": 20,
            "paa": 30,
            "max_manning_pct": 180,
            "flug_quota_per_phase": 2,
            "ipug_quota_per_phase": 0,
            "upgrade_sortie_fraction": 0.5,
            "flug_window_start": 250,
            "ipug_window_start": 400,
        }
        design = PolicyDesign.from_mapping(values, physics_config.policy)
        schedule_values = {}
        for epoch in range(1, 4):
            for name, value in values.items():
                schedule_values[f"epoch{epoch}_{name}"] = value
        schedule = EpochPolicySchedule.from_flat_mapping(
            schedule_values,
            physics_config.policy,
            epoch_count=3,
            total_phases=physics_config.model.years_to_run * 3,
        )

        constant = evaluate_design(design, physics_config, seed=1234)
        dynamic = evaluate_policy_schedule(schedule, physics_config, seed=1234)

        self.assertEqual(constant.status, "ok")
        self.assertEqual(dynamic.status, "ok")
        self.assertEqual(dynamic.phase_backend, "physics")
        self.assertAlmostEqual(dynamic.phi, constant.phi, places=12)
        self.assertEqual(dynamic.feasible, constant.feasible)
        self.assertEqual(dynamic.active_constraint, constant.active_constraint)
        self.assertAlmostEqual(
            dynamic.active_constraint_value,
            constant.active_constraint_value,
            places=12,
        )
        for name, value in constant.constraints.items():
            self.assertIn(name, dynamic.constraints)
            self.assertAlmostEqual(dynamic.constraints[name], value, places=12)
        for name in [
            "final_total_pilots",
            "min_total_pilots_after_assessment_start",
            "max_wg_rap_shortfall_after_assessment_start",
            "max_fl_rap_shortfall_after_assessment_start",
            "max_ip_rap_shortfall_after_assessment_start",
        ]:
            self.assertIn(name, dynamic.raw_metrics)
            self.assertAlmostEqual(
                dynamic.raw_metrics[name],
                constant.raw_metrics[name],
                places=12,
            )

    def test_parallel_schedule_evaluation_keeps_metadata(self):
        feature_names = dynamic_feature_names(self.config.policy, 3)
        row = {"schedule_id": "s0", "schedule_source": "unit", "sample_index": 17}
        for name in feature_names:
            row[name] = 0.5
        schedules = pd.DataFrame([row])

        def fake_job(job):
            schedule_id, _values, _raw, metadata, _config, _epochs, _phases, _seed = job
            return (
                schedule_id,
                metadata,
                EvaluationResult(
                    design={name: 0.5 for name in feature_names},
                    raw_design={name: 0.5 for name in feature_names},
                    applied_design={name: 0.5 for name in feature_names},
                    raw_metrics={},
                    constraints={"total_pilots_final": -1.0},
                    phi=-0.01,
                    feasible=True,
                    active_constraint="total_pilots_final",
                    active_constraint_value=-1.0,
                    status="ok",
                    phase_backend="physics",
                ),
            )

        with patch("src.viability.evaluator._evaluate_schedule_job", side_effect=fake_job):
            result = evaluate_schedules_parallel(schedules, self.config, epoch_count=3, workers=1)

        self.assertEqual(result.loc[0, "schedule_id"], "s0")
        self.assertEqual(result.loc[0, "schedule_source"], "unit")
        self.assertEqual(result.loc[0, "sample_index"], 17)
        self.assertEqual(result.loc[0, "phase_backend"], "physics")


if __name__ == "__main__":
    unittest.main()
