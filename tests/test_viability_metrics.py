import math
import unittest

import pandas as pd

from src.viability.config import ConstraintScalesConfig, RequirementsConfig
from src.viability.metrics import (
    aggregate_violation,
    compute_constraints,
    compute_raw_metrics,
)


class ViabilityMetricsTest(unittest.TestCase):
    def test_metrics_aggregate_squadron_history_by_phase(self):
        history = pd.DataFrame(
            [
                _row(2040, 1, 1, total=100, line=90, ip=20, fl=30, wg_short=1.0),
                _row(2040, 1, 2, total=120, line=100, ip=20, fl=30, wg_short=3.0),
                _row(2040, 2, 1, total=110, line=95, ip=25, fl=30, wg_short=0.0),
                _row(2040, 2, 2, total=130, line=105, ip=25, fl=30, wg_short=-2.0),
            ]
        )

        metrics = compute_raw_metrics(history, assessment_start_year=2040)

        self.assertEqual(metrics["final_total_pilots"], 240.0)
        self.assertEqual(metrics["final_line_pilots"], 200.0)
        self.assertEqual(metrics["min_total_pilots_after_assessment_start"], 220.0)
        self.assertEqual(metrics["max_wg_rap_shortfall_after_assessment_start"], 2.0)
        self.assertEqual(metrics["mean_wg_rap_shortfall_after_assessment_start"], 0.5)
        self.assertTrue(
            math.isclose(
                metrics["min_experience_ratio_after_assessment_start"],
                100.0 / 190.0,
            )
        )

    def test_rap_metrics_preserve_negative_slack_above_target(self):
        history = pd.DataFrame(
            [
                _row(2040, 1, 1, total=100, line=90, ip=20, fl=30, wg_short=-1.0),
                _row(2040, 1, 2, total=120, line=100, ip=20, fl=30, wg_short=-3.0),
                _row(2040, 2, 1, total=110, line=95, ip=25, fl=30, wg_short=-2.0),
                _row(2040, 2, 2, total=130, line=105, ip=25, fl=30, wg_short=-4.0),
            ]
        )

        metrics = compute_raw_metrics(history, assessment_start_year=2040)
        requirements = RequirementsConfig(
            target_total_pilots=None,
            target_line_pilots=None,
            min_experience_ratio=None,
            allowed_wg_rap_shortfall=0.0,
            allowed_fl_rap_shortfall=None,
            allowed_ip_rap_shortfall=None,
            allowed_utc_1_wg_shortfall=None,
            allowed_utc_1_fl_shortfall=None,
            allowed_utc_2_wg_shortfall=None,
            allowed_utc_2_fl_shortfall=None,
            target_staff_ips=None,
            target_staff_fls=None,
            allowed_unallocated_iron=None,
        )

        constraints = compute_constraints(metrics, requirements)

        self.assertEqual(metrics["max_wg_rap_shortfall_after_assessment_start"], -2.0)
        self.assertEqual(metrics["mean_wg_rap_shortfall_after_assessment_start"], -2.5)
        self.assertEqual(constraints["wg_rap"], -2.0)

    def test_staff_constraints_use_window_minimums(self):
        history = pd.DataFrame(
            [
                _row(
                    2040,
                    1,
                    1,
                    total=100,
                    line=90,
                    ip=20,
                    fl=30,
                    wg_short=0.0,
                    staff_ips=8,
                    staff_fls=4,
                ),
                _row(
                    2040,
                    2,
                    1,
                    total=100,
                    line=90,
                    ip=20,
                    fl=30,
                    wg_short=0.0,
                    staff_ips=5,
                    staff_fls=2,
                ),
            ]
        )

        metrics = compute_raw_metrics(history, assessment_start_year=2040)
        requirements = RequirementsConfig(
            target_total_pilots=None,
            target_line_pilots=None,
            min_experience_ratio=None,
            allowed_wg_rap_shortfall=None,
            allowed_fl_rap_shortfall=None,
            allowed_ip_rap_shortfall=None,
            allowed_utc_1_wg_shortfall=None,
            allowed_utc_1_fl_shortfall=None,
            allowed_utc_2_wg_shortfall=None,
            allowed_utc_2_fl_shortfall=None,
            target_staff_ips=10.0,
            target_staff_fls=3.0,
            allowed_unallocated_iron=None,
        )

        constraints = compute_constraints(metrics, requirements)

        self.assertEqual(metrics["final_staff_ips"], 5.0)
        self.assertEqual(metrics["min_staff_ips_after_assessment_start"], 5.0)
        self.assertEqual(constraints["staff_ips"], 5.0)
        self.assertEqual(constraints["staff_fls"], 1.0)

    def test_constraints_use_positive_violation_sign_convention(self):
        raw_metrics = {
            "final_total_pilots": 240.0,
            "min_total_pilots_after_assessment_start": 220.0,
            "min_line_pilots_after_assessment_start": 200.0,
            "final_staff_ips": 12.0,
            "final_staff_fls": 8.0,
            "min_staff_ips_after_assessment_start": 12.0,
            "min_staff_fls_after_assessment_start": 8.0,
            "min_experience_ratio_after_assessment_start": 0.52,
            "max_wg_rap_shortfall_after_assessment_start": 0.25,
            "max_fl_rap_shortfall_after_assessment_start": 0.0,
            "max_ip_rap_shortfall_after_assessment_start": 1.5,
        }
        requirements = RequirementsConfig(
            target_total_pilots=230.0,
            target_line_pilots=None,
            min_experience_ratio=None,
            allowed_wg_rap_shortfall=0.5,
            allowed_fl_rap_shortfall=0.0,
            allowed_ip_rap_shortfall=0.0,
            allowed_utc_1_wg_shortfall=None,
            allowed_utc_1_fl_shortfall=None,
            allowed_utc_2_wg_shortfall=None,
            allowed_utc_2_fl_shortfall=None,
            target_staff_ips=None,
            target_staff_fls=None,
            allowed_unallocated_iron=None,
        )

        constraints = compute_constraints(raw_metrics, requirements)
        scales = ConstraintScalesConfig(
            total_pilots=10.0,
            line_pilots=100.0,
            wg_rap=1.0,
            fl_rap=1.0,
            ip_rap=1.0,
            utc_1_wg=1.0,
            utc_1_fl=1.0,
            utc_2_wg=1.0,
            utc_2_fl=1.0,
            staff_ips=10.0,
            staff_fls=10.0,
            experience_ratio=0.05,
            unallocated_iron=50.0,
        )
        phi, active_constraint, active_value = aggregate_violation(constraints, scales)

        self.assertEqual(constraints["total_pilots_final"], -10.0)
        self.assertEqual(constraints["total_pilots_window"], 10.0)
        self.assertEqual(constraints["wg_rap"], -0.25)
        self.assertEqual(constraints["fl_rap"], 0.0)
        self.assertEqual(constraints["ip_rap"], 1.5)
        self.assertEqual(phi, 1.5)
        self.assertEqual(active_constraint, "ip_rap")
        self.assertEqual(active_value, 1.5)

    def test_utc_rap_constraints_use_assessment_window_maxima(self):
        history = pd.DataFrame(
            [
                _row(
                    2040,
                    1,
                    1,
                    total=100,
                    line=90,
                    ip=20,
                    fl=30,
                    wg_short=0.0,
                    utc_1_wg_short=0.5,
                    utc_1_fl_short=0.2,
                ),
                _row(
                    2040,
                    2,
                    1,
                    total=100,
                    line=90,
                    ip=20,
                    fl=30,
                    wg_short=0.0,
                    utc_1_wg_short=1.0,
                    utc_1_fl_short=0.4,
                ),
            ]
        )

        metrics = compute_raw_metrics(history, assessment_start_year=2040)
        requirements = RequirementsConfig(
            target_total_pilots=None,
            target_line_pilots=None,
            min_experience_ratio=None,
            allowed_wg_rap_shortfall=None,
            allowed_fl_rap_shortfall=None,
            allowed_ip_rap_shortfall=None,
            allowed_utc_1_wg_shortfall=0.25,
            allowed_utc_1_fl_shortfall=0.0,
            allowed_utc_2_wg_shortfall=None,
            allowed_utc_2_fl_shortfall=None,
            target_staff_ips=None,
            target_staff_fls=None,
            allowed_unallocated_iron=None,
        )

        constraints = compute_constraints(metrics, requirements)

        self.assertEqual(metrics["max_utc_1_wg_rap_shortfall_after_assessment_start"], 1.0)
        self.assertEqual(metrics["max_utc_1_fl_rap_shortfall_after_assessment_start"], 0.4)
        self.assertEqual(constraints["utc_1_wg"], 0.75)
        self.assertEqual(constraints["utc_1_fl"], 0.4)

    def test_unallocated_iron_constraint_uses_caf_phase_sum_maximum(self):
        history = pd.DataFrame(
            [
                _row(2040, 1, 1, total=100, line=90, ip=20, fl=30, wg_short=0.0, unallocated=10),
                _row(2040, 1, 2, total=100, line=90, ip=20, fl=30, wg_short=0.0, unallocated=5),
                _row(2040, 2, 1, total=100, line=90, ip=20, fl=30, wg_short=0.0, unallocated=40),
                _row(2040, 2, 2, total=100, line=90, ip=20, fl=30, wg_short=0.0, unallocated=0),
            ]
        )

        metrics = compute_raw_metrics(history, assessment_start_year=2040)
        requirements = RequirementsConfig(
            target_total_pilots=None,
            target_line_pilots=None,
            min_experience_ratio=None,
            allowed_wg_rap_shortfall=None,
            allowed_fl_rap_shortfall=None,
            allowed_ip_rap_shortfall=None,
            allowed_utc_1_wg_shortfall=None,
            allowed_utc_1_fl_shortfall=None,
            allowed_utc_2_wg_shortfall=None,
            allowed_utc_2_fl_shortfall=None,
            target_staff_ips=None,
            target_staff_fls=None,
            allowed_unallocated_iron=0.0,
        )

        constraints = compute_constraints(metrics, requirements)

        self.assertEqual(metrics["max_caf_unallocated_iron_after_assessment_start"], 40.0)
        self.assertEqual(constraints["unallocated_iron"], 40.0)


def _row(
    year,
    phase,
    squadron,
    total,
    line,
    ip,
    fl,
    wg_short,
    staff_ips=0,
    staff_fls=0,
    utc_1_wg_short=0.0,
    utc_1_fl_short=0.0,
    utc_2_wg_short=0.0,
    utc_2_fl_short=0.0,
    unallocated=0,
):
    return {
        "year": year,
        "phase": phase,
        "squadron_id": squadron,
        "total_pilots": total,
        "line_pilots": line,
        "staff_ips": staff_ips,
        "staff_fls": staff_fls,
        "ip_qty": ip,
        "fl_qty": fl,
        "wg_rap_shortfall": wg_short,
        "fl_rap_shortfall": 0.0,
        "ip_rap_shortfall": 0.0,
        "utc_1_wg_rap_shortfall": utc_1_wg_short,
        "utc_1_fl_rap_shortfall": utc_1_fl_short,
        "utc_2_wg_rap_shortfall": utc_2_wg_short,
        "utc_2_fl_rap_shortfall": utc_2_fl_short,
        "unallocated_iron": unallocated,
    }


if __name__ == "__main__":
    unittest.main()
