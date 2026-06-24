import unittest

from src.models import (
    AssignedUTCRank,
    Assignment,
    Pilot,
    Qual,
    SquadronConfig,
    Upgrade,
    monthly_sortie_rap_target,
)
from src.rap_state import UTC_RAP_SHORTFALL_COLUMNS, utc_sortie_rap_shortfall_columns


class UTCrapShortfallTest(unittest.TestCase):
    def test_empty_utc_cohorts_report_zero_shortfall(self):
        pilots = [
            Pilot(
                qual=Qual.WG,
                upgrade=Upgrade.NONE,
                active=True,
                current_assignment=Assignment.LINE,
                assigned_utc=AssignedUTCRank.UNASSIGNED,
                sortie_rap_monthly=5.0,
            )
        ]

        columns = utc_sortie_rap_shortfall_columns(pilots)

        self.assertEqual(set(columns), set(UTC_RAP_SHORTFALL_COLUMNS))
        self.assertTrue(all(value == 0.0 for value in columns.values()))

    def test_utc_cohort_shortfall_uses_mean_monthly_sortie_rap(self):
        wg_target = monthly_sortie_rap_target(Qual.WG)
        fl_target = monthly_sortie_rap_target(Qual.FL)
        pilots = [
            Pilot(
                qual=Qual.FL,
                upgrade=Upgrade.NONE,
                active=True,
                current_assignment=Assignment.LINE,
                assigned_utc=AssignedUTCRank.UTC_1,
                sortie_rap_monthly=6.0,
            ),
            Pilot(
                qual=Qual.FL,
                upgrade=Upgrade.NONE,
                active=True,
                current_assignment=Assignment.LINE,
                assigned_utc=AssignedUTCRank.UTC_1,
                sortie_rap_monthly=4.0,
            ),
            Pilot(
                qual=Qual.WG,
                upgrade=Upgrade.NONE,
                active=True,
                current_assignment=Assignment.LINE,
                assigned_utc=AssignedUTCRank.UTC_1,
                sortie_rap_monthly=wg_target - 1.0,
            ),
        ]

        columns = utc_sortie_rap_shortfall_columns(pilots)

        self.assertAlmostEqual(columns["utc_1_fl_rap_shortfall"], fl_target - 5.0)
        self.assertAlmostEqual(columns["utc_1_wg_rap_shortfall"], 1.0)

    def test_mqt_students_excluded_from_utc_wg_and_fl_cohorts(self):
        wg_target = monthly_sortie_rap_target(Qual.WG)
        fl_target = monthly_sortie_rap_target(Qual.FL)
        pilots = [
            Pilot(
                qual=Qual.WG,
                upgrade=Upgrade.MQT,
                active=True,
                current_assignment=Assignment.LINE,
                assigned_utc=AssignedUTCRank.UTC_1,
                sortie_rap_monthly=0.0,
            ),
            Pilot(
                qual=Qual.WG,
                upgrade=Upgrade.NONE,
                active=True,
                current_assignment=Assignment.LINE,
                assigned_utc=AssignedUTCRank.UTC_1,
                sortie_rap_monthly=wg_target,
            ),
            Pilot(
                qual=Qual.FL,
                upgrade=Upgrade.NONE,
                active=True,
                current_assignment=Assignment.LINE,
                assigned_utc=AssignedUTCRank.UTC_1,
                sortie_rap_monthly=fl_target,
            ),
        ]

        columns = utc_sortie_rap_shortfall_columns(pilots)

        self.assertAlmostEqual(columns["utc_1_wg_rap_shortfall"], 0.0)
        self.assertAlmostEqual(columns["utc_1_fl_rap_shortfall"], 0.0)

    def test_store_stats_from_physics_includes_utc_columns(self):
        cfg = SquadronConfig(
            paa=21,
            ute=10.0,
            experience_ratio=0.5,
            ip_qty=4,
            mqt_students=0,
            flug_students=0,
            ipug_students=0,
            total_pilots=30,
            id=1,
        )
        cfg.pilots = [
            Pilot(
                qual=Qual.FL,
                upgrade=Upgrade.NONE,
                active=True,
                current_assignment=Assignment.LINE,
                assigned_utc=AssignedUTCRank.UTC_1,
                sortie_rap_monthly=monthly_sortie_rap_target(Qual.FL),
            )
        ]
        row = cfg.store_stats_from_physics(2040, 1, 120.0)

        for column in UTC_RAP_SHORTFALL_COLUMNS:
            self.assertIn(column, row)

    def test_store_stats_from_physics_excludes_mqt_only_from_wg_rates(self):
        cfg = SquadronConfig(
            paa=21,
            ute=10.0,
            experience_ratio=0.5,
            ip_qty=2,
            id=1,
        )
        cfg.pilots = [
            Pilot(
                qual=Qual.WG,
                upgrade=Upgrade.MQT,
                active=True,
                current_assignment=Assignment.LINE,
                sortie_rap_monthly=0.0,
                sortie_blue_monthly=0.0,
            ),
            Pilot(
                qual=Qual.WG,
                upgrade=Upgrade.NONE,
                active=True,
                current_assignment=Assignment.LINE,
                sortie_rap_monthly=8.0,
                sortie_blue_monthly=4.0,
            ),
            Pilot(
                qual=Qual.WG,
                upgrade=Upgrade.FLUG,
                active=True,
                current_assignment=Assignment.LINE,
                sortie_rap_monthly=6.0,
                sortie_blue_monthly=3.0,
            ),
            Pilot(
                qual=Qual.FL,
                upgrade=Upgrade.IPUG,
                active=True,
                current_assignment=Assignment.LINE,
                sortie_rap_monthly=7.0,
                sortie_blue_monthly=2.0,
            ),
        ]
        row = cfg.store_stats_from_physics(2040, 1, 120.0)

        self.assertAlmostEqual(row["wg_rate_mo"], 7.0)
        self.assertAlmostEqual(row["wg_rate_blue"], 3.5)
        self.assertAlmostEqual(row["fl_rate_mo"], 7.0)
        self.assertAlmostEqual(row["fl_rate_blue"], 2.0)


if __name__ == "__main__":
    unittest.main()
