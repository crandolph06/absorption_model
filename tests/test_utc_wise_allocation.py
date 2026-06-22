"""UTC-ordered RAP / support allocation when ``utc_wise_allocation`` is enabled."""
from __future__ import annotations

import unittest

from src.engine import (
    assign_sortie,
    assign_sortie_policy,
    create_pilots,
    run_phase_simulation,
)
from src.models import (
    AssignedUTCRank,
    PHASE_DAYS_PER_NOTIONAL_MONTH,
    Pilot,
    Qual,
    SquadronConfig,
    Upgrade,
)
from src.simulation_config import SimulationConfig

# Tests use legacy iron split unless explicitly exercising upgrade_sortie_fraction.
_LEGACY_SIM_KWARGS = {"upgrade_sortie_fraction": None}


def _make_cfg_and_pilots() -> tuple[SquadronConfig, list[Pilot]]:
    total = 20
    ip = 4
    fl = 6
    cfg = SquadronConfig(
        ute=10.0,
        paa=12,
        id=1,
        total_pilots=total,
        ip_qty=ip,
        experience_ratio=(fl + ip) / total,
    )
    pilots = create_pilots(cfg)
    return cfg, pilots


class UTCWiseAllocationTest(unittest.TestCase):
    def test_assign_sortie_prefers_lower_utc_when_flag_on(self):
        cfg, pilots = _make_cfg_and_pilots()
        cfg.pilots = pilots
        wg_pilots = [p for p in pilots if p.qual == Qual.WG and p.upgrade == Upgrade.NONE]
        self.assertGreaterEqual(len(wg_pilots), 2)
        utc1, utc3 = wg_pilots[0], wg_pilots[1]
        utc1.assigned_utc = AssignedUTCRank.UTC_1
        utc3.assigned_utc = AssignedUTCRank.UTC_3
        phase_days = 120.0
        for pilot in (utc1, utc3):
            pilot.set_rap_requirement()

        ok = assign_sortie(
            cfg,
            [utc1, utc3],
            phase_days,
            utc_wise_allocation=True,
        )

        self.assertTrue(ok)
        self.assertEqual(utc1.sortie_phase, 1)
        self.assertEqual(utc3.sortie_phase, 0)

    def test_rap_priority_skips_pilots_who_made_sortie_rap(self):
        cfg, pilots = _make_cfg_and_pilots()
        fls = [p for p in pilots if p.qual == Qual.FL and p.upgrade == Upgrade.NONE]
        utc1, utc2 = fls[0], fls[1]
        utc1.assigned_utc = AssignedUTCRank.UTC_1
        utc2.assigned_utc = AssignedUTCRank.UTC_2
        phase_days = 120.0
        for pilot in (utc1, utc2):
            pilot.set_rap_requirement()

        months = phase_days / PHASE_DAYS_PER_NOTIONAL_MONTH
        rap_sorties = utc1.target_sorties * months
        utc1.sortie_phase = rap_sorties

        ok = assign_sortie_policy(
            cfg, [utc1, utc2], phase_days, mode="rap_priority", utc_wise=True,
        )
        self.assertTrue(ok)
        self.assertEqual(utc1.sortie_phase, rap_sorties)
        self.assertEqual(utc2.sortie_phase, 1)

    def test_scarce_iron_rap_priority_does_not_overfly_before_equity(self):
        """With tight iron, UTC-wise RAP priority should not exceed RAP targets."""
        cfg, pilots = _make_cfg_and_pilots()
        cfg.mqt_students = 0
        cfg.flug_students = 0
        cfg.ipug_students = 0
        cfg.ute = 2.0
        cfg.paa = 3
        for index, pilot in enumerate(pilots):
            pilot.sorties_flown = 1000 - index

        run_phase_simulation(
            cfg,
            pilots,
            sim_config=SimulationConfig(
                utc_wise_allocation=True,
                phase_length_days=30,
                **_LEGACY_SIM_KWARGS,
            ),
            auto_graduate=False,
        )

        phase_days = 30.0
        months = phase_days / PHASE_DAYS_PER_NOTIONAL_MONTH
        for pilot in pilots:
            if pilot.upgrade != Upgrade.NONE:
                continue
            if pilot.target_sorties <= 0:
                continue
            expected = pilot.target_sorties * months
            credited = pilot.sortie_rap_credit(months)
            self.assertLessEqual(
                credited,
                expected + 1e-6,
                msg=f"{pilot.qual.name} credited {credited} > RAP {expected}",
            )

    def test_abundant_iron_equity_pass_can_exceed_rap(self):
        """With excess iron, leftover equity pass may distribute sorties above RAP."""
        total = 10
        ip = 2
        fl = 3
        wg = 5
        cfg = SquadronConfig(
            ute=20.0,
            paa=5,
            id=1,
            total_pilots=total,
            ip_qty=ip,
            experience_ratio=(fl + ip) / total,
            mqt_students=0,
            flug_students=0,
            ipug_students=0,
        )
        pilots = create_pilots(cfg)
        for index, pilot in enumerate(pilots):
            pilot.sorties_flown = 100 - index

        run_phase_simulation(
            cfg,
            pilots,
            sim_config=SimulationConfig(
                utc_wise_allocation=True,
                phase_length_days=120,
                **_LEGACY_SIM_KWARGS,
            ),
            auto_graduate=False,
        )

        phase_days = 120.0
        months = phase_days / PHASE_DAYS_PER_NOTIONAL_MONTH
        wg_line = [
            p for p in pilots
            if p.qual == Qual.WG and p.upgrade == Upgrade.NONE
        ]
        self.assertGreater(len(wg_line), 0)
        rap_target = wg_line[0].target_sorties * months
        max_credited = max(p.sortie_rap_credit(months) for p in wg_line)
        self.assertGreater(
            max_credited,
            rap_target + 1e-6,
            msg="Expected equity pass to push at least one WG above RAP with abundant iron",
        )
        sortie_counts = [p.sortie_phase for p in wg_line]
        self.assertLess(
            max(sortie_counts) - min(sortie_counts),
            5.0,
            msg="Leftover sorties should be spread roughly evenly across wingmen",
        )

    def test_update_rap_scenarios_assigns_enum_ranks(self):
        cfg, pilots = _make_cfg_and_pilots()
        cfg.pilots = pilots
        fls = [p for p in pilots if p.qual == Qual.FL]
        for index, pilot in enumerate(fls):
            pilot.sorties_flown = 1000 - index

        cfg.update_rap_scenarios()

        ranked = [p for p in pilots if p.assigned_utc == AssignedUTCRank.UTC_1]
        self.assertGreater(len(ranked), 0)
        self.assertTrue(all(isinstance(p.assigned_utc, AssignedUTCRank) for p in pilots if p.active))

    def test_ips_can_fill_fl_utc_slots(self):
        cfg, pilots = _make_cfg_and_pilots()
        cfg.pilots = pilots
        for pilot in pilots:
            if pilot.qual == Qual.IP:
                pilot.sorties_flown = 2000
            elif pilot.qual == Qual.FL:
                pilot.sorties_flown = 100

        cfg.update_rap_scenarios()

        utc1_ips = [
            p for p in pilots
            if p.qual == Qual.IP and p.assigned_utc == AssignedUTCRank.UTC_1
        ]
        self.assertGreater(len(utc1_ips), 0)

    def test_utc_wise_phase_run_assigns_utc_ranks(self):
        cfg, pilots = _make_cfg_and_pilots()
        fls = [p for p in pilots if p.qual == Qual.FL and p.upgrade == Upgrade.NONE]
        for index, pilot in enumerate(fls):
            pilot.sorties_flown = 500 - index

        run_phase_simulation(
            cfg,
            pilots,
            sim_config=SimulationConfig(utc_wise_allocation=True, **_LEGACY_SIM_KWARGS),
            auto_graduate=False,
        )

        utc1 = [
            p for p in pilots
            if p.assigned_utc == AssignedUTCRank.UTC_1 and p.upgrade == Upgrade.NONE
        ]
        self.assertGreater(len(utc1), 0)
        self.assertGreater(sum(p.sortie_phase + p.sim_phase for p in pilots), 0)

    def test_utc_wise_ct_prefers_utc1_under_scarce_iron(self):
        """With limited CT iron, UTC 1 line pilots should fly more than UTC 3."""
        total = 40
        ip = 4
        fl = 16
        wg = 20
        cfg = SquadronConfig(
            ute=2.0,
            paa=3,
            id=1,
            total_pilots=total,
            ip_qty=ip,
            experience_ratio=(fl + ip) / total,
        )
        pilots = create_pilots(cfg)
        for index, pilot in enumerate(pilots):
            pilot.sorties_flown = 1000 - index

        run_phase_simulation(
            cfg,
            pilots,
            sim_config=SimulationConfig(
                utc_wise_allocation=True,
                phase_length_days=30,
                **_LEGACY_SIM_KWARGS,
            ),
            auto_graduate=False,
        )

        utc1 = [
            p for p in pilots
            if p.assigned_utc == AssignedUTCRank.UTC_1 and p.upgrade == Upgrade.NONE
        ]
        utc3 = [
            p for p in pilots
            if p.assigned_utc == AssignedUTCRank.UTC_3 and p.upgrade == Upgrade.NONE
        ]
        self.assertGreater(len(utc1), 0)
        self.assertGreater(len(utc3), 0)

        avg_utc1 = sum(p.sortie_phase for p in utc1) / len(utc1)
        avg_utc3 = sum(p.sortie_phase for p in utc3) / len(utc3)
        self.assertGreater(avg_utc1, avg_utc3)


if __name__ == "__main__":
    unittest.main()
