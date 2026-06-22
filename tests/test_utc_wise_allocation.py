"""UTC-ordered RAP / support allocation when ``utc_wise_allocation`` is enabled."""
from __future__ import annotations

import unittest

from src.engine import (
    assign_sortie,
    create_pilots,
    run_phase_simulation,
)
from src.models import (
    AssignedUTCRank,
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

        ok = assign_sortie(
            cfg,
            [utc1, utc3],
            phase_days,
            utc_wise_allocation=True,
        )

        self.assertTrue(ok)
        self.assertEqual(utc1.sortie_phase, 1)
        self.assertEqual(utc3.sortie_phase, 0)

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
