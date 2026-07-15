import random
import unittest

from src.manning_config import get_initial_squadrons, TEST_SQUADRON_DATA
from src.manning_engine import CAFSimulation
from src.models import Pilot, Qual, Upgrade, Assignment, SquadronConfig
from src.simulation_config import SimulationConfig


class ManningPhysicsBackendTest(unittest.TestCase):
    def test_decrement_adsc_for_squadron(self):
        sim = CAFSimulation(
            annual_intake=0,
            retention_rate=0.5,
            round_robin=True,
            use_physics_allocator=True,
            sim_config=SimulationConfig(phase_length_days=120),
        )
        squadrons = get_initial_squadrons(2026, TEST_SQUADRON_DATA)
        sq = squadrons[0]
        pilot = Pilot(
            qual=Qual.WG,
            upgrade=Upgrade.NONE,
            year_group=2026,
            adsc_remaining=8,
            active=True,
            squadron_id=sq.id,
            current_assignment=Assignment.LINE,
        )
        sq.pilots = [pilot]

        sim._decrement_adsc_for_squadron(sq)
        self.assertEqual(pilot.adsc_remaining, 4.0)

        sim._decrement_adsc_for_squadron(sq)
        self.assertEqual(pilot.adsc_remaining, 0.0)

    def test_physics_path_applies_retention_after_adsc_countdown(self):
        random.seed(0)
        sim = CAFSimulation(
            annual_intake=3,
            retention_rate=0.0,
            round_robin=True,
            use_physics_allocator=True,
            use_upgrade_quotas=False,
            sim_config=SimulationConfig(phase_length_days=120),
        )
        squadrons = get_initial_squadrons(2026, TEST_SQUADRON_DATA)
        history = sim.run_simulation(years_to_run=12, squadron_configs=squadrons, ute=3.0)

        later_intake_separations = history.loc[history["year"] >= 2033, "separated"].sum()
        self.assertGreater(later_intake_separations, 0, "intake pilots should separate once ADSC reaches zero")
        self.assertGreater(history["separated"].sum(), 0, "some separations should occur over the horizon")

    def test_brain_path_still_decrements_adsc_via_apply_phase_aging(self):
        from src.models import AgingRate, SquadronConfig

        sim = CAFSimulation(
            annual_intake=0,
            retention_rate=0.5,
            round_robin=True,
            use_physics_allocator=False,
            brain=object(),
        )
        sim._brain_n_outputs = lambda: 16
        sim.predict_rates_fast = lambda: [[0.0] * 16]

        sq = SquadronConfig(id=1, paa=10, ute=10.0, ip_qty=2, pilots=[], experience_ratio=0.5)
        pilot = Pilot(
            qual=Qual.WG,
            upgrade=Upgrade.NONE,
            year_group=2026,
            adsc_remaining=8,
            active=True,
            squadron_id=1,
            current_assignment=Assignment.LINE,
        )
        sq.pilots = [pilot]
        sim.squadrons = [sq]

        rates = AgingRate(wg_phase=0.0, wg_blue_phase=0.0)
        sq.apply_phase_aging(rates, sim.sim_config.phase_length_days)
        self.assertEqual(pilot.adsc_remaining, 4.0)

    def test_physics_path_uses_layer2_flug_enrollment(self):
        """Window-selected FLUG must enter Layer-1 syllabus, not free-graduate unused."""
        from src.engine import graduate_completed_upgrades, run_phase_simulation

        pilots = []
        for _ in range(3):
            pilots.append(
                Pilot(
                    qual=Qual.IP,
                    upgrade=Upgrade.NONE,
                    sorties_flown=500,
                    current_assignment=Assignment.LINE,
                    active=True,
                    squadron_id=1,
                )
            )
        for _ in range(4):
            pilots.append(
                Pilot(
                    qual=Qual.FL,
                    upgrade=Upgrade.NONE,
                    sorties_flown=350,
                    current_assignment=Assignment.LINE,
                    active=True,
                    squadron_id=1,
                )
            )
        high = Pilot(
            qual=Qual.WG,
            upgrade=Upgrade.NONE,
            sorties_flown=280,
            current_assignment=Assignment.LINE,
            active=True,
            squadron_id=1,
        )
        low = Pilot(
            qual=Qual.WG,
            upgrade=Upgrade.NONE,
            sorties_flown=60,
            current_assignment=Assignment.LINE,
            active=True,
            squadron_id=1,
        )
        pilots.extend([high, low])
        sq = SquadronConfig(
            id=1,
            paa=12,
            ute=10.0,
            pilots=pilots,
            experience_ratio=0.7,
        )
        sq.new_phase_upgrades(
            flug_window_start=250,
            ipug_window_start=400,
            use_upgrade_quotas=True,
            flug_quota=1,
            ipug_quota=0,
        )
        sq.update_stats()
        self.assertEqual(high.upgrade, Upgrade.FLUG)
        self.assertEqual(low.upgrade, Upgrade.NONE)

        sim = CAFSimulation(
            annual_intake=0,
            retention_rate=1.0,
            round_robin=True,
            use_physics_allocator=True,
            use_upgrade_quotas=True,
            sim_config=SimulationConfig(phase_length_days=30, allocation_noise=0.0),
        )
        run_phase_simulation(
            sq,
            sq.pilots,
            auto_graduate=False,
            sim_config=sim.sim_config,
        )
        self.assertGreater(
            len(high.incomplete_syllabus_items),
            0,
            "Layer-2 FLUG enrollee must receive a syllabus",
        )
        self.assertEqual(low.upgrade, Upgrade.NONE)

        graduate_completed_upgrades(sq.pilots)
        self.assertEqual(high.qual, Qual.WG)
        self.assertEqual(high.upgrade, Upgrade.FLUG)

    def test_add_sortie_increments_lifetime_sorties_flown(self):
        """Physics allocator must advance sorties_flown (FLUG window is sortie-gated)."""
        pilot = Pilot(
            qual=Qual.WG,
            upgrade=Upgrade.NONE,
            sorties_flown=240,
            current_assignment=Assignment.LINE,
            active=True,
        )
        pilot.add_sortie(1.3, side="Blue")
        self.assertEqual(pilot.sorties_flown, 241)
        self.assertEqual(pilot.sortie_phase, 1)

    def test_physics_flug_enrollment_continues_after_phase_one(self):
        """WGs just below the FLUG window must become eligible once they fly CT."""
        from src.engine import run_phase_simulation
        from src.models import SquadronConfig

        pilots = []
        for _ in range(3):
            pilots.append(
                Pilot(
                    qual=Qual.IP,
                    upgrade=Upgrade.NONE,
                    sorties_flown=500,
                    flight_hours_flown=600,
                    current_assignment=Assignment.LINE,
                    active=True,
                    squadron_id=1,
                )
            )
        for _ in range(4):
            pilots.append(
                Pilot(
                    qual=Qual.FL,
                    upgrade=Upgrade.NONE,
                    sorties_flown=300,
                    flight_hours_flown=250,
                    current_assignment=Assignment.LINE,
                    active=True,
                    squadron_id=1,
                )
            )
        near = Pilot(
            qual=Qual.WG,
            upgrade=Upgrade.NONE,
            sorties_flown=249,
            flight_hours_flown=200,
            current_assignment=Assignment.LINE,
            active=True,
            squadron_id=1,
        )
        pilots.append(near)
        sq = SquadronConfig(id=1, paa=18, ute=20.0, pilots=pilots, experience_ratio=0.7)
        sim_cfg = SimulationConfig(phase_length_days=30, allocation_noise=0.0)

        # Phase with no FLUG-eligible pilots yet.
        sq.new_phase_upgrades(
            flug_window_start=250,
            ipug_window_start=9999,
            use_upgrade_quotas=True,
            flug_quota=1,
            ipug_quota=0,
        )
        self.assertEqual(near.upgrade, Upgrade.NONE)
        run_phase_simulation(sq, sq.pilots, sim_config=sim_cfg, auto_graduate=False)
        self.assertGreaterEqual(near.sorties_flown, 250)

        # Next enrollment should pick the newly eligible WG.
        sq.new_phase_upgrades(
            flug_window_start=250,
            ipug_window_start=9999,
            use_upgrade_quotas=True,
            flug_quota=1,
            ipug_quota=0,
        )
        self.assertEqual(near.upgrade, Upgrade.FLUG)


if __name__ == "__main__":
    unittest.main()
