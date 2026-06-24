import random
import unittest

from src.manning_config import get_initial_squadrons, TEST_SQUADRON_DATA
from src.manning_engine import CAFSimulation
from src.models import Pilot, Qual, Upgrade, Assignment
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


if __name__ == "__main__":
    unittest.main()
