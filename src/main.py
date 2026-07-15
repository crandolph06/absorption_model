from src.engine import enroll_upgrade_students, run_phase_simulation, print_phase_summary
from src.manning_config import get_initial_squadrons, TEST_SQUADRON_DATA
from src.simulation_config import SimulationConfig

SIM_CONFIG = SimulationConfig(phase_length_days=30)

if __name__ == "__main__":
    squadrons = get_initial_squadrons(2026, TEST_SQUADRON_DATA)

    for squadron in squadrons:
        # Initial roster may already have upgrade tags from squadron seeds.
        run_phase_simulation(
            squadron,
            squadron.pilots,
            sim_config=SIM_CONFIG,
            debug_verbose=True,
        )
        print_phase_summary(squadron.pilots, squadron, verbose=False)

    # Optional: second phase on same roster — carryover retries incomplete_syllabus_items.
    # New quota enrollment (if desired) must be explicit via enroll_upgrade_students.
    RUN_SECOND_PHASE = False
    if RUN_SECOND_PHASE:
        print("\n--- Phase 2 (carryover) ---")
        squadron = squadrons[0]
        enroll_upgrade_students(squadron, squadron.pilots)
        run_phase_simulation(squadron, squadron.pilots, sim_config=SIM_CONFIG)
        print_phase_summary(squadron.pilots, squadron, verbose=False)
