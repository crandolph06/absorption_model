from numpy import true_divide
from src.models import SquadronConfig
from src.engine import create_pilots, run_phase_simulation, print_phase_summary
from src.manning_config import get_initial_squadrons, TEST_SQUADRON_DATA

if __name__ == "__main__":
    squadrons = get_initial_squadrons(2026, TEST_SQUADRON_DATA)

    for squadron in squadrons:
        run_phase_simulation(squadron, squadron.pilots, debug_verbose=True, pre_seed_upgrades=True)
        print_phase_summary(squadron.pilots, squadron, verbose=True)

    # Optional: second phase on same roster — carryover retries incomplete_syllabus_items
    RUN_SECOND_PHASE = False
    if RUN_SECOND_PHASE:
        print("\n--- Phase 2 (carryover) ---")
        run_phase_simulation(squadron, squadron.pilots, pre_seed_upgrades=False)
        print_phase_summary(squadron.pilots, squadron, verbose=False)