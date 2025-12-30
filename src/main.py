from models import SquadronConfig
from engine import run_phase_simulation, print_phase_summary # You'll need to move print_summary to engine

if __name__ == "__main__":
    cfg = SquadronConfig(
        ute=5,
        paa=4,
        mqt_students=2,
        flug_students=1,
        ipug_students=1,
        total_pilots=10,
        experience_ratio=0.5,
        ip_qty=2,
        phase_length_days=30
    )

    # Create pilots (you might want to move pilot creation out of run_phase 
    # if you want to persist them across multiple phases later)
    from engine import create_pilots
    pilots = create_pilots(cfg)

    # Run
    run_phase_simulation(cfg, pilots)
    
    # Report
    print_phase_summary(pilots, cfg)