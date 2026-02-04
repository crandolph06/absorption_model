from src.models import SquadronConfig
from src.engine import run_phase_simulation, print_phase_summary # You'll need to move print_summary to engine

if __name__ == "__main__":
    cfg = SquadronConfig(
        ute=10,
        paa=18,
        id=99,
        mqt_students=5,
        flug_students=3,
        ipug_students=3,
        total_pilots=30,
        experience_ratio=0.5,
        ip_qty=4,
        phase_length_days=120,
        avg_sortie_dur = 1.3
    )

    from engine import create_pilots
    pilots = create_pilots(cfg)

    run_phase_simulation(cfg, pilots)
    
    print_phase_summary(pilots, cfg, verbose=False)