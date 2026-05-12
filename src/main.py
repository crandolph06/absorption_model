from src.models import SquadronConfig
from src.engine import create_pilots, run_phase_simulation, print_phase_summary

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

    pilots = create_pilots(cfg)

    run_phase_simulation(cfg, pilots)

    h = cfg.last_phase_upgrade_handoff
    if h is not None:
        print(
            f"Upgrade handoff — MQT complete: {h.mqt_syllabus_complete}, "
            f"FLUG complete: {h.flug_syllabus_complete}, IPUG complete: {h.ipug_syllabus_complete}, "
            f"deferred syllabus lines: {len(h.deferred_requirements)}"
        )

    print_phase_summary(pilots, cfg, verbose=False)