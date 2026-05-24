from src.models import EventType, SquadronConfig, Upgrade
from src.engine import create_pilots, run_phase_simulation, print_phase_summary
from src.single_phase_handoff_metrics import handoff_iteration_metrics

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

    pending = cfg.pending_deferred_requirements
    metrics = handoff_iteration_metrics(pending)
    print(
        f"Upgrade carryover — incomplete students: "
        f"MQT={int(metrics['incomplete_mqt_students'])}, "
        f"FLUG={int(metrics['incomplete_flug_students'])}, "
        f"IPUG={int(metrics['incomplete_ipug_students'])}; "
        f"syllabus lines: {len(pending)}"
    )
    for upgrade in (Upgrade.MQT, Upgrade.FLUG, Upgrade.IPUG):
        by_pilot: dict[int, dict[str, int]] = {}
        for item in pending:
            if item.upgrade != upgrade:
                continue
            row = by_pilot.setdefault(item.student_pilot_id, {"sorties": 0, "sims": 0})
            if item.event_type == EventType.SIM:
                row["sims"] += 1
            else:
                row["sorties"] += 1
        if by_pilot:
            print(f"  {upgrade.value}: {len(by_pilot)} student(s)")
            for pilot_id, counts in sorted(by_pilot.items()):
                print(
                    f"    pilot_id={pilot_id}: "
                    f"{counts['sorties']} sorties, {counts['sims']} sims remaining"
                )

    print_phase_summary(pilots, cfg, verbose=False)
