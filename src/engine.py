"""Single-phase squadron training simulation (orchestration entry point)."""

from typing import List, Set

from src.models import (
    SquadronConfig,
    Pilot,
    Qual,
    Upgrade,
    DeferredSyllabusItem,
    DeferralSignature,
)
from src.syllabi import (
    CONTINUATION_PROFILE,
    FLUG_SYLLABUS,
    IPUG_SYLLABUS,
    MQT_SYLLABUS,
    ContinuationProfile,
)
from src import rules
from src.engine_handoff import finalize_pending_carryover
from src.engine_syllabus import (
    assign_sortie,
    process_pending_deferred_requirements,
    run_upgrade_program,
)
from src.engine_sim import (
    allocate_ep_sim,
    allocate_extra_rap_sims,
    initial_sim_session_budget,
)


def create_pilots(cfg: SquadronConfig) -> List[Pilot]:
    """Generates the initial list of pilots based on configuration."""
    experienced = int(cfg.total_pilots * cfg.experience_ratio)

    if experienced > cfg.total_pilots:
        raise ValueError("Experienced pilots cannot exceed total pilots")

    ip_count = cfg.ip_qty
    if ip_count > experienced:
        raise ValueError("IP quantity cannot exceed experienced pilots")

    fl_count = experienced - ip_count
    wg_count = cfg.total_pilots - experienced

    if cfg.mqt_students + cfg.flug_students > wg_count:
        raise ValueError("WG upgrade quantity cannot exceed WG quantity")

    pilots: List[Pilot] = []
    for _ in range(wg_count):
        pilots.append(Pilot(Qual.WG, squadron_id=cfg.id))
    for _ in range(fl_count):
        pilots.append(Pilot(Qual.FL, squadron_id=cfg.id))
    for _ in range(ip_count):
        pilots.append(Pilot(Qual.IP, squadron_id=cfg.id))
    return pilots


def total_phase_capacity(cfg: SquadronConfig) -> float:
    return cfg.ute * cfg.paa


def select_upgrade_students(pilots: List[Pilot], upgrade_type: Upgrade, count: int) -> List[Pilot]:
    """
    Identifies eligible pilots and marks them for upgrade.
    """
    candidates = [p for p in pilots if rules.can_start_upgrade(p, upgrade_type)]
    selected = candidates[:count]

    for p in selected:
        if p.upgrade != upgrade_type:
            p.upgrade_syllabus_done.clear()
        p.upgrade = upgrade_type
        p.sorties_at_upgrade_start = p.sorties_flown
        p.sims_at_upgrade_start = p.sims_flown

    return selected


def allocate_continuation_training(
    cfg: SquadronConfig,
    pilots: List[Pilot],
    profile: ContinuationProfile,
    total_capacity: int,
    noise: float,
):
    """
    Spend remaining nominal wing sorties (``total_capacity`` minus syllabus usage) on CT.
    """
    used_sorties = sum(p.sortie_phase for p in pilots)
    remaining_capacity = max(0, total_capacity - used_sorties)

    if remaining_capacity <= 0:
        return

    ct_candidates = [p for p in pilots if p.upgrade != Upgrade.MQT]
    if not ct_candidates:
        return

    raw_qty = [(b, remaining_capacity * b.fraction) for b in profile.buckets]
    base_qty = {b: int(x) for b, x in raw_qty}

    leftover = remaining_capacity - sum(base_qty.values())
    sorted_remainders = sorted(raw_qty, key=lambda x: x[1] - int(x[1]), reverse=True)

    for i in range(leftover):
        bucket = sorted_remainders[i % len(sorted_remainders)][0]
        base_qty[bucket] += 1

    for bucket, qty in base_qty.items():
        eligible = [p for p in ct_candidates if rules.can_fill_seat(p, bucket.min_qual, None)]
        for _ in range(qty):
            assign_sortie(eligible, bucket.side, noise, cfg.avg_sortie_dur)


def run_phase_simulation(cfg: SquadronConfig, pilots: List[Pilot], allocation_noise: float = 0.0):
    """
    Run one training phase.

    Simulator **session** budget for the phase is ``sim_sessions_monthly × phase_months``.
    Allocation order: pending deferrals → upgrade syllabi → continuation sorties →
    EP sim → extra sim RAP toward 3 / month total.
    """
    for p in pilots:
        p.reset_phase_counters()

    phase_months = cfg.phase_length_months
    sim_session_budget = [initial_sim_session_budget(cfg)]

    deferred_requirements: List[DeferredSyllabusItem] = []
    completed_keys: Set[DeferralSignature] = set()
    process_pending_deferred_requirements(
        cfg,
        pilots,
        allocation_noise,
        deferred_requirements,
        completed_keys,
        sim_session_budget,
    )

    select_upgrade_students(pilots, Upgrade.MQT, cfg.mqt_students)
    select_upgrade_students(pilots, Upgrade.FLUG, cfg.flug_students)
    select_upgrade_students(pilots, Upgrade.IPUG, cfg.ipug_students)

    mqt_students = [p for p in pilots if p.upgrade == Upgrade.MQT]
    flug_students = [p for p in pilots if p.upgrade == Upgrade.FLUG]
    ipug_students = [p for p in pilots if p.upgrade == Upgrade.IPUG]

    run_upgrade_program(
        cfg,
        MQT_SYLLABUS,
        mqt_students,
        pilots,
        Upgrade.MQT,
        allocation_noise,
        deferred_requirements,
        completed_keys,
        sim_session_budget,
    )
    run_upgrade_program(
        cfg,
        FLUG_SYLLABUS,
        flug_students,
        pilots,
        Upgrade.FLUG,
        allocation_noise,
        deferred_requirements,
        completed_keys,
        sim_session_budget,
    )
    run_upgrade_program(
        cfg,
        IPUG_SYLLABUS,
        ipug_students,
        pilots,
        Upgrade.IPUG,
        allocation_noise,
        deferred_requirements,
        completed_keys,
        sim_session_budget,
    )

    total_capacity = max(0, round(total_phase_capacity(cfg) * phase_months))
    allocate_continuation_training(cfg, pilots, CONTINUATION_PROFILE, total_capacity, allocation_noise)

    allocate_ep_sim(cfg, pilots, allocation_noise, sim_session_budget)
    allocate_extra_rap_sims(cfg, pilots, allocation_noise, sim_session_budget)

    for p in pilots:
        p.set_rap_requirement()
        p.update_total(cfg.phase_length_days)
        p.update_monthly(cfg.phase_length_days)

    finalize_pending_carryover(cfg, mqt_students, flug_students, ipug_students)

    return pilots


def print_phase_summary(pilots: List[Pilot], cfg: SquadronConfig, verbose: bool = True):
    print(f"\n=== Phase Summary ({cfg.phase_length_days} d ≈ {cfg.phase_length_months:.2f} mo) ===")

    groups = {
        "MQT Students": [p for p in pilots if p.upgrade == Upgrade.MQT],
        "FLUG Students": [p for p in pilots if p.upgrade == Upgrade.FLUG],
        "IPUG Students": [p for p in pilots if p.upgrade == Upgrade.IPUG],
        "Line Wingmen": [p for p in pilots if p.qual == Qual.WG and p.upgrade == Upgrade.NONE],
        "Line FLs": [p for p in pilots if p.qual == Qual.FL and p.upgrade == Upgrade.NONE],
        "IPs": [p for p in pilots if p.qual == Qual.IP],
    }

    for name, group in groups.items():
        if not group:
            print(f"{name}: None")
            continue

        avg_sorties = sum(p.sortie_monthly for p in group) / len(group)
        avg_blue_sorties = sum(p.sortie_blue_monthly for p in group) / len(group)
        avg_red_sorties = sum(p.sortie_red_monthly for p in group) / len(group)
        avg_sims = sum(p.sim_monthly for p in group) / len(group)
        avg_sim_sf = sum(p.sim_rap_shortfall for p in group) / len(group)
        print(
            f"{name} ({len(group)}): Avg Mo. Sorties {avg_sorties:.1f}, Blue {avg_blue_sorties:.1f}, "
            f"Red {avg_red_sorties:.1f}; Avg Mo. Sims {avg_sims:.1f}, Avg sim RAP shortfall {avg_sim_sf:.2f}"
        )

    if verbose:
        for name, group in groups.items():
            for p in group:
                print(
                    f"{p.qual}/{p.upgrade}: Sorties/mo {p.sortie_monthly:.2f}, "
                    f"Blue {p.sortie_blue_monthly:.2f}, Red {p.sortie_red_monthly:.2f}; "
                    f"Sims/mo {p.sim_monthly:.2f}; sortie RAP shortfall {p.rap_shortfall:.2f}, "
                    f"sim RAP shortfall {p.sim_rap_shortfall:.2f}"
                )
