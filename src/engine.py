import random
from typing import List, Optional
from src.models import (
    SquadronConfig,
    Pilot,
    Qual,
    Upgrade,
    EventType,
    DeferredSyllabusItem,
    PhaseUpgradeHandoff,
)
from src.syllabi import SyllabusEvent, ContinuationProfile, MQT_SYLLABUS, FLUG_SYLLABUS, IPUG_SYLLABUS, CONTINUATION_PROFILE
from src import rules


# ----------------------
# Pilot Creation
# ----------------------
def create_pilots(cfg: SquadronConfig) -> List[Pilot]:
    """
    Generates the initial list of pilots based on configuration.
    """
    experienced = int(cfg.total_pilots * cfg.experience_ratio)

    if experienced > cfg.total_pilots:
        raise ValueError("Experienced pilots cannot exceed total pilots")
    
    ip_count = cfg.ip_qty
    if ip_count > experienced:
        raise ValueError("IP quantity cannot exceed experienced pilots")
    
    fl_count = experienced - ip_count
    wg_count = cfg.total_pilots - experienced

    if cfg.mqt_students + cfg.flug_students > wg_count:
        raise ValueError ("WG upgrade quantity cannot exceed WG quantity")
    
    return ([Pilot(Qual.WG) for _ in range(wg_count)] +
            [Pilot(Qual.FL) for _ in range(fl_count)] +
            [Pilot(Qual.IP) for _ in range(ip_count)])

def _syllabus_track_complete(
    students: List[Pilot], upgrade: Upgrade, deferred: List[DeferredSyllabusItem]
) -> bool:
    """True if no students in this track or no deferred lines for this upgrade."""
    if not students:
        return True
    return not any(d.upgrade == upgrade for d in deferred)


def total_phase_capacity(cfg: SquadronConfig) -> float:
    return cfg.ute * cfg.paa

# ----------------------
# Selection Phase
# ----------------------
def select_upgrade_students(pilots: List[Pilot], upgrade_type: Upgrade, count: int) -> List[Pilot]:
    """
    Identifies eligible pilots and marks them for upgrade.
    """
    candidates = [p for p in pilots if rules.can_start_upgrade(p, upgrade_type)]
    
    # Simple selection: take the first available 
    # (Future improvement: Sort by experience/seniority)
    selected = candidates[:count]
    
    for p in selected:
        p.upgrade = upgrade_type
        
    return selected

# ----------------------
# Allocation Helpers
# ----------------------
def assign_sortie(candidates: List[Pilot], side: str = "Blue", noise: float = 0.0, avg_sortie_dur: float = 1.3) -> bool:
    """
    Selects a candidate to fly a sortie by lowest effective phase utilization.

    ``noise`` (>= 0): each pilot's ``sortie_phase`` is perturbed by
    ``uniform(-noise, noise)`` before sorting, so the lowest-load pilot is not
    always picked—representing scheduling friction and ops variance. When
    ``noise`` is 0, allocation is deterministic (minimum ``sortie_phase`` wins).
    """
    if not candidates:
        return False

    n = max(0.0, float(noise))

    def _effective_util(p: Pilot) -> float:
        jitter = random.uniform(-n, n) if n > 0 else 0.0
        return p.sortie_phase + jitter

    candidates.sort(key=_effective_util)

    winner = candidates[0]

    winner.add_sortie(avg_sortie_dur, side)

    return True

# ----------------------
# Syllabus Execution
# ----------------------
def _ips_for_syllabus(all_pilots: List[Pilot], syllabus_upgrade_type: Upgrade) -> List[Pilot]:
    return [p for p in all_pilots if rules.can_fill_seat(p, Qual.IP, syllabus_upgrade_type)]


def _wg_blue_candidates(all_pilots: List[Pilot], syllabus_upgrade_type: Upgrade, student: Pilot) -> List[Pilot]:
    c = [p for p in all_pilots if rules.can_fill_seat(p, Qual.WG, syllabus_upgrade_type)]
    return [p for p in c if p is not student]


def _fl_blue_candidates(all_pilots: List[Pilot], syllabus_upgrade_type: Upgrade, student: Pilot) -> List[Pilot]:
    c = [p for p in all_pilots if rules.can_fill_seat(p, Qual.FL, syllabus_upgrade_type)]
    return [p for p in c if p is not student]


def _wg_red_candidates(all_pilots: List[Pilot], syllabus_upgrade_type: Upgrade, student: Pilot) -> List[Pilot]:
    c = [p for p in all_pilots if rules.can_fill_seat(p, Qual.WG, syllabus_upgrade_type)]
    return [p for p in c if p is not student]


def _fl_red_candidates(all_pilots: List[Pilot], syllabus_upgrade_type: Upgrade, student: Pilot) -> List[Pilot]:
    c = [p for p in all_pilots if rules.can_fill_seat(p, Qual.FL, syllabus_upgrade_type)]
    return [p for p in c if p is not student]


def _can_fulfill_student_slot(
    event: SyllabusEvent,
    student: Pilot,
    all_pilots: List[Pilot],
    syllabus_upgrade_type: Upgrade,
    noise: float = 0.0,
) -> bool:
    """
    True only if every required seat for this syllabus repetition has at least one eligible pilot.
    Prevents crediting MQT (or FLUG/IPUG) student sorties when IPs or wingmen cannot be filled.

    ``noise`` is accepted for API symmetry with allocation; preflight remains deterministic.
    Reserved for future stochastic rules (e.g. last-minute cancellations) without changing
    the current hard feasibility checks.
    """
    if event.num_instructor > 0 and not _ips_for_syllabus(all_pilots, syllabus_upgrade_type):
        return False
    if event.num_blue_wg > 0 and not _wg_blue_candidates(all_pilots, syllabus_upgrade_type, student):
        return False
    if event.num_blue_fl > 0 and not _fl_blue_candidates(all_pilots, syllabus_upgrade_type, student):
        return False
    if event.num_red_wg > 0 and not _wg_red_candidates(all_pilots, syllabus_upgrade_type, student):
        return False
    if event.num_red_fl > 0 and not _fl_red_candidates(all_pilots, syllabus_upgrade_type, student):
        return False
    return True


def _apply_student_training_event(
    cfg: SquadronConfig,
    event: SyllabusEvent,
    student: Pilot,
    all_pilots: List[Pilot],
    syllabus_upgrade_type: Upgrade,
    noise: float,
) -> None:
    """Assign instructors and wingmen, then credit the student (sortie or sim). Caller ensures fulfillment."""
    ips = _ips_for_syllabus(all_pilots, syllabus_upgrade_type)
    for _ in range(event.num_instructor):
        assign_sortie(ips, "Blue", noise, cfg.avg_sortie_dur)

    if event.event_type == EventType.SIM:
        student.sim_phase += 1
        student.hours_phase += cfg.avg_sortie_dur
    else:
        student.add_sortie(cfg.avg_sortie_dur, "Blue")

    for _ in range(event.num_blue_wg):
        assign_sortie(
            _wg_blue_candidates(all_pilots, syllabus_upgrade_type, student),
            "Blue",
            noise,
            cfg.avg_sortie_dur,
        )
    for _ in range(event.num_blue_fl):
        assign_sortie(
            _fl_blue_candidates(all_pilots, syllabus_upgrade_type, student),
            "Blue",
            noise,
            cfg.avg_sortie_dur,
        )
    for _ in range(event.num_red_wg):
        assign_sortie(
            _wg_red_candidates(all_pilots, syllabus_upgrade_type, student),
            "Red",
            noise,
            cfg.avg_sortie_dur,
        )
    for _ in range(event.num_red_fl):
        assign_sortie(
            _fl_red_candidates(all_pilots, syllabus_upgrade_type, student),
            "Red",
            noise,
            cfg.avg_sortie_dur,
        )


def process_syllabus_event(
    cfg: SquadronConfig,
    event: SyllabusEvent,
    upgrade_students: List[Pilot],
    all_pilots: List[Pilot],
    syllabus_upgrade_type: Upgrade,
    noise: float,
    deferred_requirements: Optional[List[DeferredSyllabusItem]] = None,
):
    """
    Allocates sorties (and sim events) for a syllabus line item.

    Student credit is applied only when the full crew requirement for that repetition
    can be satisfied (e.g. no MQT student sorties with zero IPs when the event requires an IP).
    """
    for student in upgrade_students:
        for _ in range(event.num_student):
            if not _can_fulfill_student_slot(
                event, student, all_pilots, syllabus_upgrade_type, noise
            ):
                if deferred_requirements is not None:
                    deferred_requirements.append(
                        DeferredSyllabusItem(
                            upgrade=syllabus_upgrade_type,
                            event_name=event.name,
                            event_type=event.event_type,
                            student_year_group=student.year_group,
                            student_squadron_id=student.squadron_id,
                        )
                    )
                continue
            _apply_student_training_event(cfg, event, student, all_pilots, syllabus_upgrade_type, noise)

def run_upgrade_program(
    cfg: SquadronConfig,
    syllabus: List[SyllabusEvent],
    students: List[Pilot],
    all_pilots: List[Pilot],
    upgrade_type: Upgrade,
    noise: float,
    deferred_requirements: Optional[List[DeferredSyllabusItem]] = None,
):
    for event in syllabus:
        process_syllabus_event(
            cfg,
            event,
            students,
            all_pilots,
            upgrade_type,
            noise,
            deferred_requirements,
        )

# ----------------------
# Continuation Training (CT)
# ----------------------
def allocate_continuation_training(
    cfg: SquadronConfig,
    pilots: List[Pilot],
    profile: ContinuationProfile,
    total_capacity: int,
    noise: float
):
    # Calculate how much capacity is left
    used_sorties = sum(p.sortie_phase for p in pilots)
    remaining_capacity = max(0, total_capacity - used_sorties)

    if remaining_capacity <= 0:
        return

    # CT: anyone not in MQT may draw continuation sorties (FLUG/IPUG often need them for RAP).
    ct_candidates = [p for p in pilots if p.upgrade != Upgrade.MQT]

    if not ct_candidates:
        return

    # Calculate bucket sizes
    raw_qty = [(b, remaining_capacity * b.fraction) for b in profile.buckets]
    base_qty = {b: int(x) for b, x in raw_qty}
    
    # Distribute leftover "fractional" sorties
    leftover = remaining_capacity - sum(base_qty.values())
    sorted_remainders = sorted(raw_qty, key=lambda x: x[1]-int(x[1]), reverse=True)
    
    for i in range(leftover):
        bucket = sorted_remainders[i % len(sorted_remainders)][0]
        base_qty[bucket] += 1

    # Execute allocation per bucket
    for bucket, qty in base_qty.items():
        # Find eligible pilots for this specific CT bucket
        # We access the internal hierarchy check from rules since CT doesn't have a syllabus upgrade type
        eligible = [p for p in ct_candidates if rules._qual_hierarchy_check(p.qual, bucket.min_qual)]
        
        for _ in range(qty):
            assign_sortie(eligible, bucket.side, noise, cfg.avg_sortie_dur)

# ----------------------
# Main Simulation Phase
# ----------------------
def run_phase_simulation(cfg: SquadronConfig, pilots: List[Pilot], allocation_noise: float = 0.0):
    """
    Run one training phase. ``allocation_noise`` perturbs who gets each sortie slot
    (see ``assign_sortie``); keep at 0 for deterministic least-loaded allocation.
    """
    # 1. Reset Phase Counters
    for p in pilots:
        p.reset_phase_counters()

    # 2. Select Students
    # Note: If pilots were already assigned upgrades in a previous phase, 
    # you might want to adjust this logic.
    mqt_students = select_upgrade_students(pilots, Upgrade.MQT, cfg.mqt_students)
    flug_students = select_upgrade_students(pilots, Upgrade.FLUG, cfg.flug_students)
    ipug_students = select_upgrade_students(pilots, Upgrade.IPUG, cfg.ipug_students)

    deferred_requirements: List[DeferredSyllabusItem] = []

    # 3. Execute Syllabi
    run_upgrade_program(cfg, MQT_SYLLABUS, mqt_students, pilots, Upgrade.MQT, allocation_noise, deferred_requirements)
    run_upgrade_program(cfg, FLUG_SYLLABUS, flug_students, pilots, Upgrade.FLUG, allocation_noise, deferred_requirements)
    run_upgrade_program(cfg, IPUG_SYLLABUS, ipug_students, pilots, Upgrade.IPUG, allocation_noise, deferred_requirements)

    # 4. Continuation Training — capacity = (PAA × UTE) sorties/month × phase length in notional months.
    phase_months = cfg.phase_length_months
    total_capacity = max(0, round(total_phase_capacity(cfg) * phase_months))
    
    allocate_continuation_training(cfg, pilots, CONTINUATION_PROFILE, total_capacity, allocation_noise)

    # 5. Finalize stats (RAP shortfall scales with phase_length_days via update_total)
    for p in pilots:
        p.set_rap_requirement()
        p.update_total(cfg.phase_length_days)
        p.update_monthly(cfg.phase_length_days)

    cfg.last_phase_upgrade_handoff = PhaseUpgradeHandoff(
        mqt_syllabus_complete=_syllabus_track_complete(mqt_students, Upgrade.MQT, deferred_requirements),
        flug_syllabus_complete=_syllabus_track_complete(flug_students, Upgrade.FLUG, deferred_requirements),
        ipug_syllabus_complete=_syllabus_track_complete(ipug_students, Upgrade.IPUG, deferred_requirements),
        deferred_requirements=list(deferred_requirements),
    )

    return pilots

# ----------------------
# Reporting
# ----------------------
def print_phase_summary(pilots: List[Pilot], cfg: SquadronConfig, verbose: bool = True):
    print(f"\n=== Phase Summary ({cfg.phase_length_days} d ≈ {cfg.phase_length_months:.2f} mo) ===")
    
    groups = {
        "MQT Students": [p for p in pilots if p.upgrade == Upgrade.MQT],
        "FLUG Students": [p for p in pilots if p.upgrade == Upgrade.FLUG],
        "IPUG Students": [p for p in pilots if p.upgrade == Upgrade.IPUG],
        "Line Wingmen": [p for p in pilots if p.qual == Qual.WG and p.upgrade == Upgrade.NONE],
        "Line FLs": [p for p in pilots if p.qual == Qual.FL and p.upgrade == Upgrade.NONE],
        "IPs": [p for p in pilots if p.qual == Qual.IP]
    }

    for name, group in groups.items():
        if not group:
            print(f"{name}: None")
            continue
            
        avg_sorties = sum(p.sortie_monthly for p in group) / len(group)
        avg_blue_sorties = sum(p.sortie_blue_monthly for p in group) / len(group)
        avg_red_sorties = sum(p.sortie_red_monthly for p in group) / len(group)
        print(f"{name} ({len(group)}): Avg Mo. Sorties {avg_sorties:.1f}, Avg Mo. Blue Sorties {avg_blue_sorties:.1f}, Avg Mo. Red Sorties {avg_red_sorties:.1f}")

    if verbose:
        for name, group in groups.items():
            for p in group:
                print(f'{p.qual}/{p.upgrade}: Monthly {p.sortie_monthly}, Blue {p.sortie_blue_monthly}, Red {p.sortie_red_monthly}')