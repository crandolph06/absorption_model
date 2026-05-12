import math
import random
from typing import List, Optional, Set, Tuple

from src.models import (
    SIM_RAP_MONTHLY,
    SquadronConfig,
    Pilot,
    Qual,
    Upgrade,
    EventType,
    DeferredSyllabusItem,
    PhaseUpgradeHandoff,
)
from src.syllabi import (
    CONTINUATION_PROFILE,
    FLUG_SYLLABUS,
    IPUG_SYLLABUS,
    MQT_SYLLABUS,
    ContinuationProfile,
    SyllabusEvent,
)
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


def initial_sim_session_budget(cfg: SquadronConfig) -> float:
    """
    Total simulator **session** budget for the phase (fractional sessions allowed).

    Each month the wing flies about ``cfg.sim_sessions_monthly`` session lines; each
    line has ``cfg.sim_bays_per_session`` bays (packing: up to that many pilots in one
    session time slice, or ``ceil(P / bays)`` sessions for one training evolution with
    ``P`` concurrent pilots).
    """
    return max(0.0, float(cfg.sim_sessions_monthly) * cfg.phase_length_months)


def _solo_sim_session_fraction(cfg: SquadronConfig) -> float:
    """Sessions consumed by one solo EP / RAP sim when four solos pack into one 4-bay session."""
    b = max(1, int(cfg.sim_bays_per_session))
    return 1.0 / float(b)


def _syllabus_sim_session_cost(event: SyllabusEvent, cfg: SquadronConfig) -> float:
    """
    Sessions consumed for one student repetition of this SIM (concurrent crew ``P``).

    CAF upgrade syllabi are written so no SIM line needs more than ``sim_bays_per_session``
    (4) people at once, so ``P <= 4`` and this is typically **1** session. The ``ceil(P/b)``
    form stays correct if a future row ever exceeds four concurrent seats.
    """
    p = (
        1
        + event.num_instructor
        + event.num_blue_wg
        + event.num_blue_fl
        + event.num_red_wg
        + event.num_red_fl
    )
    b = max(1, int(cfg.sim_bays_per_session))
    return float(math.ceil(p / float(b)))

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
        p.sorties_at_upgrade_start = p.sorties_flown
        p.sims_at_upgrade_start = p.sims_flown

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


def assign_sim_participant(
    candidates: List[Pilot], noise: float = 0.0, avg_event_dur: float = 1.3
) -> bool:
    """
    Pick a pilot for one simulator event (upgrade sim IP, sim wingman, or sim RAP).
    Uses lowest ``sim_phase`` with optional noise, analogous to ``assign_sortie``.
    """
    if not candidates:
        return False

    n = max(0.0, float(noise))

    def _effective_sim_util(p: Pilot) -> float:
        jitter = random.uniform(-n, n) if n > 0 else 0.0
        return p.sim_phase + jitter

    pool = list(candidates)
    pool.sort(key=_effective_sim_util)
    pool[0].add_sim(avg_event_dur)
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


def _full_syllabus_for_upgrade(upgrade: Upgrade) -> List[SyllabusEvent]:
    if upgrade == Upgrade.MQT:
        return list(MQT_SYLLABUS)
    if upgrade == Upgrade.FLUG:
        return list(FLUG_SYLLABUS)
    if upgrade == Upgrade.IPUG:
        return list(IPUG_SYLLABUS)
    return []


def _find_pilot_for_deferred(all_pilots: List[Pilot], item: DeferredSyllabusItem) -> Optional[Pilot]:
    for p in all_pilots:
        if p.upgrade != item.upgrade:
            continue
        if p.year_group == item.student_year_group and p.squadron_id == item.student_squadron_id:
            return p
    return None


def _deferral_signature(
    upgrade: Upgrade,
    syllabus_event_index: int,
    student: Pilot,
    student_event_repetition: int,
) -> Tuple[Upgrade, int, int, int, int]:
    return (
        upgrade,
        syllabus_event_index,
        student.year_group,
        student.squadron_id,
        student_event_repetition,
    )


def process_pending_deferred_requirements(
    cfg: SquadronConfig,
    pilots: List[Pilot],
    allocation_noise: float,
    deferred_requirements: List[DeferredSyllabusItem],
    completed_keys: Set[Tuple[Upgrade, int, int, int, int]],
    sim_session_budget: Optional[List[float]] = None,
) -> None:
    """Replay carried-over syllabus deferrals (SORTIE and SIM) before the main syllabus walk."""
    pending = list(cfg.pending_deferred_requirements)
    cfg.pending_deferred_requirements.clear()
    for item in sorted(
        pending,
        key=lambda d: (
            d.upgrade.value,
            d.syllabus_event_index,
            d.student_year_group,
            d.student_squadron_id,
            d.student_event_repetition,
        ),
    ):
        student = _find_pilot_for_deferred(pilots, item)
        if student is None:
            continue
        syllabus = _full_syllabus_for_upgrade(item.upgrade)
        if not syllabus or not (0 <= item.syllabus_event_index < len(syllabus)):
            continue
        event = syllabus[item.syllabus_event_index]
        if item.event_type != event.event_type:
            continue
        sig = _deferral_signature(
            item.upgrade,
            item.syllabus_event_index,
            student,
            item.student_event_repetition,
        )
        if sig in completed_keys:
            continue
        if event.event_type == EventType.SIM and sim_session_budget is not None:
            cost = _syllabus_sim_session_cost(event, cfg)
            if sim_session_budget[0] + 1e-9 < cost:
                deferred_requirements.append(item)
                continue
        if not _can_fulfill_student_slot(
            event, student, pilots, item.upgrade, allocation_noise
        ):
            deferred_requirements.append(item)
            continue
        if event.event_type == EventType.SIM and sim_session_budget is not None:
            sim_session_budget[0] -= _syllabus_sim_session_cost(event, cfg)
        _apply_student_training_event(
            cfg, event, student, pilots, item.upgrade, allocation_noise
        )
        completed_keys.add(sig)


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
    """
    Crew and credit one syllabus event.

    Sorties credit ``sortie_phase`` / ``add_sortie``. Upgrade SIMs credit ``sim_phase`` /
    ``add_sim`` for every participant; instructors are IPs only when ``num_instructor`` > 0.
    """
    if event.event_type == EventType.SIM:
        ips = _ips_for_syllabus(all_pilots, syllabus_upgrade_type)
        for _ in range(event.num_instructor):
            assign_sim_participant(ips, noise, cfg.avg_sortie_dur)
        student.add_sim(cfg.avg_sortie_dur)
        for _ in range(event.num_blue_wg):
            assign_sim_participant(
                _wg_blue_candidates(all_pilots, syllabus_upgrade_type, student),
                noise,
                cfg.avg_sortie_dur,
            )
        for _ in range(event.num_blue_fl):
            assign_sim_participant(
                _fl_blue_candidates(all_pilots, syllabus_upgrade_type, student),
                noise,
                cfg.avg_sortie_dur,
            )
        for _ in range(event.num_red_wg):
            assign_sim_participant(
                _wg_red_candidates(all_pilots, syllabus_upgrade_type, student),
                noise,
                cfg.avg_sortie_dur,
            )
        for _ in range(event.num_red_fl):
            assign_sim_participant(
                _fl_red_candidates(all_pilots, syllabus_upgrade_type, student),
                noise,
                cfg.avg_sortie_dur,
            )
        return

    ips = _ips_for_syllabus(all_pilots, syllabus_upgrade_type)
    for _ in range(event.num_instructor):
        assign_sortie(ips, "Blue", noise, cfg.avg_sortie_dur)

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
    syllabus_event_index: int,
    deferred_requirements: Optional[List[DeferredSyllabusItem]] = None,
    completed_keys: Optional[Set[Tuple[Upgrade, int, int, int, int]]] = None,
    sim_session_budget: Optional[List[float]] = None,
):
    """
    Allocate one syllabus line (SORTIE or SIM). SIM rows consume simulator **session**
    budget from ``sim_session_budget`` (see ``_syllabus_sim_session_cost``).
    """
    for student in upgrade_students:
        for student_event_repetition in range(event.num_student):
            if completed_keys is not None:
                sig = _deferral_signature(
                    syllabus_upgrade_type,
                    syllabus_event_index,
                    student,
                    student_event_repetition,
                )
                if sig in completed_keys:
                    continue
            if event.event_type == EventType.SIM and sim_session_budget is not None:
                cost = _syllabus_sim_session_cost(event, cfg)
                if sim_session_budget[0] + 1e-9 < cost:
                    if deferred_requirements is not None:
                        deferred_requirements.append(
                            DeferredSyllabusItem(
                                upgrade=syllabus_upgrade_type,
                                event_name=event.name,
                                event_type=EventType.SIM,
                                syllabus_event_index=syllabus_event_index,
                                student_event_repetition=student_event_repetition,
                                student_year_group=student.year_group,
                                student_squadron_id=student.squadron_id,
                            )
                        )
                    continue
            if not _can_fulfill_student_slot(
                event, student, all_pilots, syllabus_upgrade_type, noise
            ):
                if deferred_requirements is not None:
                    deferred_requirements.append(
                        DeferredSyllabusItem(
                            upgrade=syllabus_upgrade_type,
                            event_name=event.name,
                            event_type=event.event_type,
                            syllabus_event_index=syllabus_event_index,
                            student_event_repetition=student_event_repetition,
                            student_year_group=student.year_group,
                            student_squadron_id=student.squadron_id,
                        )
                    )
                continue
            if event.event_type == EventType.SIM and sim_session_budget is not None:
                sim_session_budget[0] -= _syllabus_sim_session_cost(event, cfg)
            _apply_student_training_event(
                cfg, event, student, all_pilots, syllabus_upgrade_type, noise
            )


def run_upgrade_program(
    cfg: SquadronConfig,
    syllabus: List[SyllabusEvent],
    students: List[Pilot],
    all_pilots: List[Pilot],
    upgrade_type: Upgrade,
    noise: float,
    deferred_requirements: Optional[List[DeferredSyllabusItem]] = None,
    completed_keys: Optional[Set[Tuple[Upgrade, int, int, int, int]]] = None,
    sim_session_budget: Optional[List[float]] = None,
):
    for syllabus_event_index, event in enumerate(syllabus):
        process_syllabus_event(
            cfg,
            event,
            students,
            all_pilots,
            upgrade_type,
            noise,
            syllabus_event_index,
            deferred_requirements,
            completed_keys,
            sim_session_budget,
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


def allocate_ep_sim(
    cfg: SquadronConfig,
    pilots: List[Pilot],
    noise: float,
    sim_session_budget: List[float],
) -> None:
    """
    One EP sim per pilot per notional month (tracks ``Pilot.ep_sim_phase``).

    Uses ``1 / sim_bays_per_session`` of a session line per solo EP when four solos
    pack into one 4-bay session.
    """
    months = cfg.phase_length_months
    if months <= 0:
        return
    frac = _solo_sim_session_fraction(cfg)
    eligible = [p for p in pilots if p.qual in (Qual.WG, Qual.FL, Qual.IP)]
    n = max(0.0, float(noise))
    while sim_session_budget[0] + 1e-9 >= frac:
        need_ep = [p for p in eligible if p.ep_sim_phase + 1e-9 < months]
        if not need_ep:
            break

        def _ep_key(p: Pilot) -> Tuple[float, float]:
            ep_def = months - p.ep_sim_phase
            jitter = random.uniform(-n, n) if n > 0 else 0.0
            return (-ep_def, p.sim_phase + jitter)

        need_ep.sort(key=_ep_key)
        pilot = need_ep[0]
        pilot.add_sim(cfg.avg_sortie_dur, count_ep=True)
        sim_session_budget[0] -= frac


def allocate_extra_rap_sims(
    cfg: SquadronConfig,
    pilots: List[Pilot],
    noise: float,
    sim_session_budget: List[float],
) -> None:
    """
    After upgrade syllabus and EP allocation, top up solo sims toward ``SIM_RAP_MONTHLY``
    (3 / month) using the same fractional session cost as EP packing.
    """
    months = cfg.phase_length_months
    if months <= 0:
        return
    target = SIM_RAP_MONTHLY * months
    frac = _solo_sim_session_fraction(cfg)
    eligible = [p for p in pilots if p.qual in (Qual.WG, Qual.FL, Qual.IP)]
    while sim_session_budget[0] + 1e-9 >= frac:
        best_p: Optional[Pilot] = None
        best_def = 0.0
        for p in eligible:
            deficit = max(0.0, target - p.sim_phase)
            if deficit > best_def:
                best_def = deficit
                best_p = p
        if best_p is None or best_def < 1e-9:
            break
        assign_sim_participant([best_p], noise, cfg.avg_sortie_dur)
        sim_session_budget[0] -= frac


# ----------------------
# Main Simulation Phase
# ----------------------
def run_phase_simulation(cfg: SquadronConfig, pilots: List[Pilot], allocation_noise: float = 0.0):
    """
    Run one training phase.

    Simulator **session** budget for the phase is ``sim_sessions_monthly × phase_months``
    (default ~30 sessions / month). Each SIM syllabus evolution uses ``ceil(P / bays)``
    sessions for ``P`` concurrent participants; solo EP and extra RAP sims use
    ``1 / bays`` of a session (four solos per 4-bay line).

    Allocation order: pending deferrals → upgrade syllabi (sorties + SIMs) → continuation
    sorties → EP sim (1 / pilot / month) → extra sim RAP toward 3 / month total.
    """
    # 1. Reset Phase Counters
    for p in pilots:
        p.reset_phase_counters()

    phase_months = cfg.phase_length_months
    sim_session_budget = [initial_sim_session_budget(cfg)]

    deferred_requirements: List[DeferredSyllabusItem] = []
    completed_keys: Set[Tuple[Upgrade, int, int, int, int]] = set()
    process_pending_deferred_requirements(
        cfg,
        pilots,
        allocation_noise,
        deferred_requirements,
        completed_keys,
        sim_session_budget,
    )

    # 2. Select new upgrade starts; include anyone already in-track from a prior phase.
    select_upgrade_students(pilots, Upgrade.MQT, cfg.mqt_students)
    select_upgrade_students(pilots, Upgrade.FLUG, cfg.flug_students)
    select_upgrade_students(pilots, Upgrade.IPUG, cfg.ipug_students)

    mqt_students = [p for p in pilots if p.upgrade == Upgrade.MQT]
    flug_students = [p for p in pilots if p.upgrade == Upgrade.FLUG]
    ipug_students = [p for p in pilots if p.upgrade == Upgrade.IPUG]

    # 3. Execute syllabi (sorties + SIMs)
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

    # 4. Continuation Training — capacity = (PAA × UTE) sorties/month × phase length in notional months.
    total_capacity = max(0, round(total_phase_capacity(cfg) * phase_months))

    allocate_continuation_training(cfg, pilots, CONTINUATION_PROFILE, total_capacity, allocation_noise)

    # 5. EP sim (1 / pilot / month), then remaining sim RAP toward 3 / month (solo packing).
    allocate_ep_sim(cfg, pilots, allocation_noise, sim_session_budget)
    allocate_extra_rap_sims(cfg, pilots, allocation_noise, sim_session_budget)

    # 6. Finalize stats (sortie RAP vs ``sortie_phase``; sim RAP vs ``sim_phase``)
    for p in pilots:
        p.set_rap_requirement()
        p.update_total(cfg.phase_length_days)
        p.update_monthly(cfg.phase_length_days)

    cfg.pending_deferred_requirements = list(deferred_requirements)

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
        avg_sims = sum(p.sim_monthly for p in group) / len(group)
        avg_sim_sf = sum(p.sim_rap_shortfall for p in group) / len(group)
        print(
            f"{name} ({len(group)}): Avg Mo. Sorties {avg_sorties:.1f}, Blue {avg_blue_sorties:.1f}, Red {avg_red_sorties:.1f}; "
            f"Avg Mo. Sims {avg_sims:.1f}, Avg sim RAP shortfall {avg_sim_sf:.2f}"
        )

    if verbose:
        for name, group in groups.items():
            for p in group:
                print(
                    f'{p.qual}/{p.upgrade}: Sorties/mo {p.sortie_monthly:.2f}, Blue {p.sortie_blue_monthly:.2f}, Red {p.sortie_red_monthly:.2f}; '
                    f'Sims/mo {p.sim_monthly:.2f}; sortie RAP shortfall {p.rap_shortfall:.2f}, sim RAP shortfall {p.sim_rap_shortfall:.2f}'
                )