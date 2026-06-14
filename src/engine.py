import math
import random
from typing import List, Optional, Set
from src.models import (
    EventType,
    SquadronConfig,
    Pilot,
    Qual,
    Upgrade,
    MAX_MONTHLY_EVENTS,
    PHASE_DAYS_PER_NOTIONAL_MONTH,
)
from src.simulation_config import SimulationConfig
from src.syllabi import SyllabusEvent, ContinuationProfile, ContinuationBucket
from src import rules
from src.syllabi import (
    MQT_SYLLABUS,
    FLUG_SYLLABUS,
    IPUG_SYLLABUS,
    CONTINUATION_PROFILE,
    incomplete_burden,
    syllabus_burden_fraction,
    syllabus_burden_per_student,
)

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

def total_phase_capacity(cfg: SquadronConfig) -> float:
    return cfg.ute * cfg.paa


def phase_upgrade_metrics(
    pilots: List[Pilot],
    mqt_syllabus: List[SyllabusEvent] = MQT_SYLLABUS,
    flug_syllabus: List[SyllabusEvent] = FLUG_SYLLABUS,
    ipug_syllabus: List[SyllabusEvent] = IPUG_SYLLABUS,
) -> dict:
    """
    End-of-phase deferral stats from ``Pilot.incomplete_syllabus_items``.

    Counts **line slots** (student + support), not syllabus event rows.
    ``remaining_*_syllabi_sorties_only`` is squad deferred sortie slots divided by
    one-student sortie syllabus burden (syllabi-worth of outstanding sortie work).
    """
    deferred_mqt_sorties = deferred_flug_sorties = deferred_ipug_sorties = 0
    deferred_mqt_sims = deferred_flug_sims = deferred_ipug_sims = 0
    held_back_mqt = held_back_flug = held_back_ipug = 0

    mqt_sortie_burden, mqt_sim_burden = syllabus_burden_per_student(mqt_syllabus)
    flug_sortie_burden, flug_sim_burden = syllabus_burden_per_student(flug_syllabus)
    ipug_sortie_burden, ipug_sim_burden = syllabus_burden_per_student(ipug_syllabus)

    for p in pilots:
        if not p.incomplete_syllabus_items:
            continue
        sortie_slots, sim_slots = incomplete_burden(p.incomplete_syllabus_items)

        if p.upgrade == Upgrade.MQT:
            deferred_mqt_sorties += sortie_slots
            deferred_mqt_sims += sim_slots
            held_back_mqt += 1
        elif p.upgrade == Upgrade.FLUG:
            deferred_flug_sorties += sortie_slots
            deferred_flug_sims += sim_slots
            held_back_flug += 1
        elif p.upgrade == Upgrade.IPUG:
            deferred_ipug_sorties += sortie_slots
            deferred_ipug_sims += sim_slots
            held_back_ipug += 1

    mqt_sortie_frac, mqt_sim_frac = syllabus_burden_fraction(
        deferred_mqt_sorties, deferred_mqt_sims, mqt_sortie_burden, mqt_sim_burden
    )
    flug_sortie_frac, flug_sim_frac = syllabus_burden_fraction(
        deferred_flug_sorties, deferred_flug_sims, flug_sortie_burden, flug_sim_burden
    )
    ipug_sortie_frac, ipug_sim_frac = syllabus_burden_fraction(
        deferred_ipug_sorties, deferred_ipug_sims, ipug_sortie_burden, ipug_sim_burden
    )

    return {
        "deferred_mqt_sorties": deferred_mqt_sorties,
        "deferred_flug_sorties": deferred_flug_sorties,
        "deferred_ipug_sorties": deferred_ipug_sorties,
        "deferred_mqt_sims": deferred_mqt_sims,
        "deferred_flug_sims": deferred_flug_sims,
        "deferred_ipug_sims": deferred_ipug_sims,
        "held_back_mqt": held_back_mqt,
        "held_back_flug": held_back_flug,
        "held_back_ipug": held_back_ipug,
        "remaining_mqt_syllabi": mqt_sortie_frac + mqt_sim_frac,
        "remaining_flug_syllabi": flug_sortie_frac + flug_sim_frac,
        "remaining_ipug_syllabi": ipug_sortie_frac + ipug_sim_frac,
        "remaining_mqt_syllabi_sorties_only": mqt_sortie_frac,
        "remaining_flug_syllabi_sorties_only": flug_sortie_frac,
        "remaining_ipug_syllabi_sorties_only": ipug_sortie_frac,
        "remaining_mqt_syllabi_sims_only": mqt_sim_frac,
        "remaining_flug_syllabi_sims_only": flug_sim_frac,
        "remaining_ipug_syllabi_sims_only": ipug_sim_frac,
    }


def graduate_completed_upgrades(pilots: List[Pilot]) -> None:
    """Graduate upgrade students with no incomplete syllabus lines (Layer 1 end-of-phase)."""
    for p in pilots:
        if p.upgrade != Upgrade.NONE and not p.incomplete_syllabus_items:
            p.graduate()


def apply_deferred_burden_to_squadron(cfg: SquadronConfig, metrics: dict) -> None:
    """Store carryover iron/sim burden on ``cfg`` for the next phase."""
    cfg.deferred_sortie_burden = int(
        metrics["deferred_mqt_sorties"]
        + metrics["deferred_flug_sorties"]
        + metrics["deferred_ipug_sorties"]
    )
    cfg.deferred_sim_burden = int(
        metrics["deferred_mqt_sims"]
        + metrics["deferred_flug_sims"]
        + metrics["deferred_ipug_sims"]
    )
    cfg.mqt_sortie_carry = metrics["remaining_mqt_syllabi_sorties_only"]
    cfg.flug_sortie_carry = metrics["remaining_flug_syllabi_sorties_only"]
    cfg.ipug_sortie_carry = metrics["remaining_ipug_syllabi_sorties_only"]
    cfg.mqt_sim_carry = metrics["remaining_mqt_syllabi_sims_only"]
    cfg.flug_sim_carry = metrics["remaining_flug_syllabi_sims_only"]
    cfg.ipug_sim_carry = metrics["remaining_ipug_syllabi_sims_only"]


# ----------------------
# Selection Phase
# ----------------------
def select_upgrade_students(pilots: List[Pilot], upgrade_type: Upgrade, count: int) -> List[Pilot]:
    """Pick up to ``count`` new students (``upgrade == NONE``). Carryover students stay in upgrade."""
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
def _eligible_for_event(candidates: List[Pilot], phase_length_days: float) -> List[Pilot]:
    return [p for p in candidates if p.has_events_capacity(phase_length_days)]


def _total_phase_events(p: Pilot) -> float:
    return p.sortie_phase + p.sim_phase


def _type_specific_event_count(p: Pilot, event_type: EventType, side: str = "Blue") -> float:
    """Tie-breaker: least sims, blue sorties, or red sorties for the event being assigned."""
    if event_type == EventType.SIM:
        return p.sim_phase
    if side == "Red":
        return p.sortie_red_phase
    return p.sortie_blue_phase


_QUAL_RANK = {Qual.WG: 0, Qual.FL: 1, Qual.IP: 2}


def _qual_rank(p: Pilot) -> int:
    """Tie-breaker: prefer WG over FL over IP when event loads are equal."""
    return _QUAL_RANK[p.qual]


def _allocation_sort_key(
    p: Pilot, event_type: EventType, side: str = "Blue", noise: float = 0.0
) -> tuple:
    """Primary: total phase events; tie-break: event type count, then qual (WG < FL < IP)."""
    return (
        _total_phase_events(p) + random.uniform(0, noise),
        _type_specific_event_count(p, event_type, side),
        _qual_rank(p),
    )


def _can_assign_distinct_from_pool(pool: List[Pilot], count: int, phase_length_days: float) -> bool:
    """Whether ``count`` distinct pilots in ``pool`` can each take one more event under the cap."""
    if count <= 0:
        return True
    months = float(phase_length_days) / PHASE_DAYS_PER_NOTIONAL_MONTH
    if months <= 0:
        return False
    max_phase_events = MAX_MONTHLY_EVENTS * months
    eligible = _eligible_for_event(pool, phase_length_days)
    if len(eligible) < count:
        return False
    usage = {id(p): p.phase_events() for p in eligible}
    picked: Set[int] = set()
    for _ in range(count):
        available = [
            p for p in eligible
            if id(p) not in picked and usage[id(p)] + 1 <= max_phase_events + 1e-9
        ]
        if not available:
            return False
        available.sort(key=lambda p: (usage[id(p)], _qual_rank(p)))
        winner = available[0]
        picked.add(id(winner))
        usage[id(winner)] += 1
    return True


def assign_sortie(
    cfg: SquadronConfig,
    candidates: List[Pilot],
    phase_length_days: float,
    side: str = "Blue",
    noise: float = 0.0,
    exclude: Optional[Set[int]] = None,
    single_ship: bool = False,
) -> bool:
    """
    Selects the best candidate (lowest total events, then least blue/red sorties) to fly a sortie.
    Returns True if a pilot was found and assigned, False otherwise.
    """
    exclude = exclude if exclude is not None else set()
    candidates = [p for p in candidates if id(p) not in exclude]
    candidates = _eligible_for_event(candidates, phase_length_days)
    if not candidates:
        return False

    candidates.sort(key=lambda p: _allocation_sort_key(p, EventType.SORTIE, side, noise))

    winner = candidates[0]
    winner.add_sortie(avg_sortie_dur=cfg.avg_sortie_dur, side=side, single_ship=single_ship)
    exclude.add(id(winner))
    return True


def assign_sim(
    cfg: SquadronConfig,
    candidates: List[Pilot],
    phase_length_days: float,
    noise: float = 0.0,
    exclude: Optional[Set[int]] = None,
) -> bool:
    """
    Selects the best candidate (lowest total events, then least sims) for a simulator event.
    Returns True if a pilot was found and assigned, False otherwise.
    """
    exclude = exclude if exclude is not None else set()
    candidates = [p for p in candidates if id(p) not in exclude]
    candidates = _eligible_for_event(candidates, phase_length_days)
    if not candidates:
        return False

    candidates.sort(key=lambda p: _allocation_sort_key(p, EventType.SIM, noise=noise))
    winner = candidates[0]
    winner.add_sim(cfg.avg_sortie_dur)
    exclude.add(id(winner))
    return True

def check_syllabus_resources(
    event: SyllabusEvent,
    all_pilots: List[Pilot],
    syllabus_upgrade_type: Upgrade,
    total_capacity: int,
    cfg: SquadronConfig,
    phase_length_days: float,
    student: Optional[Pilot] = None,
) -> bool:
    """Enough distinct support pilots, iron/sim capacity, and event-cap headroom for one student line."""
    ips = [
        p for p in all_pilots
        if rules.can_fill_seat(pilot=p, min_qual=Qual.IP) and p is not student
    ]
    if len(ips) < event.num_instructor:
        return False
    if not _can_assign_distinct_from_pool(ips, event.num_instructor, phase_length_days):
        return False

    wg_pool = [
        p for p in all_pilots
        if rules.can_fill_seat(pilot=p, min_qual=Qual.WG) and p is not student
    ]
    fl_pool = [
        p for p in all_pilots
        if rules.can_fill_seat(pilot=p, min_qual=Qual.FL) and p is not student
    ]
    wg_seats = event.num_blue_wg + event.num_red_wg
    fl_seats = event.num_blue_fl + event.num_red_fl
    if len(wg_pool) < wg_seats:
        return False
    if len(fl_pool) < fl_seats:
        return False
    if not _can_assign_distinct_from_pool(wg_pool, wg_seats, phase_length_days):
        return False
    if not _can_assign_distinct_from_pool(fl_pool, fl_seats, phase_length_days):
        return False

    if event.event_type == EventType.SORTIE:
        slots = 1 + event.num_instructor + event.num_blue_wg + event.num_blue_fl + event.num_red_wg + event.num_red_fl
        if sum(p.sortie_phase for p in all_pilots) + slots > total_capacity:
            return False
    return True
# ----------------------
# Syllabus Execution
# ----------------------
def process_syllabus_event(
    event: SyllabusEvent, 
    upgrade_students: List[Pilot], 
    all_pilots: List[Pilot], 
    syllabus_upgrade_type: Upgrade,
    noise: float,
    cfg: SquadronConfig,
    phase_length_days: float,
    total_capacity: int,
):
    """
    Allocates sorties for a specific syllabus event.
    CRITICAL FIX: Support sorties are now generated PER student sortie.
    """
    for student in upgrade_students:
        for _ in range(event.num_student):
            if not student.has_events_capacity(phase_length_days):
                if event not in student.incomplete_syllabus_items:
                    student.incomplete_syllabus_items.append(event)
                continue
            if not check_syllabus_resources(
                event, all_pilots, syllabus_upgrade_type, total_capacity, cfg, phase_length_days, student=student
            ):
                if event not in student.incomplete_syllabus_items:
                    student.incomplete_syllabus_items.append(event)
                continue
            if event in student.incomplete_syllabus_items:
                student.incomplete_syllabus_items.remove(event)

            line_assigned: Set[int] = {id(student)}
            line_ok = True
            if event.event_type == EventType.SIM:
                for _ in range(event.num_instructor):
                    ips = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.IP)]
                    if not assign_sim(cfg, ips, phase_length_days, noise, exclude=line_assigned):
                        line_ok = False
                        break
                if line_ok:
                    for _ in range(event.num_blue_fl):
                        candidates = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.FL)]
                        if not assign_sim(cfg, candidates, phase_length_days, noise, exclude=line_assigned):
                            line_ok = False
                            break
                if line_ok:
                    for _ in range(event.num_red_fl):
                        candidates = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.FL)]
                        if not assign_sim(cfg, candidates, phase_length_days, noise, exclude=line_assigned):
                            line_ok = False
                if line_ok:
                    for _ in range(event.num_blue_wg):
                        candidates = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.WG)]
                        if not assign_sim(cfg, candidates, phase_length_days, noise, exclude=line_assigned):
                            line_ok = False
                            break
                if line_ok:
                    for _ in range(event.num_red_wg):
                        candidates = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.WG)]
                        if not assign_sim(cfg, candidates, phase_length_days, noise, exclude=line_assigned):
                            line_ok = False
                            break
                if line_ok:
                    student.add_sim(cfg.avg_sortie_dur)
            else:
                for _ in range(event.num_instructor):
                    ips = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.IP)]
                    if not assign_sortie(cfg, ips, phase_length_days, "Blue", noise, exclude=line_assigned):
                        line_ok = False
                        break
                if line_ok:
                    for _ in range(event.num_blue_fl):
                        candidates = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.FL)]
                        if not assign_sortie(cfg, candidates, phase_length_days, "Blue", noise, exclude=line_assigned):
                            line_ok = False
                            break
                if line_ok:
                    for _ in range(event.num_red_fl):
                        candidates = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.FL)]
                        if not assign_sortie(cfg, candidates, phase_length_days, "Red", noise, exclude=line_assigned):
                            line_ok = False
                if line_ok:
                    for _ in range(event.num_blue_wg):
                        candidates = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.WG)]
                        if not assign_sortie(cfg, candidates, phase_length_days, "Blue", noise, exclude=line_assigned):
                            line_ok = False
                            break
                if line_ok:
                    for _ in range(event.num_red_wg):
                        candidates = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.WG)]
                        if not assign_sortie(cfg, candidates, phase_length_days, "Red", noise, exclude=line_assigned):
                            line_ok = False
                            break
                if line_ok:
                    student.add_sortie(cfg.avg_sortie_dur, "Blue")

            if not line_ok and event not in student.incomplete_syllabus_items:
                student.incomplete_syllabus_items.append(event)

def run_upgrade_program(
    syllabus: List[SyllabusEvent],
    students: List[Pilot],
    all_pilots: List[Pilot],
    upgrade_type: Upgrade,
    noise: float,
    cfg: SquadronConfig,
    phase_length_days: float,
    total_capacity: int,
):
    for event in syllabus:
        process_syllabus_event(
            event, students, all_pilots, upgrade_type, noise,
            cfg=cfg, phase_length_days=phase_length_days, total_capacity=total_capacity,
        )

# ----------------------
# Continuation Training (CT)
# ----------------------
_CT_ROUND_ROBIN_ORDER = {
    ("Blue", Qual.FL): 0,
    ("Red", Qual.FL): 1,
    ("Blue", Qual.WG): 2,
    ("Red", Qual.WG): 3,
}


def _ct_bucket_round_robin_key(bucket: ContinuationBucket) -> int:
    return _CT_ROUND_ROBIN_ORDER.get((bucket.side, bucket.min_qual), 99)


def _allocate_ct_buckets_round_robin(
    buckets: List[ContinuationBucket],
    remaining: dict,
    ct_candidates: List[Pilot],
    cfg: SquadronConfig,
    phase_length_days: float,
    noise: float,
    single_ship: bool = False,
) -> int:
    """Assign CT sorties round-robin across ``buckets``; return count assigned."""
    assigned = 0
    while sum(remaining.get(b, 0) for b in buckets) > 0:
        assigned_this_pass = False
        for bucket in buckets:
            if remaining.get(bucket, 0) <= 0:
                continue
            eligible = [
                p for p in ct_candidates
                if rules.can_fill_seat(pilot=p, min_qual=bucket.min_qual)
            ]
            if assign_sortie(
                cfg=cfg,
                candidates=eligible,
                phase_length_days=phase_length_days,
                side=bucket.side,
                noise=noise,
                single_ship=single_ship,
            ):
                remaining[bucket] -= 1
                assigned += 1
                assigned_this_pass = True
            else:
                remaining[bucket] = 0

        if not assigned_this_pass:
            break
    return assigned


def allocate_continuation_training(
    pilots: List[Pilot],
    profile: ContinuationProfile,
    total_capacity: int,
    noise: float,
    cfg: SquadronConfig,
    phase_length_days: float,
    ct_sortie_cap: Optional[int] = None,
):
    if ct_sortie_cap is not None:
        remaining_capacity = max(0, ct_sortie_cap)
    else:
        used_sorties = sum(p.sortie_phase for p in pilots)
        remaining_capacity = max(0, total_capacity - used_sorties)

    if remaining_capacity <= 0:
        return

    # Identify CT candidates (anyone NOT in an active upgrade)
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

    # FL-led CT first so we know shortfall before wingmen fly.
    fl_buckets = sorted(
        [b for b in base_qty if b.min_qual == Qual.FL],
        key=_ct_bucket_round_robin_key,
    )
    wg_buckets = sorted(
        [b for b in base_qty if b.min_qual == Qual.WG],
        key=_ct_bucket_round_robin_key,
    )
    remaining = dict(base_qty)
    fl_planned = sum(base_qty[b] for b in fl_buckets)

    fl_assigned = _allocate_ct_buckets_round_robin(
        fl_buckets, remaining, ct_candidates, cfg, phase_length_days, noise
    )
    fl_ct_shortfall = fl_planned - fl_assigned

    # WG CT after FL mix; tag single-ship when FL buckets could not be fully staffed.
    if wg_buckets:
        _allocate_ct_buckets_round_robin(
            wg_buckets,
            remaining,
            ct_candidates,
            cfg,
            phase_length_days,
            noise,
            single_ship=fl_ct_shortfall > 0,
        )

def allocate_sim_rap(
    pilots: List[Pilot],
    cfg: SquadronConfig,
    phase_length_days: float,
    noise: float = 0.0,
) -> None:
    """
    Top up sim RAP events after syllabus + continuation training.

    Sims are allocated one at a time (not bulk-assigned), subject to the per-pilot
    monthly event cap and optionally sim-wing capacity.
    """
    phase_months = float(phase_length_days) / PHASE_DAYS_PER_NOTIONAL_MONTH
    if math.isfinite(cfg.sim_sessions_monthly):
        sim_capacity = int(cfg.sim_sessions_monthly * cfg.sim_bays_per_session * phase_months)
        sim_capacity = max(0, sim_capacity - int(cfg.deferred_sim_burden))
    else:
        sim_capacity = None  # no sim-bay limit (e.g. SIM_SESSIONS_MONTHLY = inf)
    used_sims = int(round(sum(p.sim_phase for p in pilots)))

    while sim_capacity is None or used_sims < sim_capacity:
        pool = []
        for p in pilots:
            target = p.target_sims * phase_months
            shortfall = target - p.sim_phase
            if shortfall <= 1e-9:
                continue
            if not p.has_events_capacity(phase_length_days):
                continue
            pool.append((shortfall, p))
        if not pool:
            break

        max_shortfall = max(s for s, _ in pool)
        candidates = [p for s, p in pool if s >= max_shortfall - 1e-9]
        if not assign_sim(cfg, candidates, phase_length_days, noise):
            break
        used_sims += 1

# ----------------------
# Main Simulation Phase
# ----------------------
def _print_allocation_debug(pilots: List[Pilot], stage: str) -> None:
    """One line per pilot; fixed-width columns for narrow terminals."""
    print(f"--- {stage} ---")
    print(f"{'Q/U':<9}{'bl':>4}{'rd':>4}{'so':>4}{'sm':>4}{'tt':>4}")
    for p in pilots:
        qu = f"{p.qual.name}/{p.upgrade.name}"
        tot = p.sortie_phase + p.sim_phase
        print(
            f"{qu:<9}"
            f"{p.sortie_blue_phase:>4.0f}{p.sortie_red_phase:>4.0f}"
            f"{p.sortie_phase:>4.0f}{p.sim_phase:>4.0f}{tot:>4.0f}"
        )


def run_phase_simulation(
    cfg: SquadronConfig,
    pilots: List[Pilot],
    debug_verbose: bool = False,
    pre_seed_upgrades: bool = False,
    sim_config: Optional[SimulationConfig] = None,
    mqt_syllabus: Optional[List[SyllabusEvent]] = None,
    flug_syllabus: Optional[List[SyllabusEvent]] = None,
    ipug_syllabus: Optional[List[SyllabusEvent]] = None,
    continuation_profile: Optional[ContinuationProfile] = None,
):
    sim = sim_config or SimulationConfig()
    phase_length_days = float(sim.phase_length_days)
    phase_months = sim.phase_length_months
    noise = sim.allocation_noise
    mqt_syllabus = mqt_syllabus or MQT_SYLLABUS
    flug_syllabus = flug_syllabus or FLUG_SYLLABUS
    ipug_syllabus = ipug_syllabus or IPUG_SYLLABUS
    continuation_profile = continuation_profile or CONTINUATION_PROFILE

    # Pilots with open syllabus lines at phase start retry those in step 3 only (not step 4).
    carryover_ids = {id(p) for p in pilots if p.incomplete_syllabus_items}

    for p in pilots:
        p.reset_phase_counters()
        p.set_rap_requirement()

    if not pre_seed_upgrades:
        syllabus_mqt = select_upgrade_students(pilots, Upgrade.MQT, cfg.mqt_students)
        syllabus_flug = select_upgrade_students(pilots, Upgrade.FLUG, cfg.flug_students)
        syllabus_ipug = select_upgrade_students(pilots, Upgrade.IPUG, cfg.ipug_students)
    else:
        syllabus_mqt = [
            p for p in pilots
            if p.upgrade == Upgrade.MQT and id(p) not in carryover_ids
        ]
        syllabus_flug = [
            p for p in pilots
            if p.upgrade == Upgrade.FLUG and id(p) not in carryover_ids
        ]
        syllabus_ipug = [
            p for p in pilots
            if p.upgrade == Upgrade.IPUG and id(p) not in carryover_ids
        ]

    total_iron = max(
        0,
        int(total_phase_capacity(cfg) * phase_months) - cfg.deferred_sortie_burden,
    )
    upgrade_capacity, ct_sortie_cap = sim.phase_sortie_budgets(total_iron)

    # 3. Carryover: incomplete lines only (not a full new syllabus)
    for upgrade_type in (Upgrade.MQT, Upgrade.FLUG, Upgrade.IPUG):
        for student in [p for p in pilots if p.upgrade == upgrade_type and p.incomplete_syllabus_items]:
            for event in list(student.incomplete_syllabus_items):
                process_syllabus_event(
                    event, [student], pilots, upgrade_type, noise,
                    cfg, phase_length_days, upgrade_capacity,
                )

    # 4. Full syllabus for new students only (carryover excluded above)
    run_upgrade_program(mqt_syllabus, syllabus_mqt, pilots, Upgrade.MQT, noise, cfg, phase_length_days, upgrade_capacity)
    if debug_verbose:
        _print_allocation_debug(pilots, "MQT")
    run_upgrade_program(ipug_syllabus, syllabus_ipug, pilots, Upgrade.IPUG, noise, cfg, phase_length_days, upgrade_capacity)
    if debug_verbose:
        _print_allocation_debug(pilots, "IPUG")
    run_upgrade_program(flug_syllabus, syllabus_flug, pilots, Upgrade.FLUG, noise, cfg, phase_length_days, upgrade_capacity)
    if debug_verbose:
        _print_allocation_debug(pilots, "FLUG")
    # 5. Continuation Training
    allocate_continuation_training(
        pilots, continuation_profile, total_iron, noise, cfg, phase_length_days,
        ct_sortie_cap=ct_sortie_cap,
    )
    if debug_verbose:
        _print_allocation_debug(pilots, "CT")

    # 6. Sim RAP (discrete allocation; syllabus sims already credited above)
    allocate_sim_rap(pilots, cfg, phase_length_days, noise)
    if debug_verbose:
        _print_allocation_debug(pilots, "SimRAP")
    graduate_completed_upgrades(pilots)

    metrics = phase_upgrade_metrics(
        pilots,
        mqt_syllabus=mqt_syllabus,
        flug_syllabus=flug_syllabus,
        ipug_syllabus=ipug_syllabus,
    )
    apply_deferred_burden_to_squadron(cfg, metrics)

    # 7. Finalize monthly stats and RAP shortfalls
    for p in pilots:
        p.update_total(phase_length_days)
        p.update_monthly(phase_length_days)

    return pilots

# ----------------------
# Reporting
# ----------------------
def print_phase_summary(pilots: List[Pilot], cfg: SquadronConfig, verbose: bool = True):
    print("\n=== Phase Summary ===")
    
    groups = {
        "MQT Students": [p for p in pilots if p.upgrade == Upgrade.MQT],
        "Incomplete MQT Students": [p for p in pilots if p.upgrade == Upgrade.MQT and p.incomplete_syllabus_items],
        "FLUG Students": [p for p in pilots if p.upgrade == Upgrade.FLUG],
        "Incomplete FLUG Students": [p for p in pilots if p.upgrade == Upgrade.FLUG and p.incomplete_syllabus_items],
        "IPUG Students": [p for p in pilots if p.upgrade == Upgrade.IPUG],
        "Incomplete IPUG Students": [p for p in pilots if p.upgrade == Upgrade.IPUG and p.incomplete_syllabus_items],
        "Line Wingmen": [p for p in pilots if p.qual == Qual.WG and p.upgrade == Upgrade.NONE],
        "Line FLs": [p for p in pilots if p.qual == Qual.FL and p.upgrade == Upgrade.NONE],
        "IPs": [p for p in pilots if p.qual == Qual.IP]
    }

    for name, group in groups.items():
        if not group:
            print(f"{name}: None")
            continue
            
        avg_sorties = sum(p.sortie_monthly for p in group) / len(group)
        avg_sims = sum(p.sim_monthly for p in group) / len(group)
        avg_events = sum(p.sortie_monthly + p.sim_monthly for p in group) / len(group)
        avg_blue_sorties = sum(p.sortie_blue_monthly for p in group) / len(group)
        avg_red_sorties = sum(p.sortie_red_monthly for p in group) / len(group)
        if verbose:
            print(
                f"{name} ({len(group)}): Avg Mo. Sorties {avg_sorties:.1f}, "
                f"Sims {avg_sims:.1f}, Total Events {avg_events:.1f}, "
                f"Blue {avg_blue_sorties:.1f}, Red {avg_red_sorties:.1f}"
            )

    if verbose:
        for name, group in groups.items():
            for p in group:
                print(
                    f"{p.qual}/{p.upgrade}: Sorties {p.sortie_monthly:.1f}/mo, "
                    f"Sims {p.sim_monthly:.1f}/mo, "
                    f"Total {p.sortie_monthly + p.sim_monthly:.1f}/mo"
                )