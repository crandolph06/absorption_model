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
)
from src.syllabi import SyllabusEvent, ContinuationProfile
from src import rules
from src.syllabi import MQT_SYLLABUS, FLUG_SYLLABUS, IPUG_SYLLABUS, CONTINUATION_PROFILE

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


def phase_upgrade_metrics(pilots: List[Pilot]) -> dict:
    """
    End-of-phase syllabus stats read from ``Pilot.incomplete_syllabus_items``
    and ``Pilot.upgrade`` — same state the simulator uses for carryover (lines 289–295).
    """
    deferred_mqt_sorties = deferred_flug_sorties = deferred_ipug_sorties = 0
    deferred_mqt_sims = deferred_flug_sims = deferred_ipug_sims = 0

    mqt_syllabus_sorties = sum(1 for e in MQT_SYLLABUS if e.event_type == EventType.SORTIE)
    mqt_syllabus_sims = sum(1 for e in MQT_SYLLABUS if e.event_type == EventType.SIM)
    flug_syllabus_sorties = sum(1 for e in FLUG_SYLLABUS if e.event_type == EventType.SORTIE)
    flug_syllabus_sims = sum(1 for e in FLUG_SYLLABUS if e.event_type == EventType.SIM)
    ipug_syllabus_sorties = sum(1 for e in IPUG_SYLLABUS if e.event_type == EventType.SORTIE)
    ipug_syllabus_sims = sum(1 for e in IPUG_SYLLABUS if e.event_type == EventType.SIM)

    for p in pilots:
        n_sorties = sum(1 for e in p.incomplete_syllabus_items if e.event_type == EventType.SORTIE)
        n_sims = sum(1 for e in p.incomplete_syllabus_items if e.event_type == EventType.SIM)

        if p.upgrade == Upgrade.MQT:
            deferred_mqt_sorties += n_sorties
            deferred_mqt_sims += n_sims
        elif p.upgrade == Upgrade.FLUG:
            deferred_flug_sorties += n_sorties
            deferred_flug_sims += n_sims
        elif p.upgrade == Upgrade.IPUG:
            deferred_ipug_sorties += n_sorties
            deferred_ipug_sims += n_sims

    remaining_mqt_syllabi = (deferred_mqt_sorties + deferred_mqt_sims) / (mqt_syllabus_sorties + mqt_syllabus_sims)
    remaining_mqt_syllabi_sorties_only = deferred_mqt_sorties / mqt_syllabus_sorties
    remaining_flug_syllabi = (deferred_flug_sorties + deferred_flug_sims) / (flug_syllabus_sorties + flug_syllabus_sims)
    remaining_flug_syllabi_sorties_only = deferred_flug_sorties / flug_syllabus_sorties
    remaining_ipug_syllabi = (deferred_ipug_sorties + deferred_ipug_sims) / (ipug_syllabus_sorties + ipug_syllabus_sims)
    remaining_ipug_syllabi_sorties_only = deferred_ipug_sorties / ipug_syllabus_sorties

    return {
        "deferred_mqt_sorties": deferred_mqt_sorties,
        "deferred_flug_sorties": deferred_flug_sorties,
        "deferred_ipug_sorties": deferred_ipug_sorties,
        "deferred_mqt_sims": deferred_mqt_sims,
        "deferred_flug_sims": deferred_flug_sims,
        "deferred_ipug_sims": deferred_ipug_sims,
        "remaining_mqt_syllabi": remaining_mqt_syllabi,
        "remaining_flug_syllabi": remaining_flug_syllabi,
        "remaining_ipug_syllabi": remaining_ipug_syllabi,
        "remaining_mqt_syllabi_sorties_only": remaining_mqt_syllabi_sorties_only,
        "remaining_flug_syllabi_sorties_only": remaining_flug_syllabi_sorties_only,
        "remaining_ipug_syllabi_sorties_only": remaining_ipug_syllabi_sorties_only,
    }


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
def _eligible_for_event(candidates: List[Pilot], cfg: SquadronConfig) -> List[Pilot]:
    return [p for p in candidates if p.has_events_capacity(cfg.phase_length_days)]


def _total_phase_events(p: Pilot) -> float:
    return p.sortie_phase + p.sim_phase


def _type_specific_event_count(p: Pilot, event_type: EventType, side: str = "Blue") -> float:
    """Tie-breaker: least sims, blue sorties, or red sorties for the event being assigned."""
    if event_type == EventType.SIM:
        return p.sim_phase
    if side == "Red":
        return p.sortie_red_phase
    return p.sortie_blue_phase


def _allocation_sort_key(
    p: Pilot, event_type: EventType, side: str = "Blue", noise: float = 0.0
) -> tuple:
    """Primary: total phase events; tie-break: count for this event type (lowest first)."""
    return (
        _total_phase_events(p) + random.uniform(0, noise),
        _type_specific_event_count(p, event_type, side),
    )


def _can_assign_distinct_from_pool(pool: List[Pilot], count: int, cfg: SquadronConfig) -> bool:
    """Whether ``count`` distinct pilots in ``pool`` can each take one more event under the cap."""
    if count <= 0:
        return True
    months = cfg.phase_length_days / 30.0
    if months <= 0:
        return False
    max_phase_events = MAX_MONTHLY_EVENTS * months
    eligible = _eligible_for_event(pool, cfg)
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
        available.sort(key=lambda p: usage[id(p)])
        winner = available[0]
        picked.add(id(winner))
        usage[id(winner)] += 1
    return True


def assign_sortie(
    cfg: SquadronConfig,
    candidates: List[Pilot],
    side: str = "Blue",
    noise: float = 0.0,
    exclude: Optional[Set[int]] = None,
) -> bool:
    """
    Selects the best candidate (lowest total events, then least blue/red sorties) to fly a sortie.
    Returns True if a pilot was found and assigned, False otherwise.
    """
    exclude = exclude if exclude is not None else set()
    candidates = [p for p in candidates if id(p) not in exclude]
    candidates = _eligible_for_event(candidates, cfg)
    if not candidates:
        return False

    candidates.sort(key=lambda p: _allocation_sort_key(p, EventType.SORTIE, side, noise))

    winner = candidates[0]
    winner.add_sortie(avg_sortie_dur=cfg.avg_sortie_dur, side=side)
    exclude.add(id(winner))
    return True


def assign_sim(
    cfg: SquadronConfig,
    candidates: List[Pilot],
    noise: float = 0.0,
    exclude: Optional[Set[int]] = None,
) -> bool:
    """
    Selects the best candidate (lowest total events, then least sims) for a simulator event.
    Returns True if a pilot was found and assigned, False otherwise.
    """
    exclude = exclude if exclude is not None else set()
    candidates = [p for p in candidates if id(p) not in exclude]
    candidates = _eligible_for_event(candidates, cfg)
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
    student: Optional[Pilot] = None,
) -> bool:
    """Enough distinct support pilots, iron/sim capacity, and event-cap headroom for one student line."""
    ips = [
        p for p in all_pilots
        if rules.can_fill_seat(pilot=p, min_qual=Qual.IP) and p is not student
    ]
    if len(ips) < event.num_instructor:
        return False
    if not _can_assign_distinct_from_pool(ips, event.num_instructor, cfg):
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
    if not _can_assign_distinct_from_pool(wg_pool, wg_seats, cfg):
        return False
    if not _can_assign_distinct_from_pool(fl_pool, fl_seats, cfg):
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
    total_capacity: int,
):
    """
    Allocates sorties for a specific syllabus event.
    CRITICAL FIX: Support sorties are now generated PER student sortie.
    """
    for student in upgrade_students:
        for _ in range(event.num_student):
            if not student.has_events_capacity(cfg.phase_length_days):
                if event not in student.incomplete_syllabus_items:
                    student.incomplete_syllabus_items.append(event)
                continue
            if not check_syllabus_resources(
                event, all_pilots, syllabus_upgrade_type, total_capacity, cfg, student=student
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
                    if not assign_sim(cfg, ips, noise, exclude=line_assigned):
                        line_ok = False
                        break
                if line_ok:
                    for _ in range(event.num_blue_wg):
                        candidates = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.WG)]
                        if not assign_sim(cfg, candidates, noise, exclude=line_assigned):
                            line_ok = False
                            break
                if line_ok:
                    for _ in range(event.num_blue_fl):
                        candidates = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.FL)]
                        if not assign_sim(cfg, candidates, noise, exclude=line_assigned):
                            line_ok = False
                            break
                if line_ok:
                    for _ in range(event.num_red_wg):
                        candidates = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.WG)]
                        if not assign_sim(cfg, candidates, noise, exclude=line_assigned):
                            line_ok = False
                            break
                if line_ok:
                    for _ in range(event.num_red_fl):
                        candidates = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.FL)]
                        if not assign_sim(cfg, candidates, noise, exclude=line_assigned):
                            line_ok = False
                if line_ok:
                    student.add_sim(cfg.avg_sortie_dur)
            else:
                for _ in range(event.num_instructor):
                    ips = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.IP)]
                    if not assign_sortie(cfg, ips, "Blue", noise, exclude=line_assigned):
                        line_ok = False
                        break
                if line_ok:
                    for _ in range(event.num_blue_wg):
                        candidates = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.WG)]
                        if not assign_sortie(cfg, candidates, "Blue", noise, exclude=line_assigned):
                            line_ok = False
                            break
                if line_ok:
                    for _ in range(event.num_blue_fl):
                        candidates = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.FL)]
                        if not assign_sortie(cfg, candidates, "Blue", noise, exclude=line_assigned):
                            line_ok = False
                            break
                if line_ok:
                    for _ in range(event.num_red_wg):
                        candidates = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.WG)]
                        if not assign_sortie(cfg, candidates, "Red", noise, exclude=line_assigned):
                            line_ok = False
                            break
                if line_ok:
                    for _ in range(event.num_red_fl):
                        candidates = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.FL)]
                        if not assign_sortie(cfg, candidates, "Red", noise, exclude=line_assigned):
                            line_ok = False
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
    total_capacity: int,
):
    for event in syllabus:
        process_syllabus_event(event, students, all_pilots, upgrade_type, noise, cfg=cfg, total_capacity=total_capacity)

# ----------------------
# Continuation Training (CT)
# ----------------------
def allocate_continuation_training(
    pilots: List[Pilot],
    profile: ContinuationProfile,
    total_capacity: int,
    noise: float,
    cfg: SquadronConfig
):
    # Calculate how much capacity is left
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

    # Execute allocation per bucket
    for bucket, qty in base_qty.items():
        # Find eligible pilots for this specific CT bucket
        # We access the internal hierarchy check from rules since CT doesn't have a syllabus upgrade type
        eligible = [
            p for p in ct_candidates
            if rules.can_fill_seat(pilot=p, min_qual=bucket.min_qual)
        ]

        for _ in range(qty):
            if not assign_sortie(cfg=cfg, candidates=eligible, side=bucket.side, noise=noise):
                break

def allocate_sim_rap(
    pilots: List[Pilot],
    cfg: SquadronConfig,
    noise: float = 0.0,
) -> None:
    """
    Top up sim RAP events after syllabus + continuation training.

    Sims are allocated one at a time (not bulk-assigned), subject to the per-pilot
    monthly event cap and optionally sim-wing capacity.
    """
    phase_months = cfg.phase_length_days / 30.0
    if math.isfinite(cfg.sim_sessions_monthly):
        sim_capacity = int(cfg.sim_sessions_monthly * cfg.sim_bays_per_session * phase_months)
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
            if not p.has_events_capacity(cfg.phase_length_days):
                continue
            pool.append((shortfall, p))
        if not pool:
            break

        max_shortfall = max(s for s, _ in pool)
        candidates = [p for s, p in pool if s >= max_shortfall - 1e-9]
        if not assign_sim(cfg, candidates, noise):
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


def run_phase_simulation(cfg: SquadronConfig, pilots: List[Pilot], allocation_noise: float = 0.0, debug_verbose: bool = False, pre_seed_upgrades: bool = False):

    for p in pilots:
        p.reset_phase_counters()
        p.set_rap_requirement()

    # 2. New students only (carryover already have upgrade MQT/FLUG/IPUG)
    if not pre_seed_upgrades:
        mqt_students = select_upgrade_students(pilots, Upgrade.MQT, cfg.mqt_students)
        flug_students = select_upgrade_students(pilots, Upgrade.FLUG, cfg.flug_students)
        ipug_students = select_upgrade_students(pilots, Upgrade.IPUG, cfg.ipug_students)
    else:
        mqt_students = [p for p in pilots if p.upgrade == Upgrade.MQT]
        flug_students = [p for p in pilots if p.upgrade == Upgrade.FLUG]
        ipug_students = [p for p in pilots if p.upgrade == Upgrade.IPUG]

    phase_months = cfg.phase_length_days / 30.0
    total_capacity = int(total_phase_capacity(cfg) * phase_months)

    # 3. Carryover: incomplete lines only (not a full new syllabus)
    for upgrade_type in (Upgrade.MQT, Upgrade.FLUG, Upgrade.IPUG):
        for student in [p for p in pilots if p.upgrade == upgrade_type and p.incomplete_syllabus_items]:
            for event in list(student.incomplete_syllabus_items):
                process_syllabus_event(
                    event, [student], pilots, upgrade_type, allocation_noise, cfg, total_capacity
                )

    # 4. Full syllabus for new students only
    run_upgrade_program(MQT_SYLLABUS, mqt_students, pilots, Upgrade.MQT, allocation_noise, cfg=cfg, total_capacity=total_capacity)
    if debug_verbose:
        _print_allocation_debug(pilots, "MQT")
    run_upgrade_program(IPUG_SYLLABUS, ipug_students, pilots, Upgrade.IPUG, allocation_noise, cfg=cfg, total_capacity=total_capacity)
    if debug_verbose:
        _print_allocation_debug(pilots, "IPUG")
    run_upgrade_program(FLUG_SYLLABUS, flug_students, pilots, Upgrade.FLUG, allocation_noise, cfg=cfg, total_capacity=total_capacity)
    if debug_verbose:
        _print_allocation_debug(pilots, "FLUG")
    # 5. Continuation Training
    allocate_continuation_training(pilots, CONTINUATION_PROFILE, total_capacity, allocation_noise, cfg=cfg)
    if debug_verbose:
        _print_allocation_debug(pilots, "CT")

    # 6. Sim RAP (discrete allocation; syllabus sims already credited above)
    allocate_sim_rap(pilots, cfg, allocation_noise)
    if debug_verbose:
        _print_allocation_debug(pilots, "SimRAP")
    # 7. Finalize monthly stats and RAP shortfalls
    for p in pilots:
        p.update_total(cfg.phase_length_days)
        p.update_monthly(cfg.phase_length_days)

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