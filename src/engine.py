import heapq
import itertools
import math
import random
from typing import List, Literal, Optional, Set

SortieAllocationMode = Literal["rap_priority", "equity"]
from src.models import (
    Assignment,
    AssignedUTCRank,
    EventType,
    SquadronConfig,
    Pilot,
    Qual,
    Upgrade,
    MAX_MONTHLY_EVENTS,
    PHASE_DAYS_PER_NOTIONAL_MONTH,
    UTC_ALLOCATION_ORDER,
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


def assess_pipeline_self_termination(
    pilots: List[Pilot],
    metrics: dict,
    phase_length_days: float,
    cfg: SquadronConfig,
    phase_gross_iron: int,
    single_ship_monthly_cap: float,
) -> dict:
    """
    Phase pipeline diagnostics and iron-based self-termination.

    ``pipeline_deferred_due_to_ip`` tracks upgrade deferrals attributed to required IP
    seats (leading indicator only). ``self_terminating_phase`` is True when squadron
    iron is left unallocated because no supervised or capped single-ship CT sortie can
    still be assigned.
    """
    months = float(phase_length_days) / PHASE_DAYS_PER_NOTIONAL_MONTH
    deferred_slots = (
        metrics["deferred_mqt_sorties"]
        + metrics["deferred_flug_sorties"]
        + metrics["deferred_ipug_sorties"]
        + metrics["deferred_mqt_sims"]
        + metrics["deferred_flug_sims"]
        + metrics["deferred_ipug_sims"]
    )
    held_back = (
        metrics["held_back_mqt"]
        + metrics["held_back_flug"]
        + metrics["held_back_ipug"]
    )
    has_upgrade_deferral = deferred_slots > 0 or held_back > 0

    ips = [
        p for p in pilots
        if p.active and p.qual == Qual.IP and p.current_assignment == Assignment.LINE
    ]
    ip_available = [p for p in ips if p.has_events_capacity(phase_length_days)]
    ip_at_cap = [p for p in ips if not p.has_events_capacity(phase_length_days)]

    max_ip_events_mo = 0.0
    if ips and months > 0:
        max_ip_events_mo = max(p.phase_events() / months for p in ips)

    deferred_due_to_ip = bool(has_upgrade_deferral and cfg.deferral_due_to_ip)
    sorties_flown = int(round(sum(p.sortie_phase for p in pilots)))
    unallocated_iron = max(0, int(phase_gross_iron) - sorties_flown)
    can_assign_more_ct = _can_assign_more_ct_sorties(
        pilots, phase_length_days, single_ship_monthly_cap,
    )
    self_terminating_phase = unallocated_iron > 0 and not can_assign_more_ct

    return {
        "self_terminating_phase": self_terminating_phase,
        "deferred_due_to_ip": deferred_due_to_ip,
        "has_upgrade_deferral": has_upgrade_deferral,
        "unallocated_iron": unallocated_iron,
        "can_assign_more_ct": can_assign_more_ct,
        "ip_count": len(ips),
        "ip_at_cap_count": len(ip_at_cap),
        "ip_available_count": len(ip_available),
        "max_ip_events_monthly": max_ip_events_mo,
    }


def _can_assign_more_ct_sorties(
    pilots: List[Pilot],
    phase_length_days: float,
    single_ship_monthly_cap: float,
) -> bool:
    """Whether another CT sortie could still be assigned under cap and seat rules."""
    ct_candidates = [p for p in pilots if p.upgrade != Upgrade.MQT]

    supervisors = [
        p for p in ct_candidates
        if rules.can_fill_seat(pilot=p, min_qual=Qual.FL)
        and p.has_events_capacity(phase_length_days)
    ]
    if supervisors:
        return True

    single_ship_wg = [
        p for p in ct_candidates
        if rules.can_fill_seat(pilot=p, min_qual=Qual.WG)
        and p.has_events_capacity(phase_length_days)
        and p.has_single_ship_allocation_capacity(phase_length_days, single_ship_monthly_cap)
    ]
    return len(single_ship_wg) > 0


def apply_pipeline_status_to_squadron(cfg: SquadronConfig, status: dict) -> None:
    """Latch phase pipeline status on ``cfg`` for history, sweeps, and manning."""
    cfg.self_terminating_phase = bool(status["self_terminating_phase"])
    cfg.unallocated_iron = int(status["unallocated_iron"])
    cfg.pipeline_deferred_due_to_ip = bool(status["deferred_due_to_ip"])
    cfg.pipeline_ip_at_cap_count = int(status["ip_at_cap_count"])
    cfg.pipeline_ip_available_count = int(status["ip_available_count"])
    cfg.pipeline_max_ip_events_monthly = float(status["max_ip_events_monthly"])
    if cfg.self_terminating_phase:
        cfg.self_terminating_run = True


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
    max_phase_events = _phase_event_limit(phase_length_days)
    if max_phase_events is None:
        return []
    return [
        p for p in candidates
        if _has_event_capacity_under_limit(p, max_phase_events)
    ]


def _candidates_for_utc(candidates: List[Pilot], utc: AssignedUTCRank) -> List[Pilot]:
    return [p for p in candidates if p.assigned_utc == utc]


def _utc_sortie_pool(
    candidates: List[Pilot],
    utc: AssignedUTCRank,
    phase_length_days: float,
) -> List[Pilot]:
    """UTC-ordered sortie pool: pilots in ``utc`` who still need sortie RAP."""
    return [
        p for p in _candidates_for_utc(candidates, utc)
        if _pilot_needs_sortie_rap(p, phase_length_days)
    ]


def _utc_sim_pool(
    candidates: List[Pilot],
    utc: AssignedUTCRank,
    phase_length_days: float,
) -> List[Pilot]:
    """UTC-ordered sim pool: pilots in ``utc`` who still need sim RAP."""
    return [
        p for p in _candidates_for_utc(candidates, utc)
        if _pilot_needs_sim_rap(p, phase_length_days)
    ]


def _ip_support_sortie_pool(all_pilots: List[Pilot], phase_length_days: float) -> List[Pilot]:
    """IPs for syllabus support: prefer those still short of sortie RAP."""
    ips = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.IP)]
    needing = [p for p in ips if _pilot_needs_sortie_rap(p, phase_length_days)]
    return needing if needing else ips


def _ip_support_sim_pool(all_pilots: List[Pilot], phase_length_days: float) -> List[Pilot]:
    """IPs for syllabus sim support: prefer those still short of sim RAP."""
    ips = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.IP)]
    needing = [p for p in ips if _pilot_needs_sim_rap(p, phase_length_days)]
    return needing if needing else ips


def _pilot_needs_sortie_rap(p: Pilot, phase_length_days: float) -> bool:
    """True when the pilot still has sortie RAP shortfall this phase."""
    months = float(phase_length_days) / PHASE_DAYS_PER_NOTIONAL_MONTH
    if months <= 0 or p.target_sorties <= 0:
        return False
    expected = p.target_sorties * months
    return p.sortie_rap_credit(months) < expected - 1e-9


def _pilot_needs_sim_rap(p: Pilot, phase_length_days: float) -> bool:
    """True when the pilot still has sim RAP shortfall this phase."""
    months = float(phase_length_days) / PHASE_DAYS_PER_NOTIONAL_MONTH
    if months <= 0 or p.target_sims <= 0:
        return False
    expected = p.target_sims * months
    return p.sim_phase < expected - 1e-9


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


def _deterministic_blue_sortie_key(p: Pilot) -> tuple:
    return (p.sortie_phase + p.sim_phase, p.sortie_blue_phase, _QUAL_RANK[p.qual])


def _deterministic_red_sortie_key(p: Pilot) -> tuple:
    return (p.sortie_phase + p.sim_phase, p.sortie_red_phase, _QUAL_RANK[p.qual])


def _deterministic_sim_key(p: Pilot) -> tuple:
    return (p.sortie_phase + p.sim_phase, p.sim_phase, _QUAL_RANK[p.qual])


def _phase_event_limit(phase_length_days: float) -> float | None:
    months = float(phase_length_days) / PHASE_DAYS_PER_NOTIONAL_MONTH
    if months <= 0:
        return None
    return MAX_MONTHLY_EVENTS * months


def _has_event_capacity_under_limit(p: Pilot, max_phase_events: float) -> bool:
    return p.sortie_phase + p.sim_phase + 1.0 <= max_phase_events + 1e-9


def _ct_heap_entry(p: Pilot, side: str, order: int) -> tuple:
    if side == "Red":
        return (*_deterministic_red_sortie_key(p), order, p)
    return (*_deterministic_blue_sortie_key(p), order, p)


def _sim_rap_shortfall(p: Pilot, phase_months: float) -> float:
    return p.target_sims * phase_months - p.sim_phase


def _sim_rap_heap_entry(p: Pilot, phase_months: float, order: int) -> tuple:
    return (-_sim_rap_shortfall(p, phase_months), *_deterministic_sim_key(p), order, p)


def _assign_ct_sortie_from_heap(
    heap: list[tuple],
    *,
    cfg: SquadronConfig,
    side: str,
    order_by_id: dict[int, int],
    max_phase_events: float,
    phase_length_days: float,
    single_ship: bool = False,
    single_ship_monthly_cap: float = 1.0,
) -> bool:
    while heap:
        entry = heapq.heappop(heap)
        pilot = entry[-1]
        if not _has_event_capacity_under_limit(pilot, max_phase_events):
            continue
        if single_ship and not pilot.has_single_ship_allocation_capacity(
            phase_length_days,
            single_ship_monthly_cap,
        ):
            continue

        current = _ct_heap_entry(pilot, side, order_by_id[id(pilot)])
        if entry[:-1] != current[:-1]:
            heapq.heappush(heap, current)
            continue

        pilot.add_sortie(
            avg_sortie_dur=cfg.avg_sortie_dur,
            side=side,
            single_ship=single_ship,
        )
        if (
            _has_event_capacity_under_limit(pilot, max_phase_events)
            and (
                not single_ship
                or pilot.has_single_ship_allocation_capacity(
                    phase_length_days,
                    single_ship_monthly_cap,
                )
            )
        ):
            heapq.heappush(heap, _ct_heap_entry(pilot, side, order_by_id[id(pilot)]))
        return True
    return False


def _assign_sim_rap_from_heap(
    heap: list[tuple],
    *,
    cfg: SquadronConfig,
    phase_months: float,
    order_by_id: dict[int, int],
    max_phase_events: float,
) -> bool:
    while heap:
        entry = heapq.heappop(heap)
        pilot = entry[-1]
        if _sim_rap_shortfall(pilot, phase_months) <= 1e-9:
            continue
        if not _has_event_capacity_under_limit(pilot, max_phase_events):
            continue

        current = _sim_rap_heap_entry(pilot, phase_months, order_by_id[id(pilot)])
        if entry[:-1] != current[:-1]:
            heapq.heappush(heap, current)
            continue

        pilot.add_sim(cfg.avg_sortie_dur)
        if (
            _sim_rap_shortfall(pilot, phase_months) > 1e-9
            and _has_event_capacity_under_limit(pilot, max_phase_events)
        ):
            heapq.heappush(
                heap,
                _sim_rap_heap_entry(pilot, phase_months, order_by_id[id(pilot)]),
            )
        return True
    return False


def _can_assign_distinct_from_pool(pool: List[Pilot], count: int, phase_length_days: float) -> bool:
    """Whether ``count`` distinct pilots in ``pool`` can each take one more event under the cap."""
    if count <= 0:
        return True
    max_phase_events = _phase_event_limit(phase_length_days)
    if max_phase_events is None:
        return False
    eligible_count = 0
    for pilot in pool:
        if _has_event_capacity_under_limit(pilot, max_phase_events):
            eligible_count += 1
            if eligible_count >= count:
                return True
    return False

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
        cfg.deferral_due_to_ip = True
        return False
    if not _can_assign_distinct_from_pool(ips, event.num_instructor, phase_length_days):
        cfg.deferral_due_to_ip = True
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
# Event Assignment
# ----------------------

def _assign_sortie_equity(
    cfg: SquadronConfig,
    candidates: List[Pilot],
    phase_length_days: float,
    side: str,
    noise: float,
    exclude: Set[int],
    single_ship: bool,
    single_ship_monthly_cap: float,
) -> bool:
    """Lowest-load pilot within ``candidates``; no RAP or UTC filtering."""
    candidates = [p for p in candidates if id(p) not in exclude]
    candidates = _eligible_for_event(candidates, phase_length_days)
    if single_ship:
        candidates = [
            p for p in candidates
            if p.has_single_ship_allocation_capacity(phase_length_days, single_ship_monthly_cap)
        ]
    if not candidates:
        return False

    if noise == 0.0:
        key = _deterministic_red_sortie_key if side == "Red" else _deterministic_blue_sortie_key
        winner = min(candidates, key=key)
    else:
        winner = min(
            candidates,
            key=lambda p: _allocation_sort_key(p, EventType.SORTIE, side, noise),
        )
    winner.add_sortie(avg_sortie_dur=cfg.avg_sortie_dur, side=side, single_ship=single_ship)
    exclude.add(id(winner))
    return True


def assign_sortie_policy(
    cfg: SquadronConfig,
    candidates: List[Pilot],
    phase_length_days: float,
    side: str = "Blue",
    noise: float = 0.0,
    exclude: Optional[Set[int]] = None,
    single_ship: bool = False,
    single_ship_monthly_cap: float = 1.0,
    *,
    mode: SortieAllocationMode = "equity",
    utc_wise: bool = False,
) -> bool:
    """
    Assign one sortie using either UTC-ordered RAP priority or squad-wide equity.

    ``rap_priority``: UTC 1 → 2 → 3 → unassigned; within each UTC only pilots
    still short of sortie RAP. No hard cap — pilots drop out once RAP is met.
    """
    exclude = exclude if exclude is not None else set()
    if mode == "rap_priority" and utc_wise:
        for utc in UTC_ALLOCATION_ORDER:
            pool = _utc_sortie_pool(candidates, utc, phase_length_days)
            if _assign_sortie_equity(
                cfg, pool, phase_length_days, side, noise, exclude,
                single_ship, single_ship_monthly_cap,
            ):
                return True
        return False
    return _assign_sortie_equity(
        cfg, candidates, phase_length_days, side, noise, exclude,
        single_ship, single_ship_monthly_cap,
    )


def assign_sortie(
    cfg: SquadronConfig,
    candidates: List[Pilot],
    phase_length_days: float,
    side: str = "Blue",
    noise: float = 0.0,
    exclude: Optional[Set[int]] = None,
    single_ship: bool = False,
    single_ship_monthly_cap: float = 1.0,
    utc_wise_allocation: bool = False,
    utc_rap_priority: bool = False,
) -> bool:
    """Backward-compatible wrapper around ``assign_sortie_policy``."""
    mode: SortieAllocationMode = (
        "rap_priority" if (utc_rap_priority or utc_wise_allocation) else "equity"
    )
    return assign_sortie_policy(
        cfg,
        candidates,
        phase_length_days,
        side,
        noise,
        exclude=exclude,
        single_ship=single_ship,
        single_ship_monthly_cap=single_ship_monthly_cap,
        mode=mode,
        utc_wise=utc_wise_allocation or utc_rap_priority,
    )


def _assign_sim_equity(
    cfg: SquadronConfig,
    candidates: List[Pilot],
    phase_length_days: float,
    noise: float,
    exclude: Set[int],
) -> bool:
    candidates = [p for p in candidates if id(p) not in exclude]
    candidates = _eligible_for_event(candidates, phase_length_days)
    if not candidates:
        return False

    if noise == 0.0:
        winner = min(candidates, key=_deterministic_sim_key)
    else:
        winner = min(
            candidates,
            key=lambda p: _allocation_sort_key(p, EventType.SIM, noise=noise),
        )
    winner.add_sim(cfg.avg_sortie_dur)
    exclude.add(id(winner))
    return True


def assign_sim_policy(
    cfg: SquadronConfig,
    candidates: List[Pilot],
    phase_length_days: float,
    noise: float = 0.0,
    exclude: Optional[Set[int]] = None,
    *,
    mode: SortieAllocationMode = "equity",
    utc_wise: bool = False,
) -> bool:
    """Assign one sim using UTC-ordered RAP priority or squad-wide equity."""
    exclude = exclude if exclude is not None else set()
    if mode == "rap_priority" and utc_wise:
        for utc in UTC_ALLOCATION_ORDER:
            pool = _utc_sim_pool(candidates, utc, phase_length_days)
            if _assign_sim_equity(cfg, pool, phase_length_days, noise, exclude):
                return True
        return False
    return _assign_sim_equity(cfg, candidates, phase_length_days, noise, exclude)


def assign_sim(
    cfg: SquadronConfig,
    candidates: List[Pilot],
    phase_length_days: float,
    noise: float = 0.0,
    exclude: Optional[Set[int]] = None,
    utc_wise_allocation: bool = False,
    utc_rap_priority: bool = False,
) -> bool:
    """Backward-compatible wrapper around ``assign_sim_policy``."""
    mode: SortieAllocationMode = (
        "rap_priority" if (utc_rap_priority or utc_wise_allocation) else "equity"
    )
    return assign_sim_policy(
        cfg,
        candidates,
        phase_length_days,
        noise,
        exclude=exclude,
        mode=mode,
        utc_wise=utc_wise_allocation or utc_rap_priority,
    )


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
    utc_wise_allocation: bool = False,
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
                    ips = _ip_support_sim_pool(all_pilots, phase_length_days)
                    if not assign_sim(cfg, ips, phase_length_days, noise, exclude=line_assigned):
                        line_ok = False
                        break
                if line_ok:
                    for _ in range(event.num_blue_fl):
                        candidates = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.FL)]
                        if not assign_sim_policy(
                            cfg, candidates, phase_length_days, noise, exclude=line_assigned,
                            mode="rap_priority", utc_wise=utc_wise_allocation,
                        ):
                            line_ok = False
                            break
                if line_ok:
                    for _ in range(event.num_red_fl):
                        candidates = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.FL)]
                        if not assign_sim_policy(
                            cfg, candidates, phase_length_days, noise, exclude=line_assigned,
                            mode="rap_priority", utc_wise=utc_wise_allocation,
                        ):
                            line_ok = False
                if line_ok:
                    for _ in range(event.num_blue_wg):
                        candidates = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.WG)]
                        if not assign_sim_policy(
                            cfg, candidates, phase_length_days, noise, exclude=line_assigned,
                            mode="rap_priority", utc_wise=utc_wise_allocation,
                        ):
                            line_ok = False
                            break
                if line_ok:
                    for _ in range(event.num_red_wg):
                        candidates = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.WG)]
                        if not assign_sim_policy(
                            cfg, candidates, phase_length_days, noise, exclude=line_assigned,
                            mode="rap_priority", utc_wise=utc_wise_allocation,
                        ):
                            line_ok = False
                            break
                if line_ok:
                    student.add_sim(cfg.avg_sortie_dur)
            else:
                for _ in range(event.num_instructor):
                    ips = _ip_support_sortie_pool(all_pilots, phase_length_days)
                    if not assign_sortie(cfg, ips, phase_length_days, "Blue", noise, exclude=line_assigned):
                        line_ok = False
                        break
                if line_ok:
                    for _ in range(event.num_blue_fl):
                        candidates = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.FL)]
                        if not assign_sortie_policy(
                            cfg, candidates, phase_length_days, "Blue", noise, exclude=line_assigned,
                            mode="rap_priority", utc_wise=utc_wise_allocation,
                        ):
                            line_ok = False
                            break
                if line_ok:
                    for _ in range(event.num_red_fl):
                        candidates = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.FL)]
                        if not assign_sortie_policy(
                            cfg, candidates, phase_length_days, "Red", noise, exclude=line_assigned,
                            mode="rap_priority", utc_wise=utc_wise_allocation,
                        ):
                            line_ok = False
                if line_ok:
                    for _ in range(event.num_blue_wg):
                        candidates = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.WG)]
                        if not assign_sortie_policy(
                            cfg, candidates, phase_length_days, "Blue", noise, exclude=line_assigned,
                            mode="rap_priority", utc_wise=utc_wise_allocation,
                        ):
                            line_ok = False
                            break
                if line_ok:
                    for _ in range(event.num_red_wg):
                        candidates = [p for p in all_pilots if rules.can_fill_seat(pilot=p, min_qual=Qual.WG)]
                        if not assign_sortie_policy(
                            cfg, candidates, phase_length_days, "Red", noise, exclude=line_assigned,
                            mode="rap_priority", utc_wise=utc_wise_allocation,
                        ):
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
    utc_wise_allocation: bool = False,
):
    for event in syllabus:
        process_syllabus_event(
            event, students, all_pilots, upgrade_type, noise,
            cfg=cfg, phase_length_days=phase_length_days, total_capacity=total_capacity,
            utc_wise_allocation=utc_wise_allocation,
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
    single_ship_monthly_cap: float = 1.0,
    sortie_mode: SortieAllocationMode = "equity",
    utc_wise: bool = False,
) -> int:
    """Assign CT sorties round-robin across ``buckets``; return count assigned."""
    if not buckets:
        return 0

    if sortie_mode == "rap_priority" and utc_wise:
        assigned = 0
        remaining_total = sum(remaining.get(b, 0) for b in buckets)
        while remaining_total > 0:
            assigned_this_pass = False
            for bucket in buckets:
                if remaining.get(bucket, 0) <= 0:
                    continue
                eligible = [
                    p for p in ct_candidates
                    if rules.can_fill_seat(pilot=p, min_qual=bucket.min_qual)
                ]
                assigned_sortie = assign_sortie_policy(
                    cfg,
                    eligible,
                    phase_length_days,
                    bucket.side,
                    noise,
                    single_ship=single_ship,
                    single_ship_monthly_cap=single_ship_monthly_cap,
                    mode="rap_priority",
                    utc_wise=True,
                )
                if assigned_sortie:
                    remaining[bucket] -= 1
                    remaining_total -= 1
                    assigned += 1
                    assigned_this_pass = True
                else:
                    remaining_total -= remaining.get(bucket, 0)
                    remaining[bucket] = 0
            if not assigned_this_pass:
                break
        return assigned

    candidate_pools = {
        min_qual: [
            p for p in ct_candidates
            if rules.can_fill_seat(pilot=p, min_qual=min_qual)
        ]
        for min_qual in {bucket.min_qual for bucket in buckets}
    }
    deterministic_heaps: dict[ContinuationBucket, list[tuple]] = {}
    order_by_id = {id(p): index for index, p in enumerate(ct_candidates)}
    max_phase_events = _phase_event_limit(phase_length_days)
    if noise == 0.0 and max_phase_events is not None:
        for bucket in buckets:
            heap = [
                _ct_heap_entry(p, bucket.side, order_by_id[id(p)])
                for p in candidate_pools[bucket.min_qual]
                if _has_event_capacity_under_limit(p, max_phase_events)
            ]
            heapq.heapify(heap)
            deterministic_heaps[bucket] = heap

    assigned = 0
    remaining_total = sum(remaining.get(b, 0) for b in buckets)
    while remaining_total > 0:
        assigned_this_pass = False
        for bucket in buckets:
            if remaining.get(bucket, 0) <= 0:
                continue
            if bucket in deterministic_heaps:
                assigned_sortie = _assign_ct_sortie_from_heap(
                    deterministic_heaps[bucket],
                    cfg=cfg,
                    side=bucket.side,
                    order_by_id=order_by_id,
                    max_phase_events=max_phase_events,
                    phase_length_days=phase_length_days,
                    single_ship=single_ship,
                    single_ship_monthly_cap=single_ship_monthly_cap,
                )
            else:
                eligible = candidate_pools[bucket.min_qual]
                assigned_sortie = assign_sortie_policy(
                    cfg,
                    eligible,
                    phase_length_days,
                    bucket.side,
                    noise,
                    single_ship=single_ship,
                    single_ship_monthly_cap=single_ship_monthly_cap,
                    mode="equity",
                )
            if assigned_sortie:
                remaining[bucket] -= 1
                remaining_total -= 1
                assigned += 1
                assigned_this_pass = True
            else:
                remaining_total -= remaining.get(bucket, 0)
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
    single_ship_monthly_cap: float = 1.0,
    utc_wise_allocation: bool = False,
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

    if utc_wise_allocation:
        fl_assigned = _allocate_ct_buckets_round_robin(
            fl_buckets,
            remaining,
            ct_candidates,
            cfg,
            phase_length_days,
            noise,
            single_ship_monthly_cap=single_ship_monthly_cap,
            sortie_mode="rap_priority",
            utc_wise=True,
        )
        fl_ct_shortfall = fl_planned - fl_assigned
        if wg_buckets:
            _allocate_ct_buckets_round_robin(
                wg_buckets,
                remaining,
                ct_candidates,
                cfg,
                phase_length_days,
                noise,
                single_ship=False,
                single_ship_monthly_cap=single_ship_monthly_cap,
                sortie_mode="rap_priority",
                utc_wise=True,
            )
        return

    fl_assigned = _allocate_ct_buckets_round_robin(
        fl_buckets, remaining, ct_candidates, cfg, phase_length_days, noise,
        single_ship_monthly_cap=single_ship_monthly_cap,
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
            single_ship_monthly_cap=single_ship_monthly_cap,
        )


def _sorties_used(pilots: List[Pilot]) -> int:
    return int(round(sum(p.sortie_phase for p in pilots)))


def allocate_leftover_sorties(
    pilots: List[Pilot],
    cfg: SquadronConfig,
    total_capacity: int,
    phase_length_days: float,
    noise: float = 0.0,
) -> int:
    """
    Equity pass on remaining sortie iron after syllabus and CT.

    No RAP filter — lowest-load line pilots receive extras until iron or
    per-pilot event capacity is exhausted.
    """
    ct_candidates = [p for p in pilots if p.upgrade != Upgrade.MQT]
    if not ct_candidates:
        return 0

    assigned = 0
    sides = itertools.cycle(["Blue", "Red"])
    while _sorties_used(pilots) < total_capacity:
        side = next(sides)
        if assign_sortie_policy(
            cfg, ct_candidates, phase_length_days, side, noise, mode="equity",
        ):
            assigned += 1
            continue
        other = "Red" if side == "Blue" else "Blue"
        if assign_sortie_policy(
            cfg, ct_candidates, phase_length_days, other, noise, mode="equity",
        ):
            assigned += 1
            continue
        break
    return assigned


def allocate_sim_rap(
    pilots: List[Pilot],
    cfg: SquadronConfig,
    phase_length_days: float,
    noise: float = 0.0,
    utc_wise_allocation: bool = False,
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

    max_phase_events = _phase_event_limit(phase_length_days)
    pilot_groups = (
        [_utc_sim_pool(pilots, utc, phase_length_days) for utc in UTC_ALLOCATION_ORDER]
        if utc_wise_allocation
        else [pilots]
    )

    if noise == 0.0 and max_phase_events is not None:
        order_by_id = {id(p): index for index, p in enumerate(pilots)}
        for group in pilot_groups:
            if not group:
                continue
            heap = [
                _sim_rap_heap_entry(p, phase_months, order_by_id[id(p)])
                for p in group
                if _sim_rap_shortfall(p, phase_months) > 1e-9
                and _has_event_capacity_under_limit(p, max_phase_events)
            ]
            heapq.heapify(heap)
            while (sim_capacity is None or used_sims < sim_capacity) and heap:
                if not _assign_sim_rap_from_heap(
                    heap,
                    cfg=cfg,
                    phase_months=phase_months,
                    order_by_id=order_by_id,
                    max_phase_events=max_phase_events,
                ):
                    break
                used_sims += 1
        return

    for group in pilot_groups:
        if not group:
            continue
        while sim_capacity is None or used_sims < sim_capacity:
            pool = []
            for p in group:
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
    auto_graduate: bool = True,
):
    sim = sim_config or SimulationConfig()
    phase_length_days = float(sim.phase_length_days)
    phase_months = sim.phase_length_months
    noise = sim.allocation_noise
    mqt_syllabus = mqt_syllabus or MQT_SYLLABUS
    flug_syllabus = flug_syllabus or FLUG_SYLLABUS
    ipug_syllabus = ipug_syllabus or IPUG_SYLLABUS
    continuation_profile = continuation_profile or CONTINUATION_PROFILE
    cfg.deferral_due_to_ip = False
    utc_wise = sim.utc_wise_allocation

    cfg.pilots = pilots

    # Pilots with open syllabus lines at phase start retry those in step 3 only (not step 4).
    carryover_ids = {id(p) for p in pilots if p.incomplete_syllabus_items}

    for p in pilots:
        p.reset_phase_counters()

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

    # Rank UTC slots on line pilots only (``upgrade == NONE``), after students are tagged.
    if utc_wise:
        cfg.update_rap_scenarios()
    for p in pilots:
        p.set_rap_requirement()

    total_iron = max(
        0,
        int(total_phase_capacity(cfg) * phase_months) - cfg.deferred_sortie_burden,
    )
    phase_gross_iron = total_iron
    upgrade_capacity, ct_sortie_cap = sim.phase_sortie_budgets(total_iron)

    # 3. Carryover: incomplete lines only (not a full new syllabus)
    for upgrade_type in (Upgrade.MQT, Upgrade.FLUG, Upgrade.IPUG):
        for student in [p for p in pilots if p.upgrade == upgrade_type and p.incomplete_syllabus_items]:
            for event in list(student.incomplete_syllabus_items):
                process_syllabus_event(
                    event, [student], pilots, upgrade_type, noise,
                    cfg, phase_length_days, upgrade_capacity,
                    utc_wise_allocation=utc_wise,
                )

    # 4. Full syllabus for new students only (carryover excluded above)
    run_upgrade_program(mqt_syllabus, syllabus_mqt, pilots, Upgrade.MQT, noise, cfg, phase_length_days, upgrade_capacity, utc_wise_allocation=utc_wise)
    if debug_verbose:
        _print_allocation_debug(pilots, "MQT")
    run_upgrade_program(flug_syllabus, syllabus_flug, pilots, Upgrade.FLUG, noise, cfg, phase_length_days, upgrade_capacity, utc_wise_allocation=utc_wise)
    if debug_verbose:
        _print_allocation_debug(pilots, "FLUG")
    run_upgrade_program(ipug_syllabus, syllabus_ipug, pilots, Upgrade.IPUG, noise, cfg, phase_length_days, upgrade_capacity, utc_wise_allocation=utc_wise)
    if debug_verbose:
        _print_allocation_debug(pilots, "IPUG")
    # 5. Continuation Training
    allocate_continuation_training(
        pilots, continuation_profile, total_iron, noise, cfg, phase_length_days,
        ct_sortie_cap=ct_sortie_cap,
        single_ship_monthly_cap=sim.single_ship_monthly_cap,
        utc_wise_allocation=utc_wise,
    )
    if debug_verbose:
        _print_allocation_debug(pilots, "CT")

    if utc_wise:
        allocate_leftover_sorties(
            pilots, cfg, total_iron, phase_length_days, noise,
        )
        if debug_verbose:
            _print_allocation_debug(pilots, "Leftover")

    # 6. Sim RAP (discrete allocation; syllabus sims already credited above)
    allocate_sim_rap(pilots, cfg, phase_length_days, noise, utc_wise_allocation=utc_wise)
    if debug_verbose:
        _print_allocation_debug(pilots, "SimRAP")

    metrics = phase_upgrade_metrics(
        pilots,
        mqt_syllabus=mqt_syllabus,
        flug_syllabus=flug_syllabus,
        ipug_syllabus=ipug_syllabus,
    )
    apply_deferred_burden_to_squadron(cfg, metrics)
    status = assess_pipeline_self_termination(
        pilots,
        metrics,
        phase_length_days,
        cfg,
        phase_gross_iron,
        sim.single_ship_monthly_cap,
    )
    apply_pipeline_status_to_squadron(cfg, status)

    # Finalize monthly stats before reporting snapshot (optionally before graduation).
    for p in pilots:
        p.update_total(phase_length_days)
        p.update_monthly(phase_length_days)

    if auto_graduate:
        graduate_completed_upgrades(pilots)

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
