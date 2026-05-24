"""Syllabus execution, crew allocation, and deferred training requirements."""

import random
from typing import List, Optional, Set, Tuple

from src.models import SquadronConfig, Pilot, Qual, Upgrade, EventType, DeferredSyllabusItem, DeferralSignature
from src.syllabi import FLUG_SYLLABUS, IPUG_SYLLABUS, MQT_SYLLABUS, SyllabusEvent
from src.engine_sim import syllabus_sim_session_cost
from src import rules


def assign_sortie(
    candidates: List[Pilot], side: str = "Blue", noise: float = 0.0, avg_sortie_dur: float = 1.3
) -> bool:
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
    candidates[0].add_sortie(avg_sortie_dur, side)
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
        if p.pilot_id == item.student_pilot_id:
            return p
    return None


def _deferral_signature(
    upgrade: Upgrade,
    syllabus_event_index: int,
    student: Pilot,
    student_event_repetition: int,
) -> DeferralSignature:
    return (
        upgrade,
        syllabus_event_index,
        student.pilot_id,
        student_event_repetition,
    )


def process_pending_deferred_requirements(
    cfg: SquadronConfig,
    pilots: List[Pilot],
    allocation_noise: float,
    deferred_requirements: List[DeferredSyllabusItem],
    completed_keys: Set[DeferralSignature],
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
            d.student_pilot_id,
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
        if (item.syllabus_event_index, item.student_event_repetition) in student.upgrade_syllabus_done:
            completed_keys.add(sig)
            continue
        if event.event_type == EventType.SIM and sim_session_budget is not None:
            cost = syllabus_sim_session_cost(event, cfg)
            if sim_session_budget[0] + 1e-9 < cost:
                deferred_requirements.append(item)
                continue
        if not _can_fulfill_student_slot(
            event, student, pilots, item.upgrade, allocation_noise
        ):
            deferred_requirements.append(item)
            continue
        if event.event_type == EventType.SIM and sim_session_budget is not None:
            sim_session_budget[0] -= syllabus_sim_session_cost(event, cfg)
        _apply_student_training_event(
            cfg, event, student, pilots, item.upgrade, allocation_noise,
            item.syllabus_event_index, item.student_event_repetition,
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


def _mark_syllabus_event_complete(
    student: Pilot, syllabus_event_index: int, student_event_repetition: int
) -> None:
    student.upgrade_syllabus_done.add((syllabus_event_index, student_event_repetition))


def _apply_student_training_event(
    cfg: SquadronConfig,
    event: SyllabusEvent,
    student: Pilot,
    all_pilots: List[Pilot],
    syllabus_upgrade_type: Upgrade,
    noise: float,
    syllabus_event_index: int,
    student_event_repetition: int,
) -> None:
    """Crew and credit one syllabus event (sortie or sim)."""
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
        _mark_syllabus_event_complete(student, syllabus_event_index, student_event_repetition)
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
    _mark_syllabus_event_complete(student, syllabus_event_index, student_event_repetition)


def process_syllabus_event(
    cfg: SquadronConfig,
    event: SyllabusEvent,
    upgrade_students: List[Pilot],
    all_pilots: List[Pilot],
    syllabus_upgrade_type: Upgrade,
    noise: float,
    syllabus_event_index: int,
    deferred_requirements: Optional[List[DeferredSyllabusItem]] = None,
    completed_keys: Optional[Set[DeferralSignature]] = None,
    sim_session_budget: Optional[List[float]] = None,
):
    """
    Allocate one syllabus line (SORTIE or SIM). SIM rows consume simulator **session**
    budget from ``sim_session_budget`` (see ``syllabus_sim_session_cost``).
    """
    for student in upgrade_students:
        for student_event_repetition in range(event.num_student):
            if (syllabus_event_index, student_event_repetition) in student.upgrade_syllabus_done:
                continue
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
                cost = syllabus_sim_session_cost(event, cfg)
                if sim_session_budget[0] + 1e-9 < cost:
                    if deferred_requirements is not None:
                        deferred_requirements.append(
                            DeferredSyllabusItem(
                                upgrade=syllabus_upgrade_type,
                                event_name=event.name,
                                event_type=EventType.SIM,
                                syllabus_event_index=syllabus_event_index,
                                student_event_repetition=student_event_repetition,
                                student_pilot_id=student.pilot_id,
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
                            student_pilot_id=student.pilot_id,
                            student_year_group=student.year_group,
                            student_squadron_id=student.squadron_id,
                        )
                    )
                continue
            if event.event_type == EventType.SIM and sim_session_budget is not None:
                sim_session_budget[0] -= syllabus_sim_session_cost(event, cfg)
            _apply_student_training_event(
                cfg,
                event,
                student,
                all_pilots,
                syllabus_upgrade_type,
                noise,
                syllabus_event_index,
                student_event_repetition,
            )


def run_upgrade_program(
    cfg: SquadronConfig,
    syllabus: List[SyllabusEvent],
    students: List[Pilot],
    all_pilots: List[Pilot],
    upgrade_type: Upgrade,
    noise: float,
    deferred_requirements: Optional[List[DeferredSyllabusItem]] = None,
    completed_keys: Optional[Set[DeferralSignature]] = None,
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
