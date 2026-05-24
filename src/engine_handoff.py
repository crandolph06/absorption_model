"""Build phase-end carryover into ``SquadronConfig.pending_deferred_requirements``."""

from typing import Iterable, List

from src.models import DeferredSyllabusItem, Pilot, SquadronConfig, Upgrade
from src.syllabi import FLUG_SYLLABUS, IPUG_SYLLABUS, MQT_SYLLABUS, SyllabusEvent


def syllabus_for_upgrade(upgrade: Upgrade) -> List[SyllabusEvent]:
    if upgrade == Upgrade.MQT:
        return list(MQT_SYLLABUS)
    if upgrade == Upgrade.FLUG:
        return list(FLUG_SYLLABUS)
    if upgrade == Upgrade.IPUG:
        return list(IPUG_SYLLABUS)
    return []


def remaining_syllabus_items(
    student: Pilot,
    upgrade: Upgrade,
    syllabus: List[SyllabusEvent],
) -> List[DeferredSyllabusItem]:
    """All syllabus rows this student has not yet completed, in curriculum order."""
    remaining: List[DeferredSyllabusItem] = []
    for syllabus_event_index, event in enumerate(syllabus):
        for student_event_repetition in range(event.num_student):
            if (syllabus_event_index, student_event_repetition) in student.upgrade_syllabus_done:
                continue
            remaining.append(
                DeferredSyllabusItem(
                    upgrade=upgrade,
                    event_name=event.name,
                    event_type=event.event_type,
                    syllabus_event_index=syllabus_event_index,
                    student_event_repetition=student_event_repetition,
                    student_pilot_id=student.pilot_id,
                    student_year_group=student.year_group,
                    student_squadron_id=student.squadron_id,
                )
            )
    return remaining


def sort_carryover(items: Iterable[DeferredSyllabusItem]) -> List[DeferredSyllabusItem]:
    return sorted(
        items,
        key=lambda d: (
            d.upgrade.value,
            d.syllabus_event_index,
            d.student_pilot_id,
            d.student_event_repetition,
        ),
    )


def finalize_pending_carryover(
    cfg: SquadronConfig,
    mqt_students: List[Pilot],
    flug_students: List[Pilot],
    ipug_students: List[Pilot],
) -> None:
    """
    Write the full remaining syllabus tail for incomplete students into
    ``cfg.pending_deferred_requirements`` (single source of truth for next phase).
    """
    carryover: List[DeferredSyllabusItem] = []
    for upgrade, students in (
        (Upgrade.MQT, mqt_students),
        (Upgrade.FLUG, flug_students),
        (Upgrade.IPUG, ipug_students),
    ):
        syllabus = syllabus_for_upgrade(upgrade)
        for student in students:
            carryover.extend(remaining_syllabus_items(student, upgrade, syllabus))

    cfg.pending_deferred_requirements = sort_carryover(carryover)
