"""Parquet / sweeper metrics derived from ``pending_deferred_requirements``."""

from typing import List

from src.models import DeferredSyllabusItem, Upgrade


def _incomplete_students(items: List[DeferredSyllabusItem], upgrade: Upgrade) -> int:
    return len({d.student_pilot_id for d in items if d.upgrade == upgrade})

def _incomplete_mqt_students(items: List[DeferredSyllabusItem], upgrade: Upgrade) -> int:
    return len({d.student_pilot_id for d in items if d.upgrade == upgrade and d.event_type})


def handoff_iteration_metrics(pending: List[DeferredSyllabusItem]) -> dict:
    """Per-iteration averages inside a sweeper Monte Carlo loop."""
    incomplete_mqt = _incomplete_students(pending, Upgrade.MQT)
    incomplete_flug = _incomplete_students(pending, Upgrade.FLUG)
    incomplete_ipug = _incomplete_students(pending, Upgrade.IPUG)
    return {
        "incomplete_mqt_students": float(incomplete_mqt),
        "incomplete_flug_students": float(incomplete_flug),
        "incomplete_ipug_students": float(incomplete_ipug),
        "mqt_syllabus_complete": 1.0 if incomplete_mqt == 0 else 0.0,
        "flug_syllabus_complete": 1.0 if incomplete_flug == 0 else 0.0,
        "ipug_syllabus_complete": 1.0 if incomplete_ipug == 0 else 0.0,
        "deferred_syllabus_lines": float(len(pending)),
        "deferred_mqt_lines": float(
            sum(1 for d in pending if d.upgrade == Upgrade.MQT)
        ),
        "deferred_flug_lines": float(
            sum(1 for d in pending if d.upgrade == Upgrade.FLUG)
        ),
        "deferred_ipug_lines": float(
            sum(1 for d in pending if d.upgrade == Upgrade.IPUG)
        ),
    }


def handoff_parquet_columns(avg: dict) -> dict:
    """Final parquet row fragment from averaged handoff iteration metrics."""
    return {
        "incomplete_mqt_students_mean": avg["incomplete_mqt_students"],
        "incomplete_flug_students_mean": avg["incomplete_flug_students"],
        "incomplete_ipug_students_mean": avg["incomplete_ipug_students"],
        "mqt_syllabus_complete_frac": avg["mqt_syllabus_complete"],
        "flug_syllabus_complete_frac": avg["flug_syllabus_complete"],
        "ipug_syllabus_complete_frac": avg["ipug_syllabus_complete"],
        "deferred_syllabus_lines_mean": avg["deferred_syllabus_lines"],
        "deferred_mqt_lines_mean": avg["deferred_mqt_lines"],
        "deferred_flug_lines_mean": avg["deferred_flug_lines"],
        "deferred_ipug_lines_mean": avg["deferred_ipug_lines"],
        "deferred_syllabus_lines_mqt_mean": avg["deferred_mqt_lines"],
        "deferred_syllabus_lines_flug_mean": avg["deferred_flug_lines"],
        "deferred_syllabus_lines_ipug_mean": avg["deferred_ipug_lines"],
    }
