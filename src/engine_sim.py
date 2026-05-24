"""Simulator session budget and post-syllabus EP / RAP sim allocation."""

import math
import random
from typing import List, Optional

from src.models import SIM_RAP_MONTHLY, SquadronConfig, Pilot, Qual
from src.syllabi import SyllabusEvent


def initial_sim_session_budget(cfg: SquadronConfig) -> float:
    """
    Total simulator **session** budget for the phase (fractional sessions allowed).

    Each month the wing flies about ``cfg.sim_sessions_monthly`` session lines; each
    line has ``cfg.sim_bays_per_session`` bays (packing: up to that many pilots in one
    session time slice, or ``ceil(P / bays)`` sessions for one training evolution with
    ``P`` concurrent pilots).
    """
    return max(0.0, float(cfg.sim_sessions_monthly) * cfg.phase_length_months)


def solo_sim_session_fraction(cfg: SquadronConfig) -> float:
    """Sessions consumed by one solo EP / RAP sim when four solos pack into one 4-bay session."""
    b = max(1, int(cfg.sim_bays_per_session))
    return 1.0 / float(b)


def syllabus_sim_session_cost(event: SyllabusEvent, cfg: SquadronConfig) -> float:
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
    frac = solo_sim_session_fraction(cfg)
    eligible = [p for p in pilots if p.qual in (Qual.WG, Qual.FL, Qual.IP)]
    n = max(0.0, float(noise))
    while sim_session_budget[0] + 1e-9 >= frac:
        need_ep = [p for p in eligible if p.ep_sim_phase + 1e-9 < months]
        if not need_ep:
            break

        def _ep_key(p: Pilot) -> tuple:
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
    frac = solo_sim_session_fraction(cfg)
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
        best_p.add_sim(cfg.avg_sortie_dur)
        sim_session_budget[0] -= frac
