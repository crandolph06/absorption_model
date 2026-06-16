"""
Parity tests: deterministic heap allocation vs legacy assign_sortie/assign_sim loops.

The heap fast path (noise=0) in Tyler's engine changes should produce identical
per-pilot assignments to the pre-heap implementation.

Run: python -m unittest tests.test_allocation_path_parity -v

Requires heap helpers in src.engine (_assign_ct_sortie_from_heap). Skips on main
until that PR is merged.
"""
from __future__ import annotations

import copy
import math
import unittest
from typing import Callable, Dict, List, Tuple

import src.engine as engine
from src import rules
from src.engine import (
    _allocate_ct_buckets_round_robin,
    _allocation_sort_key,
    _ct_bucket_round_robin_key,
    _eligible_for_event,
    allocate_sim_rap,
    assign_sim,
    assign_sortie,
    create_pilots,
)
from src.models import (
    PHASE_DAYS_PER_NOTIONAL_MONTH,
    EventType,
    Pilot,
    Qual,
    SquadronConfig,
    Upgrade,
)
from src.syllabi import ContinuationBucket, ContinuationProfile, TEST_CONTINUATION_PROFILE

HEAP_PATH_AVAILABLE = hasattr(engine, "_assign_ct_sortie_from_heap")

ALLOC_FIELDS = (
    "sortie_phase",
    "sortie_blue_phase",
    "sortie_red_phase",
    "sortie_single_ship",
    "sim_phase",
    "flight_hours_phase",
    "sim_hours_phase",
)


def _allocation_snapshot(pilots: List[Pilot]) -> Tuple[Tuple[float, ...], ...]:
    return tuple(tuple(getattr(p, field) for field in ALLOC_FIELDS) for p in pilots)


def _make_roster(
    wg: int = 6,
    fl: int = 3,
    ip: int = 2,
    mqt: int = 1,
    *,
    ute: float = 3.0,
    paa: int = 9,
    sim_sessions_monthly: float = float("inf"),
) -> Tuple[SquadronConfig, List[Pilot]]:
    total = wg + fl + ip
    exp = fl + ip
    cfg = SquadronConfig(
        ute=ute,
        paa=paa,
        id=1,
        total_pilots=total,
        ip_qty=ip,
        experience_ratio=exp / total,
        sim_sessions_monthly=sim_sessions_monthly,
    )
    pilots = create_pilots(cfg)
    wgs = [p for p in pilots if p.qual == Qual.WG]
    for i in range(mqt):
        wgs[i].upgrade = Upgrade.MQT
    for p in pilots:
        p.set_rap_requirement()
    return cfg, pilots


def _bucket_quantities(
    profile: ContinuationProfile,
    remaining_capacity: int,
) -> Dict[ContinuationBucket, int]:
    raw_qty = [(b, remaining_capacity * b.fraction) for b in profile.buckets]
    base_qty = {b: int(x) for b, x in raw_qty}
    leftover = remaining_capacity - sum(base_qty.values())
    sorted_remainders = sorted(raw_qty, key=lambda x: x[1] - int(x[1]), reverse=True)
    for i in range(leftover):
        bucket = sorted_remainders[i % len(sorted_remainders)][0]
        base_qty[bucket] += 1
    return base_qty


def _allocate_ct_buckets_round_robin_legacy(
    buckets: List[ContinuationBucket],
    remaining: Dict[ContinuationBucket, int],
    ct_candidates: List[Pilot],
    cfg: SquadronConfig,
    phase_length_days: float,
    noise: float,
    single_ship: bool = False,
    single_ship_monthly_cap: float = 1.0,
) -> int:
    """Pre-heap implementation: repeated assign_sortie calls."""
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
                single_ship_monthly_cap=single_ship_monthly_cap,
            ):
                remaining[bucket] -= 1
                assigned += 1
                assigned_this_pass = True
            else:
                remaining[bucket] = 0

        if not assigned_this_pass:
            break
    return assigned


def assign_sim_legacy(
    cfg: SquadronConfig,
    candidates: List[Pilot],
    phase_length_days: float,
    noise: float = 0.0,
    exclude: set[int] | None = None,
) -> bool:
    """Pre-heap assign_sim: sort candidates and take the first."""
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


def _run_repeated_sim_assignments(
    assign_fn,
    cfg: SquadronConfig,
    pilots: List[Pilot],
    phase_length_days: float,
    *,
    rounds: int,
    pool_filter,
) -> int:
    """Syllabus-style repeated sim picks with distinct pilots per round."""
    assigned = 0
    for _ in range(rounds):
        exclude: set[int] = set()
        pool = [p for p in pilots if pool_filter(p)]
        if assign_fn(cfg, pool, phase_length_days, noise=0.0, exclude=exclude):
            assigned += 1
    return assigned


def _allocate_sim_rap_legacy(
    pilots: List[Pilot],
    cfg: SquadronConfig,
    phase_length_days: float,
    noise: float = 0.0,
) -> None:
    """Pre-heap implementation: repeated assign_sim calls."""
    phase_months = float(phase_length_days) / PHASE_DAYS_PER_NOTIONAL_MONTH
    if math.isfinite(cfg.sim_sessions_monthly):
        sim_capacity = int(cfg.sim_sessions_monthly * cfg.sim_bays_per_session * phase_months)
        sim_capacity = max(0, sim_capacity - int(cfg.deferred_sim_burden))
    else:
        sim_capacity = None
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


def _assert_ct_parity(
    testcase: unittest.TestCase,
    *,
    phase_length_days: float,
    remaining_capacity: int,
    profile: ContinuationProfile = TEST_CONTINUATION_PROFILE,
    single_ship: bool = False,
    seed_events: Callable[[List[Pilot]], None] | None = None,
    bucket_filter: Callable[[ContinuationBucket], bool] | None = None,
) -> None:
    cfg, pilots = _make_roster()
    if seed_events is not None:
        seed_events(pilots)

    base_qty = _bucket_quantities(profile, remaining_capacity)
    buckets = sorted(base_qty.keys(), key=_ct_bucket_round_robin_key)
    if bucket_filter is not None:
        buckets = [b for b in buckets if bucket_filter(b)]
    remaining_template = {b: base_qty[b] for b in buckets}

    pilots_legacy = copy.deepcopy(pilots)
    pilots_heap = copy.deepcopy(pilots)
    ct_candidates_legacy = [p for p in pilots_legacy if p.upgrade != Upgrade.MQT]
    ct_candidates_heap = [p for p in pilots_heap if p.upgrade != Upgrade.MQT]
    remaining_legacy = dict(remaining_template)
    remaining_heap = dict(remaining_template)

    legacy_assigned = _allocate_ct_buckets_round_robin_legacy(
        buckets,
        remaining_legacy,
        ct_candidates_legacy,
        cfg,
        phase_length_days,
        noise=0.0,
        single_ship=single_ship,
    )
    heap_assigned = _allocate_ct_buckets_round_robin(
        buckets,
        remaining_heap,
        ct_candidates_heap,
        cfg,
        phase_length_days,
        noise=0.0,
        single_ship=single_ship,
    )

    testcase.assertEqual(legacy_assigned, heap_assigned)
    testcase.assertEqual(
        _allocation_snapshot(ct_candidates_legacy),
        _allocation_snapshot(ct_candidates_heap),
    )
    testcase.assertEqual(remaining_legacy, remaining_heap)


def _assert_sim_rap_parity(
    testcase: unittest.TestCase,
    *,
    phase_length_days: float,
    sim_sessions_monthly: float = float("inf"),
    seed_events: Callable[[List[Pilot]], None] | None = None,
) -> None:
    cfg, pilots = _make_roster(sim_sessions_monthly=sim_sessions_monthly)
    if seed_events is not None:
        seed_events(pilots)

    pilots_legacy = copy.deepcopy(pilots)
    pilots_heap = copy.deepcopy(pilots)

    _allocate_sim_rap_legacy(pilots_legacy, cfg, phase_length_days, noise=0.0)
    allocate_sim_rap(pilots_heap, cfg, phase_length_days, noise=0.0)

    testcase.assertEqual(_allocation_snapshot(pilots_legacy), _allocation_snapshot(pilots_heap))


def _assert_assign_sim_parity(
    testcase: unittest.TestCase,
    *,
    phase_length_days: float,
    rounds: int,
    seed_events: Callable[[List[Pilot]], None] | None = None,
    pool_filter: Callable[[Pilot], bool] | None = None,
) -> None:
    cfg, pilots = _make_roster()
    if seed_events is not None:
        seed_events(pilots)
    pool_filter = pool_filter or (lambda p: p.qual in {Qual.IP, Qual.FL, Qual.WG})

    pilots_legacy = copy.deepcopy(pilots)
    pilots_heap = copy.deepcopy(pilots)

    legacy_assigned = _run_repeated_sim_assignments(
        assign_sim_legacy,
        cfg,
        pilots_legacy,
        phase_length_days,
        rounds=rounds,
        pool_filter=pool_filter,
    )
    heap_assigned = _run_repeated_sim_assignments(
        assign_sim,
        cfg,
        pilots_heap,
        phase_length_days,
        rounds=rounds,
        pool_filter=pool_filter,
    )

    testcase.assertEqual(legacy_assigned, heap_assigned)
    testcase.assertEqual(_allocation_snapshot(pilots_legacy), _allocation_snapshot(pilots_heap))


@unittest.skipUnless(HEAP_PATH_AVAILABLE, "Heap allocation helpers not present in src.engine")
class AllocationPathParityTests(unittest.TestCase):
    def test_ct_round_robin_fl_buckets_30_day(self) -> None:
        _assert_ct_parity(
            self,
            phase_length_days=30,
            remaining_capacity=12,
            bucket_filter=lambda b: b.min_qual == Qual.FL,
        )

    def test_ct_round_robin_wg_buckets_single_ship_90_day(self) -> None:
        _assert_ct_parity(
            self,
            phase_length_days=90,
            remaining_capacity=24,
            single_ship=True,
            bucket_filter=lambda b: b.min_qual == Qual.WG,
        )

    def test_ct_round_robin_all_buckets_with_existing_events(self) -> None:
        def seed(pilots: List[Pilot]) -> None:
            for i, pilot in enumerate(pilots):
                pilot.sortie_phase = float(i % 5)
                pilot.sortie_blue_phase = float(i % 3)
                pilot.sortie_red_phase = float(i % 2)
                pilot.sim_phase = float(i % 4)

        _assert_ct_parity(
            self,
            phase_length_days=60,
            remaining_capacity=18,
            seed_events=seed,
        )

    def test_ct_round_robin_near_event_cap(self) -> None:
        phase_length_days = 90.0
        months = phase_length_days / PHASE_DAYS_PER_NOTIONAL_MONTH
        max_events = 20.0 * months

        def seed(pilots: List[Pilot]) -> None:
            for i, pilot in enumerate(pilots):
                pilot.sortie_phase = max_events - float(i % 3)
                pilot.sim_phase = float(i % 2)

        _assert_ct_parity(
            self,
            phase_length_days=phase_length_days,
            remaining_capacity=10,
            seed_events=seed,
        )

    def test_sim_rap_unlimited_capacity(self) -> None:
        _assert_sim_rap_parity(self, phase_length_days=30)

    def test_sim_rap_finite_capacity(self) -> None:
        _assert_sim_rap_parity(
            self,
            phase_length_days=90,
            sim_sessions_monthly=8.0,
        )

    def test_sim_rap_with_existing_sims_and_sorties(self) -> None:
        def seed(pilots: List[Pilot]) -> None:
            for i, pilot in enumerate(pilots):
                pilot.sortie_phase = float(i % 6)
                pilot.sortie_blue_phase = float(i % 4)
                pilot.sim_phase = float(i % 3)

        _assert_sim_rap_parity(
            self,
            phase_length_days=60,
            sim_sessions_monthly=12.0,
            seed_events=seed,
        )

    def test_assign_sim_sort_vs_min_basic(self) -> None:
        _assert_assign_sim_parity(self, phase_length_days=30, rounds=8)

    def test_assign_sim_sort_vs_min_with_existing_events(self) -> None:
        def seed(pilots: List[Pilot]) -> None:
            for i, pilot in enumerate(pilots):
                pilot.sortie_phase = float(i % 4)
                pilot.sim_phase = float(i % 5)

        _assert_assign_sim_parity(
            self,
            phase_length_days=90,
            rounds=15,
            seed_events=seed,
        )

    def test_assign_sim_sort_vs_min_ip_pool_only(self) -> None:
        _assert_assign_sim_parity(
            self,
            phase_length_days=60,
            rounds=6,
            pool_filter=lambda p: p.qual == Qual.IP,
        )


if __name__ == "__main__":
    unittest.main()
