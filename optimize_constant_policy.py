#!/usr/bin/env python3
"""
Simulation-based optimization of constant CAF manning levers (paper path 1).

Uses ``run_phase_simulation`` as the flying plant inside ``CAFSimulation`` instead
of the sortie brain. Start with a small squadron set and few years, then scale up.

Examples:
  python optimize_constant_policy.py --preset test --years 5 --trials 20
  python optimize_constant_policy.py --preset pragmatic --years 10 --trials 50 --method de
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.manning_config import SQUADRON_DATA, TEST_SQUADRON_DATA, get_initial_squadrons
from src.manning_engine import CAFSimulation
from src.manning_objective import objective_from_history, terminal_metrics
from src.models import PriorityMode
from src.simulation_config import SimulationConfig


@dataclass(frozen=True)
class ConstantManningPolicy:
    annual_intake: int
    retention_rate: float
    max_manning_pct: int
    flug_window_start: int
    ipug_window_start: int
    sq_phase_flug_intake: int
    sq_phase_ipug_intake: int
    ute: float
    round_robin: bool = True
    use_upgrade_quotas: bool = True


PRAGMATIC_DEFAULT = ConstantManningPolicy(
    annual_intake=240,
    retention_rate=0.25,
    max_manning_pct=150,
    flug_window_start=250,
    ipug_window_start=400,
    sq_phase_flug_intake=3,
    sq_phase_ipug_intake=2,
    ute=10.0,
)

# (name, lower, upper, integer)
SEARCH_BOUNDS = [
    ("annual_intake", 120, 360, True),
    ("retention_rate", 0.10, 0.45, False),
    ("max_manning_pct", 100, 180, True),
    ("flug_window_start", 150, 350, True),
    ("ipug_window_start", 300, 550, True),
    ("sq_phase_flug_intake", 0, 8, True),
    ("sq_phase_ipug_intake", 0, 6, True),
    ("ute", 6.0, 14.0, False),
]


def vector_to_policy(x: np.ndarray) -> ConstantManningPolicy:
    values = {}
    for i, (name, lo, hi, as_int) in enumerate(SEARCH_BOUNDS):
        val = float(np.clip(x[i], lo, hi))
        if as_int:
            val = int(round(val))
        values[name] = val
    return ConstantManningPolicy(
        round_robin=True,
        use_upgrade_quotas=True,
        **values,
    )


def policy_to_vector(policy: ConstantManningPolicy) -> np.ndarray:
    d = asdict(policy)
    return np.array([d[name] for name, _, _, _ in SEARCH_BOUNDS], dtype=float)


def build_sim(
    policy: ConstantManningPolicy,
    sim_config: SimulationConfig,
) -> CAFSimulation:
    sim = CAFSimulation(
        annual_intake=policy.annual_intake,
        retention_rate=policy.retention_rate,
        round_robin=policy.round_robin,
        brain=None,
        flug_window_start=policy.flug_window_start,
        ipug_window_start=policy.ipug_window_start,
        max_manning_pct=policy.max_manning_pct,
        staff_priority_mode=PriorityMode.RANDOM,
        use_upgrade_quotas=policy.use_upgrade_quotas,
        sim_config=sim_config,
        use_physics_allocator=True,
    )
    if policy.use_upgrade_quotas:
        sim.sq_phase_flug_intake = policy.sq_phase_flug_intake
        sim.sq_phase_ipug_intake = policy.sq_phase_ipug_intake
    return sim


def evaluate_policy(
    policy: ConstantManningPolicy,
    *,
    years: int,
    squadron_data,
    seed: int,
    sim_config: SimulationConfig,
) -> tuple[float, dict]:
    random.seed(seed)
    np.random.seed(seed)

    sim = build_sim(policy, sim_config)
    squadrons = get_initial_squadrons(sim.current_year, squadron_data)
    history = sim.run_simulation(years, squadrons, ute=policy.ute)
    breakdown = terminal_metrics(sim, history)
    return breakdown["cost"], breakdown


def random_search(
    n_trials: int,
    *,
    years: int,
    squadron_data,
    seed: int,
    sim_config: SimulationConfig,
    baseline: ConstantManningPolicy,
) -> tuple[ConstantManningPolicy, dict]:
    bounds = SEARCH_BOUNDS
    best_policy = baseline
    best_cost, best_metrics = evaluate_policy(
        baseline, years=years, squadron_data=squadron_data, seed=seed, sim_config=sim_config
    )
    print(f"Baseline cost: {best_cost:.4f}  terminal_pilots={best_metrics['terminal_pilots']}")

    rng = np.random.default_rng(seed)
    for trial in range(1, n_trials):
        x = []
        for _, lo, hi, as_int in bounds:
            val = rng.uniform(lo, hi)
            if as_int:
                val = int(round(val))
            x.append(val)
        policy = vector_to_policy(np.array(x, dtype=float))
        cost, metrics = evaluate_policy(
            policy, years=years, squadron_data=squadron_data, seed=seed, sim_config=sim_config
        )
        print(
            f"  trial {trial}/{n_trials} cost={cost:.4f} "
            f"shortfall={metrics['mean_shortfall']:.3f} pilots={metrics['terminal_pilots']}"
        )
        if cost < best_cost:
            best_cost = cost
            best_policy = policy
            best_metrics = metrics

    return best_policy, best_metrics


def differential_evolution_search(
    *,
    years: int,
    squadron_data,
    seed: int,
    sim_config: SimulationConfig,
    maxiter: int,
    popsize: int,
) -> tuple[ConstantManningPolicy, dict]:
    from scipy.optimize import differential_evolution

    eval_count = 0
    best_seen = {"cost": float("inf"), "metrics": None}

    def cost_fn(x):
        nonlocal eval_count
        eval_count += 1
        policy = vector_to_policy(np.array(x, dtype=float))
        cost, metrics = evaluate_policy(
            policy, years=years, squadron_data=squadron_data, seed=seed, sim_config=sim_config
        )
        if cost < best_seen["cost"]:
            best_seen["cost"] = cost
            best_seen["metrics"] = metrics
            print(
                f"  eval {eval_count} new best cost={cost:.4f} "
                f"shortfall={metrics['mean_shortfall']:.3f} pilots={metrics['terminal_pilots']}"
            )
        return cost

    bounds = [(lo, hi) for _, lo, hi, _ in SEARCH_BOUNDS]
    t0 = time.time()
    result = differential_evolution(
        cost_fn,
        bounds,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        polish=False,
        updating="immediate",
        workers=1,
    )
    elapsed = time.time() - t0
    policy = vector_to_policy(result.x)
    _, metrics = evaluate_policy(
        policy, years=years, squadron_data=squadron_data, seed=seed, sim_config=sim_config
    )
    metrics["optimizer_seconds"] = elapsed
    metrics["optimizer_evaluations"] = eval_count
    metrics["optimizer_success"] = bool(result.success)
    return policy, metrics


def main():
    parser = argparse.ArgumentParser(description="Optimize constant CAF manning levers (physics sim).")
    parser.add_argument("--years", type=int, default=5, help="Simulation horizon in years.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for initial squadrons.")
    parser.add_argument("--preset", choices=("test", "full", "pragmatic"), default="test")
    parser.add_argument("--method", choices=("random", "de"), default="random")
    parser.add_argument("--trials", type=int, default=20, help="Random-search trials or DE maxiter.")
    parser.add_argument("--popsize", type=int, default=5, help="DE population multiplier.")
    parser.add_argument("--output", type=Path, default=None, help="JSON path for best policy + metrics.")
    parser.add_argument("--phase-days", type=float, default=None, help="Override SimulationConfig phase length.")
    args = parser.parse_args()

    if args.preset == "test":
        squadron_data = TEST_SQUADRON_DATA
    elif args.preset == "full":
        squadron_data = SQUADRON_DATA
    else:
        squadron_data = SQUADRON_DATA

    sim_config = SimulationConfig()
    if args.phase_days is not None:
        sim_config = SimulationConfig(phase_length_days=args.phase_days)

    baseline = PRAGMATIC_DEFAULT
    print(
        f"Physics-backed optimization | preset={args.preset} years={args.years} "
        f"squadrons={len(squadron_data)} method={args.method}"
    )

    if args.method == "random":
        best_policy, metrics = random_search(
            args.trials,
            years=args.years,
            squadron_data=squadron_data,
            seed=args.seed,
            sim_config=sim_config,
            baseline=baseline,
        )
    else:
        try:
            best_policy, metrics = differential_evolution_search(
                years=args.years,
                squadron_data=squadron_data,
                seed=args.seed,
                sim_config=sim_config,
                maxiter=args.trials,
                popsize=args.popsize,
            )
        except ImportError:
            print("scipy not available; falling back to random search.", file=sys.stderr)
            best_policy, metrics = random_search(
                args.trials,
                years=args.years,
                squadron_data=squadron_data,
                seed=args.seed,
                sim_config=sim_config,
                baseline=baseline,
            )

    print("\n=== Best policy ===")
    print(json.dumps(asdict(best_policy), indent=2))
    print("\n=== Metrics ===")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}, indent=2))

    if args.output:
        payload = {"policy": asdict(best_policy), "metrics": metrics}
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
