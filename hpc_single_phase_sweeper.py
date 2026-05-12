import math
import os
import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from src.engine import run_phase_simulation, create_pilots
from src.models import SquadronConfig, Upgrade
from src.rap_state import (
    rap_assess,
    rap_state_code,
    rap_state_label,
    sim_rap_metrics,
    sim_rap_state_code,
    sim_rap_state_label,
    mqt_observed_sortie_metrics,
    mqt_observed_sim_metrics,
)

# --- HPC CONFIGURATION ---
PHASE_DAYS = 120
ITERATIONS_PER_CONFIG = 3
OUTPUT_DIR = "outputs/single_phase/parquet"  
CHUNK_SIZE = 500000 

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _sweep_rank_and_size():
    """
    Multi-task (e.g. Slurm ``srun -n 16``): ``SLURM_PROCID`` and ``SLURM_NTASKS`` are set.

    Override with ``SWEEP_TASK_ID`` / ``SWEEP_NUM_TASKS`` for testing. If Slurm ntasks
    is set but you are not inside ``srun`` (no ``SLURM_PROCID``), returns (0, 1) so a
    single ``python`` process does not accidentally shard.
    """
    if "SWEEP_TASK_ID" in os.environ and "SWEEP_NUM_TASKS" in os.environ:
        return int(os.environ["SWEEP_TASK_ID"]), max(1, int(os.environ["SWEEP_NUM_TASKS"]))
    if "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        return int(os.environ["SLURM_PROCID"]), max(1, int(os.environ["SLURM_NTASKS"]))
    return 0, 1


def _sweep_lists_and_shape():
    """Same parameter order as ``itertools.product`` in ``get_sweep_configs``."""
    ute_values = list(range(6, 21))
    ip_qty_values = list(range(3, 10))
    exp_ratios = list(np.linspace(0.0, 1.0, 21).round(2))
    paa_values = list(range(18, 24))
    mqt_students = list(range(0, 15))
    flug_students = list(range(0, 15))
    ipug_students = list(range(0, 15))
    total_pilots = list(range(25, 50))
    lists = [
        ute_values,
        ip_qty_values,
        exp_ratios,
        paa_values,
        mqt_students,
        flug_students,
        ipug_students,
        total_pilots,
    ]
    shape = tuple(len(x) for x in lists)
    return lists, shape


def _config_at_flat_index(g: int, lists, shape) -> tuple:
    """One full-grid tuple (same ordering as ``itertools.product(*lists)``)."""
    idxs = np.unravel_index(g, shape)
    return tuple(lists[d][idxs[d]] for d in range(len(lists)))


def iter_sharded_raw_configs(rank: int, size: int):
    """Stride the flat grid so each rank touches ~1/size of all raw combos (no 16× scan)."""
    lists, shape = _sweep_lists_and_shape()
    total = int(math.prod(shape))
    for g in range(rank, total, size):
        yield _config_at_flat_index(g, lists, shape)


def get_sweep_configs():
    ute_values = range(6, 21)
    ip_qty_values = range(3, 10)
    exp_ratios = np.linspace(0.0, 1.0, 21).round(2)
    paa_values = range(18, 24)
    mqt_students = range(0, 15)
    flug_students = range(0, 15)
    ipug_students = range(0, 15)
    total_pilots = range(25, 50)

    keys = ['ute', 'ip_qty', 'exp', 'paa', 'mqt', 'flug', 'ipug', 'total_pilots']
    values = [ute_values, ip_qty_values, exp_ratios, paa_values, mqt_students, flug_students, ipug_students, total_pilots]
    
    return keys, itertools.product(*values)

def is_valid_config(total, exp, ip_q, mqt, flug):
    """
    Fast-fail check to skip invalid math before starting simulation objects.
    Returns True if valid, False if impossible.
    """
    experienced = int(total * exp)
    wg_count = total - experienced
    
    # Rule 1: Must have enough experienced pilots for IPs
    if ip_q > experienced: return False
    
    # Rule 2: Experienced pilots cannot exceed total (redundant but safe)
    if experienced > total: return False
    
    # Rule 3: Cannot have more students than eligible candidates
    # (Approximation: MQT+FLUG students come from WG pool)
    if (mqt + flug) > wg_count: return False

    return True

def process_single_config(args):
    """
    Worker function. Runs one configuration `n` times and returns the average.
    """
    (ute, ip_q, exp, paa, mqt, flug, ipug, total) = args

    # 2. Setup Config Object
    cfg = SquadronConfig(
        paa=int(paa), ute=float(ute), experience_ratio=float(exp), ip_qty=int(ip_q),
        mqt_students=int(mqt), flug_students=int(flug), ipug_students=int(ipug),
        phase_length_days=PHASE_DAYS, total_pilots=int(total), id=99
    )

    # 3. Run Loop
    results = []
    
    for _ in range(ITERATIONS_PER_CONFIG):
        try:
            # We MUST create pilots to give the engine containers for sorties
            pilots = create_pilots(cfg)
            final_pilots = run_phase_simulation(cfg, pilots, allocation_noise=0.0)
            
            # Extract metrics + upgrade syllabus handoff (same cfg object updated each iteration)
            h = cfg.last_phase_upgrade_handoff
            if h is None:
                raise ValueError("last_phase_upgrade_handoff not set after run_phase_simulation")

            rap, blue_rap, red = rap_assess(final_pilots)
            simm = sim_rap_metrics(final_pilots)
            mqt_sorties = mqt_observed_sortie_metrics(final_pilots)
            mqt_sims = mqt_observed_sim_metrics(final_pilots)

            results.append({
                "r_code": rap_state_code(rap),
                "b_code": rap_state_code(blue_rap),
                "sim_code": sim_rap_state_code(simm),
                "mqt_mo": mqt_sorties["sortie_mo"],
                "wg_mo": rap["WG"][1], "fl_mo": rap["FL"][1], "ip_mo": rap["IP"][1],
                "wg_b_mo": blue_rap["WG"][1], "fl_b_mo": blue_rap["FL"][1], "ip_b_mo": blue_rap["IP"][1],
                "wg_r_mo": red["WG"][1], "fl_r_mo": red["FL"][1], "ip_r_mo": red["IP"][1],
                "wg_r_pct": red["WG"][0], "fl_r_pct": red["FL"][0], "ip_r_pct": red["IP"][0],
                "mqt_sim_mo": mqt_sims["sim_mo"],
                "wg_sim_mo": simm["WG"]["sim_mo"],
                "fl_sim_mo": simm["FL"]["sim_mo"],
                "ip_sim_mo": simm["IP"]["sim_mo"],
                "mqt_sim_rap_sf": mqt_sims["sim_rap_shortfall"],
                "wg_sim_rap_sf": simm["WG"]["sim_rap_shortfall"],
                "fl_sim_rap_sf": simm["FL"]["sim_rap_shortfall"],
                "ip_sim_rap_sf": simm["IP"]["sim_rap_shortfall"],
                "mqt_syllabus_complete": float(h.mqt_syllabus_complete),
                "flug_syllabus_complete": float(h.flug_syllabus_complete),
                "ipug_syllabus_complete": float(h.ipug_syllabus_complete),
                "deferred_syllabus_lines": len(h.deferred_requirements),
                "deferred_mqt_lines": sum(1 for d in h.deferred_requirements if d.upgrade == Upgrade.MQT),
                "deferred_flug_lines": sum(1 for d in h.deferred_requirements if d.upgrade == Upgrade.FLUG),
                "deferred_ipug_lines": sum(1 for d in h.deferred_requirements if d.upgrade == Upgrade.IPUG),
            })

        except ValueError:
            return None # Catch-all for edge case logic errors

    if not results:
        return None

    # 4. Average Results (Manual math is faster than Pandas for small lists)
    n = len(results)
    avg = {k: sum(d[k] for d in results) / n for k in results[0]}

    # 5. Construct Final Row
    # We add the inputs back here for the CSV
    return {
        "paa": paa, "ute": ute,
        "total_capacity": cfg.paa * cfg.ute * cfg.phase_length_months,
        "exp_ratio": exp, "ip_qty": ip_q, "total_pilots": total,
        "mqt_qty": mqt, "flug_qty": flug, "ipug_qty": ipug,
        "rap_state_code": avg["r_code"],
        "rap_state_label": rap_state_label(int(round(avg["r_code"]))),
        "blue_rap_state_code": avg["b_code"],
        "blue_rap_state_label": rap_state_label(int(round(avg["b_code"]))),
        "sim_rap_state_code": avg["sim_code"],
        "sim_rap_state_label": sim_rap_state_label(int(round(avg["sim_code"]))),
        "mqt_monthly": avg["mqt_mo"],
        "wg_monthly": avg["wg_mo"], "fl_monthly": avg["fl_mo"], "ip_monthly": avg["ip_mo"],
        "wg_blue_monthly": avg["wg_b_mo"], "fl_blue_monthly": avg["fl_b_mo"], "ip_blue_monthly": avg["ip_b_mo"],
        "wg_red_monthly": avg["wg_r_mo"], "fl_red_monthly": avg["fl_r_mo"], "ip_red_monthly": avg["ip_r_mo"],
        "wg_red_pct": avg["wg_r_pct"], "fl_red_pct": avg["fl_r_pct"], "ip_red_pct": avg["ip_r_pct"],
        "mqt_sim_monthly": avg["mqt_sim_mo"],
        "wg_sim_monthly": avg["wg_sim_mo"],
        "fl_sim_monthly": avg["fl_sim_mo"],
        "ip_sim_monthly": avg["ip_sim_mo"],
        "mqt_sim_rap_shortfall_mean": avg["mqt_sim_rap_sf"],
        "wg_sim_rap_shortfall_mean": avg["wg_sim_rap_sf"],
        "fl_sim_rap_shortfall_mean": avg["fl_sim_rap_sf"],
        "ip_sim_rap_shortfall_mean": avg["ip_sim_rap_sf"],
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

def iter_valid_configs_sharded(rank: int, size: int):
    """Valid configs for this rank only; each raw grid point visited by exactly one rank."""
    for c in iter_sharded_raw_configs(rank, size):
        if is_valid_config(c[7], c[2], c[1], c[4], c[5]):
            yield c


def _max_workers_local() -> int:
    for env in ("SWEEP_WORKERS", "SLURM_CPUS_PER_TASK"):
        v = os.environ.get(env)
        if v:
            return max(1, int(v))
    return max(1, (os.cpu_count() or 1))


def run_parallel_sweep():
    rank, size = _sweep_rank_and_size()
    parallel_tasks = size > 1
    work_dir = os.path.join(OUTPUT_DIR, f"rank_{rank:03d}") if parallel_tasks else OUTPUT_DIR
    os.makedirs(work_dir, exist_ok=True)

    print(
        f"Sweep rank {rank}/{size} | work_dir={work_dir} | "
        f"{'Slurm tasks (no nested pool)' if parallel_tasks else 'local ProcessPoolExecutor'}"
    )

    completed_batches = {
        int(f.split("_")[1].split(".")[0])
        for f in os.listdir(work_dir)
        if f.startswith("batch_") and f.endswith(".parquet")
    }

    if completed_batches:
        last_batch = max(completed_batches)
        last_file = os.path.join(work_dir, f"batch_{last_batch:04d}.parquet")
        print(f"Clean-up: Removing potentially partial file {last_file}")
        if os.path.exists(last_file):
            os.remove(last_file)
        completed_batches.discard(last_batch)

    num_completed = len(completed_batches)
    rows_to_skip = num_completed * CHUNK_SIZE

    if parallel_tasks:
        config_iter = iter_valid_configs_sharded(rank, size)
    else:
        keys, param_generator = get_sweep_configs()
        config_iter = (
            c
            for c in param_generator
            if is_valid_config(c[7], c[2], c[1], c[4], c[5])
        )

    skipped_count = 0
    if rows_to_skip > 0:
        print(f"⏩ Fast-forwarding: Skipping {rows_to_skip:,} VALID configs for this rank...")
        for _ in range(rows_to_skip):
            try:
                next(config_iter)
            except StopIteration:
                print("⏩ End of stream while skipping; nothing left to run.")
                if not parallel_tasks:
                    with open("SWEEP_COMPLETE.txt", "w") as f:
                        f.write("Done")
                return
            skipped_count += 1
        print(f"⏩ Fast-forward complete. Ready to generate Batch {num_completed + 1}")

    batch_index = num_completed
    count = rows_to_skip
    buffer = []

    print(f"Writing batches under: {work_dir}")

    if parallel_tasks:
        for c in config_iter:
            result = process_single_config(c)
            if result:
                buffer.append(result)
            if len(buffer) >= CHUNK_SIZE:
                batch_index += 1
                batch_file = os.path.join(work_dir, f"batch_{batch_index:04d}.parquet")
                pd.DataFrame(buffer).to_parquet(batch_file, index=False)
                count += len(buffer)
                print(f"Saved {batch_file} | Total processed (rank): {count}", end="\r")
                buffer = []
        if buffer:
            batch_index += 1
            batch_file = os.path.join(work_dir, f"batch_{batch_index:04d}.parquet")
            pd.DataFrame(buffer).to_parquet(batch_file, index=False)
            count += len(buffer)
    else:
        workers = _max_workers_local()
        print(f"🚀 ProcessPoolExecutor max_workers={workers}")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(process_single_config, config_iter, chunksize=2000):
                if result:
                    buffer.append(result)
                if len(buffer) >= CHUNK_SIZE:
                    batch_index += 1
                    batch_file = os.path.join(work_dir, f"batch_{batch_index:04d}.parquet")
                    pd.DataFrame(buffer).to_parquet(batch_file, index=False)
                    count += len(buffer)
                    print(f"Saved {batch_file} | Total processed: {count}", end="\r")
                    buffer = []
        if buffer:
            batch_index += 1
            batch_file = os.path.join(work_dir, f"batch_{batch_index:04d}.parquet")
            pd.DataFrame(buffer).to_parquet(batch_file, index=False)
            count += len(buffer)

    if not parallel_tasks:
        with open("SWEEP_COMPLETE.txt", "w") as f:
            f.write("Done")
    print(f"\n✅ Rank {rank} done. Valid configs processed (this rank): {count}")


if __name__ == "__main__":
    run_parallel_sweep()