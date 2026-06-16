import pandas as pd
import numpy as np
import os
import itertools
from concurrent.futures import ProcessPoolExecutor
from src.engine import run_phase_simulation, create_pilots, phase_upgrade_metrics
from src.models import SquadronConfig
from src.simulation_config import DEFAULT_PHASE_LENGTH_DAYS, SimulationConfig
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
SIM_CONFIG = SimulationConfig(phase_length_days=DEFAULT_PHASE_LENGTH_DAYS)
ITERATIONS_PER_CONFIG = 3
OUTPUT_DIR = "outputs/single_phase/parquet"  
CHUNK_SIZE = 500000 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_sweep_configs():
    ute_values = (6,)
    ip_qty_values = range(1, 10)
    exp_ratios = np.concatenate([
        np.arange(0.02, 0.10, 0.02),
        np.linspace(0.10, 1.0, 19).round(2)
    ])
    paa_values = range(18, 24)
    mqt_students = range(0, 15)
    flug_students = range(0, 15)
    ipug_students = range(0, 15)
    total_pilots = range(25, 50)

    keys = ['ute', 'ip_qty', 'exp', 'paa', 'mqt', 'flug', 'ipug', 'total_pilots']
    values = [ute_values, ip_qty_values, exp_ratios, paa_values, mqt_students, flug_students, ipug_students, total_pilots]
    
    return keys, itertools.product(*values)

def is_valid_config(total, exp, ip_q, mqt, flug, ipug):
    """
    Fast-fail check to skip invalid math before starting simulation objects.
    Returns True if valid, False if impossible.
    """
    experienced = int(total * exp)
    wg_count = total - experienced
    fl_count = experienced - ip_q
    
    # Rule 1: Must have enough experienced pilots for IPs
    if ip_q > experienced: return False
    
    # Rule 2: Experienced pilots cannot exceed total (redundant but safe)
    if experienced > total: return False
    
    # Rule 3: Cannot have more students than eligible candidates
    if (mqt + flug + ipug + ip_q) > total: return False
    if (mqt + flug) > wg_count: return False
    if ipug > fl_count: return False


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
        total_pilots=int(total), id=99
    )

    # 3. Run Loop
    results = []
    
    for _ in range(ITERATIONS_PER_CONFIG):
        try:
            # We MUST create pilots to give the engine containers for sorties
            pilots = create_pilots(cfg)
            final_pilots = run_phase_simulation(cfg, pilots, sim_config=SIM_CONFIG)
            
            rap, blue_rap, red = rap_assess(final_pilots)
            simm = sim_rap_metrics(final_pilots)
            mqt_sorties = mqt_observed_sortie_metrics(final_pilots)
            mqt_sims = mqt_observed_sim_metrics(final_pilots)
            u = phase_upgrade_metrics(final_pilots)
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
                "deferred_mqt_sorties": float(u["deferred_mqt_sorties"]),
                "deferred_flug_sorties": float(u["deferred_flug_sorties"]),
                "deferred_ipug_sorties": float(u["deferred_ipug_sorties"]),
                "deferred_mqt_sims": float(u["deferred_mqt_sims"]),
                "deferred_flug_sims": float(u["deferred_flug_sims"]),
                "deferred_ipug_sims": float(u["deferred_ipug_sims"]),
                "remaining_mqt_syllabi": float(u["remaining_mqt_syllabi"]),
                "remaining_flug_syllabi": float(u["remaining_flug_syllabi"]),
                "remaining_ipug_syllabi": float(u["remaining_ipug_syllabi"]),
                "remaining_mqt_syllabi_sorties_only": float(u["remaining_mqt_syllabi_sorties_only"]),
                "remaining_flug_syllabi_sorties_only": float(u["remaining_flug_syllabi_sorties_only"]),
                "remaining_ipug_syllabi_sorties_only": float(u["remaining_ipug_syllabi_sorties_only"]),
                "self_terminating_phase": float(cfg.self_terminating_phase),
                "self_terminating_run": float(cfg.self_terminating_run),
                "ip_at_cap_count": float(cfg.pipeline_ip_at_cap_count),
                "ip_available_count": float(cfg.pipeline_ip_available_count),
                "max_ip_events_monthly": float(cfg.pipeline_max_ip_events_monthly),
                "deferred_due_to_ip": float(cfg.pipeline_deferred_due_to_ip),
                "unallocated_iron": float(cfg.unallocated_iron),
            })

        except ValueError:
            return None # Catch-all for edge case logic errors

    if not results:
        return None

    # 4. Average Results (Manual math is faster than Pandas for small lists)
    n = len(results)
    avg = {k: sum(d[k] for d in results) / n for k in results[0]}

    # 5. Construct Final Row
    experienced = int(total * exp)
    wg_qty = total - experienced
    fl_qty = experienced - ip_q
    return {
        "paa": paa, "ute": ute,
        "total_capacity": cfg.paa * cfg.ute * SIM_CONFIG.phase_length_months,
        "exp_ratio": exp, "ip_qty": ip_q, "total_pilots": total,
        "mqt_qty": mqt, "flug_qty": flug, "ipug_qty": ipug,
        "wg_qty": wg_qty, "fl_qty": fl_qty,
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
        "deferred_mqt_sorties_mean": avg["deferred_mqt_sorties"],
        "deferred_flug_sorties_mean": avg["deferred_flug_sorties"],
        "deferred_ipug_sorties_mean": avg["deferred_ipug_sorties"],
        "deferred_mqt_sims_mean": avg["deferred_mqt_sims"],
        "deferred_flug_sims_mean": avg["deferred_flug_sims"],
        "deferred_ipug_sims_mean": avg["deferred_ipug_sims"],
        "remaining_mqt_syllabi_mean": avg["remaining_mqt_syllabi"],
        "remaining_flug_syllabi_mean": avg["remaining_flug_syllabi"],
        "remaining_ipug_syllabi_mean": avg["remaining_ipug_syllabi"],
        "remaining_mqt_syllabi_sorties_only_mean": avg["remaining_mqt_syllabi_sorties_only"],
        "remaining_flug_syllabi_sorties_only_mean": avg["remaining_flug_syllabi_sorties_only"],
        "remaining_ipug_syllabi_sorties_only_mean": avg["remaining_ipug_syllabi_sorties_only"],
        "self_terminating_phase_mean": avg["self_terminating_phase"],
        "self_terminating_run_mean": avg["self_terminating_run"],
        "ip_at_cap_count_mean": avg["ip_at_cap_count"],
        "ip_available_count_mean": avg["ip_available_count"],
        "max_ip_events_monthly_mean": avg["max_ip_events_monthly"],
        "deferred_due_to_ip_mean": avg["deferred_due_to_ip"],
        "unallocated_iron_mean": avg["unallocated_iron"],
    }

def run_parallel_sweep():
    
    print("Generating parameter space...")
    keys, param_generator = get_sweep_configs()

    batch_prefix = "batch_1_"
    completed_batches: set[int] = set()
    for f in os.listdir(OUTPUT_DIR):
        if f.startswith(batch_prefix) and f.endswith('.parquet'):
            stem = f[len(batch_prefix) : -len('.parquet')]
            if stem.isdigit():
                completed_batches.add(int(stem))


    num_completed = 0
    if completed_batches:
        last_batch = max(completed_batches)
        last_file = os.path.join(OUTPUT_DIR, f"{batch_prefix}{last_batch:04d}.parquet")
        print(f"Clean-up: Removing potentially partial file {last_file}")
        if os.path.exists(last_file):
            os.remove(last_file)
        completed_batches.remove(last_batch)
        num_completed = last_batch - 1

    rows_to_skip = num_completed * CHUNK_SIZE

    skipped_count = 0
    if rows_to_skip > 0:
        print(f"⏩ Fast-forwarding: Skipping {rows_to_skip:,} VALID configurations...")
        for c in param_generator:
            if is_valid_config(total=c[7], exp=c[2], ip_q=c[1], mqt=c[4], flug=c[5], ipug=c[6]):
                skipped_count += 1
            if skipped_count >= rows_to_skip:
                break

        print(f"⏩ Fast-forward complete. Ready to generate Batch {num_completed + 1}")

    print("🎯 Pre-filtering valid configurations...")
    valid_configs = (
        c for c in param_generator 
        if is_valid_config(total=c[7], exp=c[2], ip_q=c[1], mqt=c[4], flug=c[5], ipug=c[6])
        )

    print(f"🚀 Launching Parallel Sweep on {os.cpu_count()} cores...")

    batch_index = num_completed
    count = rows_to_skip
    buffer = []

    print(f"Writing batches to: {OUTPUT_DIR}") 

    with ProcessPoolExecutor() as executor:
        for result in executor.map(process_single_config, valid_configs, chunksize=2000):
            if result:
                buffer.append(result)

            if len(buffer) >= CHUNK_SIZE:
                batch_index += 1
                batch_file = os.path.join(OUTPUT_DIR, f"{batch_prefix}{batch_index:04d}.parquet")
                
                # Convert buffer to DataFrame and save to Parquet
                df_chunk = pd.DataFrame(buffer)
                df_chunk.to_parquet(batch_file, index=False) 
                
                count += len(buffer)
                print(f"Saved {batch_file} | Total processed: {count}", end='\r')
                buffer = [] # Clear RAM

        # Final Flush
        if buffer:
            batch_index += 1
            if batch_index not in completed_batches:
                batch_file = os.path.join(OUTPUT_DIR, f"{batch_prefix}{batch_index:04d}.parquet")
                pd.DataFrame(buffer).to_parquet(batch_file, index=False)
                count += len(buffer)

    with open("SWEEP_1_COMPLETE.txt", "w") as f:
        f.write("Done")
    print(f"\n✅ Sweep 1 Complete. Total valid configs: {count}")

if __name__ == "__main__":
    run_parallel_sweep()