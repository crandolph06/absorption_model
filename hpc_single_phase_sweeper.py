import pandas as pd
import numpy as np
import os
import itertools
from concurrent.futures import ProcessPoolExecutor
from src.engine import run_phase_simulation, create_pilots
from src.models import SquadronConfig
from src.rap_state import rap_assess, rap_state_code, rap_state_label

# --- HPC CONFIGURATION ---
PHASE_DAYS = 120
ITERATIONS_PER_CONFIG = 3
OUTPUT_DIR = "outputs/single_phase/parquet"  
CHUNK_SIZE = 500000 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_sweep_configs():
    ute_values = range(6, 21)
    ip_qty_values = range(3, 10)
    exp_ratios = np.linspace(0.0, 1.0, 21).round(2)
    paa_values = range( 18, 24)
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
            
            # Extract Metrics
            rap, blue_rap, red = rap_assess(final_pilots)
            
            # Store raw numbers for averaging
            results.append({
                "r_code": rap_state_code(rap),
                "b_code": rap_state_code(blue_rap),
                "mqt_mo": rap["MQT"][1], 
                "wg_mo": rap["WG"][1], "fl_mo": rap["FL"][1], "ip_mo": rap["IP"][1],
                "wg_b_mo": blue_rap["WG"][1], "fl_b_mo": blue_rap["FL"][1], "ip_b_mo": blue_rap["IP"][1],
                "wg_r_mo": red["WG"][1], "fl_r_mo": red["FL"][1], "ip_r_mo": red["IP"][1],
                "wg_r_pct": red["WG"][0], "fl_r_pct": red["FL"][0], "ip_r_pct": red["IP"][0]
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
        "total_capacity": cfg.paa * cfg.ute * (PHASE_DAYS / 30),
        "exp_ratio": exp, "ip_qty": ip_q, "total_pilots": total,
        "mqt_qty": mqt, "flug_qty": flug, "ipug_qty": ipug,
        "rap_state_code": avg["r_code"], 
        "rap_state_label": rap_state_label(int(round(avg["r_code"]))),
        "blue_rap_state_code": avg["b_code"], 
        "blue_rap_state_label": rap_state_label(int(round(avg["b_code"]))),
        "mqt_monthly": avg["mqt_mo"], 
        "wg_monthly": avg["wg_mo"], "fl_monthly": avg["fl_mo"], "ip_monthly": avg["ip_mo"],
        "wg_blue_monthly": avg["wg_b_mo"], "fl_blue_monthly": avg["fl_b_mo"], "ip_blue_monthly": avg["ip_b_mo"],
        "wg_red_monthly": avg["wg_r_mo"], "fl_red_monthly": avg["fl_r_mo"], "ip_red_monthly": avg["ip_r_mo"],
        "wg_red_pct": avg["wg_r_pct"], "fl_red_pct": avg["fl_r_pct"], "ip_red_pct": avg["ip_r_pct"]
    }

def run_parallel_sweep():
    
    print("Generating parameter space...")
    keys, param_generator = get_sweep_configs()

    completed_batches = {
        int(f.split('_')[1].split('.')[0]) 
        for f in os.listdir(OUTPUT_DIR) 
        if f.startswith('batch_') and f.endswith('.parquet')
    }

    if completed_batches:
        last_batch = max(completed_batches)
        last_file = os.path.join(OUTPUT_DIR, f"batch_{last_batch:04d}.parquet")
        print(f"Clean-up: Removing potentially partial file {last_file}")
        if os.path.exists(last_file): 
            os.remove(last_file)
        completed_batches.remove(last_batch)

    num_completed = len(completed_batches)
    rows_to_skip = num_completed * CHUNK_SIZE

    skipped_count = 0
    if rows_to_skip > 0:
        print(f"⏩ Fast-forwarding: Skipping {rows_to_skip:,} VALID configurations...")
        for c in param_generator:
            if is_valid_config(c[7], c[2], c[1], c[4], c[5]):
                skipped_count += 1
            if skipped_count >= rows_to_skip:
                break

        print(f"⏩ Fast-forward complete. Ready to generate Batch {num_completed + 1}")

    print("🎯 Pre-filtering valid configurations...")
    valid_configs = (
        c for c in param_generator 
        if is_valid_config(c[7], c[2], c[1], c[4], c[5])
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
                batch_file = os.path.join(OUTPUT_DIR, f"batch_{batch_index:04d}.parquet")
                
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
                batch_file = os.path.join(OUTPUT_DIR, f"batch_{batch_index:04d}.parquet")
                pd.DataFrame(buffer).to_parquet(batch_file, index=False)
                count += len(buffer)

    with open("SWEEP_COMPLETE.txt", "w") as f:
        f.write("Done")
    print(f"\n✅ Sweep Complete. Total valid configs: {count}")

if __name__ == "__main__":
    run_parallel_sweep()