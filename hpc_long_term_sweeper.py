import itertools
from concurrent.futures import ProcessPoolExecutor
import os
import joblib
import numpy as np
import pandas as pd
from src.manning_main import setup_simulation
from src.models import PriorityMode

# --- CONFIGURATION ---
YEARS_TO_RUN = 20
OUTPUT_DIR = "outputs/long_term"
CHUNK_SIZE = 1000
BRAIN_PATH = "outputs/short_term/brains"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_valid_upg_logic(flug_start, ipug_start, asd):
    return (flug_start * asd) < ipug_start

def get_valid_long_term_configs():
    annual_intake = range(100, 370, 10)
    retention_rate = np.linspace(0.4, 0.7, 7).round(2)
    max_manning = range(50, 150, 10)
    staff_logic = [PriorityMode.RANDOM, PriorityMode.IP_FIRST, PriorityMode.FL_FIRST]
    ute_val = range(6, 21)
    flug_start = range(50, 300, 10)
    ipug_start = range (100, 450, 50)

    keys = ['intake', 'retention', 'max_man', 'staff_logic', 'ute', 'flug_start_sorties', 'ipug_start_hours']
    values = [annual_intake, retention_rate, max_manning, staff_logic, ute_val, flug_start, ipug_start]

    param_generator = itertools.product(*values)

    valid_generator = (
        params for params in param_generator
        if is_valid_upg_logic(params[5], params[6], 1.3)
    )

    return keys, valid_generator
    
def load_ai_brain(brain_path):
    if os.path.exists(brain_path):
        return joblib.load(brain_path)
    else:
        print(f'Brain not found at {brain_path}. Check path and try again.') 
        return None     

def process_single_config(args):
    annual_intake, retention_rate, max_manning, staff_logic, ute_val, flug_start_sorties, ipug_start_hours, brain = args

    try:
        sim, squadrons = setup_simulation(
            round_robin=True,
            ai_brain=brain,
            sim_upgrades=True,
            flug_window_start=flug_start_sorties,
            ipug_window_start=ipug_start_hours,
            max_manning_pct=max_manning,
            staff_priority_mode=staff_logic,
        )

        if ute_val:
            for sq in squadrons:
                sq.ute = ute_val

        sim.run_simulation(
            years_to_run=YEARS_TO_RUN,
            annual_intake=annual_intake,
            retention_rate=retention_rate,
            squadron_configs=squadrons
        )

        card = sim.get_simulation_grade_card()

        card.update({
            "input_intake": annual_intake,
            "input_retention": retention_rate,
            "inpute_max_man": max_manning,
            "input_staff_mode": staff_logic.value,
            "input_ute": ute_val,
            "input_flug_start": flug_start_sorties,
            "input_ipug_start": ipug_start_hours
        })

        return card
    
    except Exception as e:
        return None

def run_long_term_sweep():
    print("🚀 Launching Long-Term Equilibrium Sweep...")
    keys, valid_gen = get_valid_long_term_configs()
    
    brain = load_ai_brain(os.path.join(BRAIN_PATH, "hpc_sortie_brain_lite.pkl"))
    gen_with_brain = (params + (brain,) for params in valid_gen)

    buffer = []
    count = 0

    with ProcessPoolExecutor() as executor:
        for result in executor.map(process_single_config, gen_with_brain, chunksize=10):
            if result:
                buffer.append(result)

            if len(buffer) >= CHUNK_SIZE:
                df_chunk = pd.DataFrame(buffer)
                output_path = f"{OUTPUT_DIR}/long_term_batch_{count:04d}.parquet"
                df_chunk.to_parquet(output_path, index=False)
                count += 1
                buffer = []
                print(f"Saved {output_path}")

        # Final Flush
        if buffer:
            pd.DataFrame(buffer).to_parquet(f"{OUTPUT_DIR}/final_long_term_batch.parquet", index=False)

    print("✅ Long-term sweep complete.")

if __name__ == "__main__":
    run_long_term_sweep()