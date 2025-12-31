import pandas as pd
import numpy as np
import os
from src.engine import run_phase_simulation, create_pilots
from src.models import SquadronConfig, Qual, Upgrade
from src.rap_state import rap_assess, rap_state_code, rap_state_label


def run_research_sweep():
    # Define your ranges based on your requirements
    ute_values = list(range(10, 21))
    ip_qty_values = list(range(3, 8))
    exp_ratios = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # Slimmed for speed, can expand later
    paa_values = [18, 24]
    # mqt_students = list(range(1, 10))
    mqt_students = [1, 3]
    # flug_students = list(range(1, 10))
    flug_students = [1, 3]
    # ipug_students = list(range(1, 10))
    ipug_students = [1, 3]
    total_pilots = [30, 35]

    results = []
    
    # We will assume a 120-day phase (4 months)
    PHASE_DAYS = 120
    ITERATIONS_PER_CONFIG = 3 

    total_combos = len(ute_values) * len(ip_qty_values) * len(exp_ratios) * len(paa_values) * len(mqt_students) * len(flug_students) * len(ipug_students)
    print(f"Starting sweep of {total_combos} configurations...")

    count = 0
    for paa in paa_values:
        for ute in ute_values:
            for ip_q in ip_qty_values:
                for exp in exp_ratios:
                    for mqt in mqt_students:
                        for flug in flug_students:
                            for ipug in ipug_students:
                                for total in total_pilots:

                                    cfg = SquadronConfig(
                                        paa=paa,
                                        ute=ute,
                                        experience_ratio=exp,
                                        ip_qty=ip_q,
                                        mqt_students=mqt,
                                        flug_students=flug,
                                        ipug_students=ipug,
                                        phase_length_days=PHASE_DAYS,
                                        total_pilots=total
                                    )

                                    for i in range(ITERATIONS_PER_CONFIG):
                                        pilots = create_pilots(cfg)
                                        # Run the simulation
                                        final_pilots = run_phase_simulation(cfg, pilots, allocation_noise=0.0)

                                        # --- COLLECT METRICS ---
                                        
                                        # RQ1: RAP Success (Non-MQT pilots making 8-9 sorties/month)
                                        rap_dict, blue_rap_dict, red_dict = rap_assess(final_pilots)
                                        rap_code = rap_state_code(rap_dict)
                                        blue_rap_code = rap_state_code(blue_rap_dict)
                                        rap_label = rap_state_label(rap_code)
                                        blue_rap_label = rap_state_label(blue_rap_code)

                                        results.append({
                                            "paa": paa,
                                            "ute": ute,
                                            "total_capacity": cfg.paa * cfg.ute,
                                            "exp_ratio": exp,
                                            "ip_qty": ip_q,
                                            "mqt_qty": mqt,
                                            "flug_qty": flug,
                                            "ipug_qty": ipug,
                                            "rap_state_code": rap_code,
                                            "rap_state_label": rap_label,
                                            "blue_rap_state_code": blue_rap_code,
                                            "blue_rap_state_label": blue_rap_label,
                                            "mqt_monthly": rap_dict["MQT"][1],
                                            "wg_monthly": rap_dict["WG"][1],
                                            "fl_monthly": rap_dict["FL"][1],
                                            "ip_monthly": rap_dict["IP"][1],
                                            "wg_blue_monthly": blue_rap_dict["WG"][1],
                                            "fl_blue_monthly": blue_rap_dict["FL"][1],
                                            "ip_blue_monthly": blue_rap_dict["IP"][1],
                                            "wg_red_pct": red_dict["WG"],
                                            "fl_red_pct": red_dict["FL"],
                                            "ip_red_pct": red_dict["IP"]
                                        })
                                    
                                    count += 1
                                    if count % 100 == 0:
                                        print(f"Processed {count}/{total_combos} configs...")

    # Save to CSV
    os.makedirs("outputs", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("outputs/research_data.csv", index=False)
    print("Done! Data saved to outputs/research_data.csv")

if __name__ == "__main__":
    run_research_sweep()