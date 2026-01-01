import pandas as pd
import numpy as np
import os
from src.engine import run_phase_simulation, create_pilots
from src.models import SquadronConfig, Qual, Upgrade
from src.rap_state import rap_assess, rap_state_code, rap_state_label

def run_research_sweep(average_iterations=True):
    # --- RANGES ---
    ute_values = list(range(10, 21))
    ip_qty_values = list(range(3, 8))
    exp_ratios = [0.3, 0.4, 0.5, 0.6, 0.7]
    paa_values = [18, 21, 24]
    mqt_students = [2, 4, 6, 8, 10]
    flug_students = [2, 4, 6, 8, 10]
    ipug_students = [2, 4, 6, 8, 10]
    total_pilots = [25, 30, 35, 40]

    PHASE_DAYS = 120
    ITERATIONS_PER_CONFIG = 3 
    OUTPUT_FILE = "outputs/research_data.csv"

    # 1. PREPARE THE FILE (Header includes total_capacity now)
    os.makedirs("outputs", exist_ok=True)
    cols = [
        "paa", "ute", "total_capacity", "exp_ratio", "ip_qty", "total_pilots", 
        "mqt_qty", "flug_qty", "ipug_qty", "rap_state_code", "rap_state_label", 
        "blue_rap_state_code", "blue_rap_state_label", "mqt_monthly", "wg_monthly", 
        "fl_monthly", "ip_monthly", "wg_blue_monthly", "fl_blue_monthly", 
        "ip_blue_monthly", "wg_red_monthly", "fl_red_monthly", "ip_red_monthly", 
        "wg_red_pct", "fl_red_pct", "ip_red_pct"
    ]
    pd.DataFrame(columns=cols).to_csv(OUTPUT_FILE, index=False)

    total_combos = len(ute_values) * len(ip_qty_values) * len(exp_ratios) * len(paa_values) * len(mqt_students) * len(flug_students) * len(ipug_students) * len(total_pilots)
    print(f"Starting sweep of {total_combos} configs... Live-writing to {OUTPUT_FILE}")

    count = 0
    for paa in paa_values:
        for ute in ute_values:
            for ip_q in ip_qty_values:
                for exp in exp_ratios:
                    for mqt in mqt_students:
                        for flug in flug_students:
                            for ipug in ipug_students:
                                for total in total_pilots:
                                    config_results = []
                                    cfg = SquadronConfig(
                                        paa=paa, ute=ute, experience_ratio=exp, ip_qty=ip_q,
                                        mqt_students=mqt, flug_students=flug, ipug_students=ipug,
                                        phase_length_days=PHASE_DAYS, total_pilots=total
                                    )

                                    for i in range(ITERATIONS_PER_CONFIG):
                                        try:
                                            pilots = create_pilots(cfg)
                                            final_pilots = run_phase_simulation(cfg, pilots, allocation_noise=0.0)
                                            rap_dict, blue_rap_dict, red_dict = rap_assess(final_pilots)

                                            r_code = rap_state_code(rap_dict)
                                            b_code = rap_state_code(blue_rap_dict)

                                            current_result = {
                                                "paa": paa, "ute": ute, 
                                                "total_capacity": cfg.paa * cfg.ute * (PHASE_DAYS / 30),
                                                "exp_ratio": exp, "ip_qty": ip_q, "total_pilots": total,
                                                "mqt_qty": mqt, "flug_qty": flug, "ipug_qty": ipug,
                                                "rap_state_code": r_code, "rap_state_label": rap_state_label(r_code),
                                                "blue_rap_state_code": b_code, "blue_rap_state_label": rap_state_label(b_code),
                                                "mqt_monthly": rap_dict["MQT"][1], "wg_monthly": rap_dict["WG"][1],
                                                "fl_monthly": rap_dict["FL"][1], "ip_monthly": rap_dict["IP"][1],
                                                "wg_blue_monthly": blue_rap_dict["WG"][1], 
                                                "fl_blue_monthly": blue_rap_dict["FL"][1], 
                                                "ip_blue_monthly": blue_rap_dict["IP"][1],
                                                "wg_red_monthly": red_dict["WG"][1], 
                                                "fl_red_monthly": red_dict["FL"][1], 
                                                "ip_red_monthly": red_dict["IP"][1], 
                                                "wg_red_pct": red_dict["WG"][0], 
                                                "fl_red_pct": red_dict["FL"][0], 
                                                "ip_red_pct": red_dict["IP"][0]
                                            }
                                            config_results.append(current_result)
                                        except ValueError:
                                            break

                                    # --- AGGREGATION & WRITE ---
                                    if config_results:
                                        if average_iterations:
                                            temp_df = pd.DataFrame(config_results)
                                            # Average numbers, flip to row
                                            final_rows = temp_df.mean(numeric_only=True).to_frame().T
                                            # Re-label strings
                                            final_rows["rap_state_label"] = rap_state_label(int(round(final_rows["rap_state_code"].iloc[0])))
                                            final_rows["blue_rap_state_label"] = rap_state_label(int(round(final_rows["blue_rap_state_code"].iloc[0])))
                                        else:
                                            final_rows = pd.DataFrame(config_results)

                                        # Ensure column order matches the CSV header
                                        final_rows = final_rows.reindex(columns=cols)
                                        final_rows.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
                            
                                    count += 1
                                    if count % 100 == 0:
                                        print(f"Processed {count}/{total_combos} configs...")
    
    print(f"Done! Final data available at {OUTPUT_FILE}")

if __name__ == "__main__":
    # One call only
    run_research_sweep(average_iterations=True)