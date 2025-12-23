import itertools
import pandas as pd

from ute_rap_model import (
    SquadronConfig,
    run_model,
    pilot_breakdown,
    Qual
)

# -------------------------------------------------
# RAP State (8 possible outcomes)
# -------------------------------------------------
def rap_state_code(per_pilot):
    wg = per_pilot[Qual.WG]["monthly"] < 9
    fl = per_pilot[Qual.FL]["monthly"] < 8
    ip = per_pilot[Qual.IP]["monthly"] < 8
    # Bitmask: WG=1, FL=2, IP=4
    return (1 if wg else 0) + (2 if fl else 0) + (4 if ip else 0)

def rap_state_label(code):
    labels = {
        0: "All Make RAP",
        1: "WG Shortfall",
        2: "FL Shortfall",
        3: "WG + FL Shortfall",
        4: "IP Shortfall",
        5: "WG + IP Shortfall",
        6: "FL + IP Shortfall",
        7: "WG + FL + IP Shortfall",
    }
    return labels[code]

# -------------------------------------------------
# Deduplication Key
# -------------------------------------------------
def outcome_key(record):
    return (
        record["UTE"],
        record["PAA"],
        record["Experience_Ratio"],
        record["IP_Qty"],
        record["Total_Upgrades"],
        record["RAP_State_Code"],
    )

# -------------------------------------------------
# Parameter Sweep
# -------------------------------------------------
def run_parameter_sweep(output_csv="ute_model_results_deduped.csv"):
    # ----------------------
    # Parameter grids
    # ----------------------
    ute_values = list(range(10, 21))  # 10-20
    ip_qty_values = list(range(3, 13))  # 3-12
    exp_ratios = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    paa_values = [18, 21, 24, 27]

    # Total upgrades bins instead of individual MQT/FL/IP
    total_upgrades_values = list(range(12, 43))  # 12-42 total upgrades

    # Uncomment below if you want to sweep individual upgrade types instead
    # mqt_values = list(range(4, 16))
    # flug_values = list(range(4, 16))
    # ipug_values = list(range(4, 13))

    records = []
    seen = set()

    # ----------------------
    # Cartesian product
    # ----------------------
    for ute, ip_qty, exp_ratio, paa, total_upgrades in itertools.product(
        ute_values,
        ip_qty_values,
        exp_ratios,
        paa_values,
        total_upgrades_values
    ):

        # Skip invalid IP > experienced combos
        experienced = round(30 * exp_ratio)
        if ip_qty > experienced:
            continue

        # Distribute remaining upgrades proportionally (placeholder split)
        remaining_upgrades = total_upgrades - ip_qty
        if remaining_upgrades < 0:
            continue
        # Roughly split remaining between MQT and FLUG
        mqt = remaining_upgrades // 2
        flug = remaining_upgrades - mqt
        ipug = ip_qty

        cfg = SquadronConfig(
            ute=ute,
            paa=paa,
            mqt_students=mqt,
            flug_students=flug,
            ipug_students=ipug,
            total_pilots=30,
            experience_ratio=exp_ratio,
            ip_qty=ip_qty
        )

        try:
            results = run_model(cfg)
        except Exception as e:
            print("FAILED CONFIG:", cfg)
            raise

        pilots = pilot_breakdown(cfg)
        per_pilot = results["per_pilot"]

        ip_to_upgrade = ip_qty / total_upgrades if total_upgrades > 0 else 0
        rap_code = rap_state_code(per_pilot)
        rap_label = rap_state_label(rap_code)

        record = {
            "UTE": ute,
            "PAA": paa,
            "IP_Qty": ip_qty,
            "Experience_Ratio": exp_ratio,
            "MQT_Students": mqt,
            "FLUG_Students": flug,
            "IPUG_Students": ipug,
            "Total_Upgrades": total_upgrades,
            "IP_to_Upgrade_Ratio": ip_to_upgrade,

            "WG_Count": pilots[Qual.WG],
            "FL_Count": pilots[Qual.FL],
            "IP_Count": pilots[Qual.IP],

            "Annual_Sortie_Capacity": results["capacity"],

            "WG_Monthly": per_pilot[Qual.WG]["monthly"],
            "FL_Monthly": per_pilot[Qual.FL]["monthly"],
            "IP_Monthly": per_pilot[Qual.IP]["monthly"],

            "WG_Blue_Monthly": per_pilot[Qual.WG]["blue_monthly"],
            "FL_Blue_Monthly": per_pilot[Qual.FL]["blue_monthly"],
            "IP_Blue_Monthly": per_pilot[Qual.IP]["blue_monthly"],

            "WG_RAP_Shortfall": results["rap_shortfall"]["WG"],
            "FL_RAP_Shortfall": results["rap_shortfall"]["FL"],
            "IP_RAP_Shortfall": results["rap_shortfall"]["IP"],

            "RAP_State_Code": rap_code,
            "RAP_State_Label": rap_label,
        }

        key = outcome_key(record)
        if key in seen:
            continue
        seen.add(key)
        records.append(record)

    # ----------------------
    # Write CSV
    # ----------------------
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df):,} unique rows to {output_csv}")


if __name__ == "__main__":
    run_parameter_sweep()
