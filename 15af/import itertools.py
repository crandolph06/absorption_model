import itertools
import pandas as pd

from ute_rap_model import(
    SquadronConfig,
    run_model,
    pilot_breakdown,
    Qual
)

def run_parameter_sweep(output_csv="ute_model_results.csv"):
    # ----------------------
    # Parameter grids
    # ----------------------
    ute_values = [10, 11, 12, 13, 14, 15]
    ip_qty_values = [3, 4, 5, 6, 7, 8]
    exp_ratios = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    mqt_values = [4, 5, 6, 7, 8, 9, 10]
    flug_values = [4, 5, 6, 7, 8, 9, 10]
    ipug_values = [4, 5, 6, 7, 8, 9, 10]
    paa_values = [18, 21, 24, 27]

    records = []

    # ----------------------
    # Cartesian product
    # ----------------------
    for (
        ute, ip_qty, exp_ratio,
        mqt, flug, ipug, paa
    ) in itertools.product(
        ute_values,
        ip_qty_values,
        exp_ratios,
        mqt_values,
        flug_values,
        ipug_values,
        paa_values
    ):

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
            # Catch invalid configs (e.g. IPs > experienced)
            continue

        pilots = pilot_breakdown(cfg)
        per_pilot = results["per_pilot"]

        record = {
            # --- Inputs ---
            "UTE": ute,
            "PAA": paa,
            "IP_Qty": ip_qty,
            "Experience_Ratio": exp_ratio,
            "MQT_Students": mqt,
            "FLUG_Students": flug,
            "IPUG_Students": ipug,

            # --- Pilot counts ---
            "WG_Count": pilots[Qual.WG],
            "FL_Count": pilots[Qual.FL],
            "IP_Count": pilots[Qual.IP],

            # --- Capacity ---
            "Annual_Capacity": results["capacity"],

            # --- Per-pilot monthly ---
            "WG_Monthly": per_pilot[Qual.WG]["total"],
            "FL_Monthly": per_pilot[Qual.FL]["total"],
            "IP_Monthly": per_pilot[Qual.IP]["total"],

            # --- Blue monthly ---
            "WG_Blue_Monthly": per_pilot[Qual.WG]["blue"],
            "FL_Blue_Monthly": per_pilot[Qual.FL]["blue"],
            "IP_Blue_Monthly": per_pilot[Qual.IP]["blue"],

            # --- RAP shortfall ---
            "WG_RAP_Shortfall": results["rap_shortfall"]["WG"],
            "FL_RAP_Shortfall": results["rap_shortfall"]["FL"],
            "IP_RAP_Shortfall": results["rap_shortfall"]["IP"],

            # --- Willsbach ---
            "WG_Willsbach_Shortfall": results["willsbach_shortfall"]["WG"],
            "FL_Willsbach_Shortfall": results["willsbach_shortfall"]["FL"],
            "IP_Willsbach_Shortfall": results["willsbach_shortfall"]["IP"],
        }

        records.append(record)

    # ----------------------
    # Write CSV
    # ----------------------
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df):,} rows to {output_csv}")


if __name__ == "__main__":
    run_parameter_sweep()
