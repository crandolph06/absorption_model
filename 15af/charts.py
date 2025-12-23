# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_csv("ute_model_results.csv")
# df["Total_Upgrades"] = df["MQT_Students"] + df["FLUG_Students"] + df["IPUG_Students"]
# df["IP_to_Upgrades"] = df["IP_Qty"] / df["Total_Upgrades"]  # new column

# def rap_severity(row):
#     wg = row["WG_RAP_Shortfall"] > 0
#     fl = row["FL_RAP_Shortfall"] > 0
#     ip = row["IP_RAP_Shortfall"] > 0
#     if not wg and not fl and not ip:
#         return 0
#     if wg and not fl and not ip:
#         return 1
#     if wg and fl and not ip:
#         return 2
#     return 3

# df["RAP_Severity"] = df.apply(rap_severity, axis=1)

# severity_colors = {
#     0: ("green", "All pilots make RAP"),
#     1: ("yellow", "WG shortfall"),
#     2: ("orange", "WG + FL shortfall"),
#     3: ("red", "WG + FL + IP shortfall"),
# }

# def scatter_by_severity(x, y, xlabel, ylabel, title, filename):
#     plt.figure(figsize=(8, 6))
#     for sev, (color, label) in severity_colors.items():
#         subset = df[df["RAP_Severity"] == sev]
#         plt.scatter(
#             subset[x],
#             subset[y],
#             color=color,
#             label=label,
#             alpha=0.6,
#             edgecolors="k",
#             s=50
#         )
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(filename)  # <-- saves chart
#     plt.show()

# # ----------------------
# # CHARTS 1–4
# scatter_by_severity("Total_Upgrades", "IP_to_Upgrades", "Total Upgrades", "IP / Upgrades Ratio",
#                     "RAP Outcomes by IP Ratio and Upgrade Load", "Figure_1_IP_Ratio.png")

# scatter_by_severity("Total_Upgrades", "Experience_Ratio", "Total Upgrades", "Experience Ratio",
#                     "RAP Outcomes by Experience Ratio and Upgrade Load", "Figure_2_ExpRatio.png")

# scatter_by_severity("Total_Upgrades", "UTE", "Total Upgrades", "UTE",
#                     "RAP Outcomes by UTE and Upgrade Load", "Figure_3_UTE.png")

# scatter_by_severity("Total_Upgrades", "PAA", "Total Upgrades", "PAA",
#                     "RAP Outcomes by PAA and Upgrade Load", "Figure_4_PAA.png")

# # ----------------------
# # CHART 5 – Blue Sortie Equity
# grouped = df.groupby("Experience_Ratio").mean(numeric_only=True)
# plt.figure(figsize=(9, 6))
# plt.plot(grouped.index, grouped["WG_Blue_Monthly"], marker="o", label="WG")
# plt.plot(grouped.index, grouped["FL_Blue_Monthly"], marker="o", label="FL")
# plt.plot(grouped.index, grouped["IP_Blue_Monthly"], marker="o", label="IP")
# plt.xlabel("Experience Ratio")
# plt.ylabel("Blue Monthly Sorties per Pilot")
# plt.title("Blue Sortie Equity vs Experience Ratio")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("Figure_5_BlueEquity.png")
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ute_rap_model import run_model, SquadronConfig

# ----------------------------
# Parameters
# ----------------------------
total_upgrades = 20
ip_ratios = np.linspace(0.1, 0.9, 20)  # IP / Upgrade ratio
ute_values = np.arange(10, 21)         # UTE from 10 to 20

# Store results
records = []

# ----------------------------
# Generate combinations and run model
# ----------------------------
for ute in ute_values:
    for ip_ratio in ip_ratios:
        ip_qty = int(round(ip_ratio * total_upgrades))
        # Simple equal split of upgrades
        mqt = flug = ipug = int(total_upgrades / 3)

        cfg = SquadronConfig(
            ute=ute,
            paa=21,
            mqt_students=mqt,
            flug_students=flug,
            ipug_students=ipug,
            total_pilots=30,
            experience_ratio=0.6,
            ip_qty=ip_qty
        )

        try:
            results = run_model(cfg)
            rap_short = results["rap_shortfall"]
            if all(v == 0 for v in rap_short.values()):
                severity = 0  # All meet RAP
            elif rap_short["WG"] > 0 and rap_short["FL"] == 0:
                severity = 1  # WG only
            elif rap_short["WG"] > 0 and rap_short["FL"] > 0:
                severity = 2  # WG + FL
            else:
                severity = 3  # WG + FL + IP
        except:
            severity = 3  # Treat invalid configs as worst-case

        records.append({
            "IP_Ratio": ip_ratio,
            "UTE": ute,
            "RAP_Severity": severity
        })

df = pd.DataFrame(records)

# ----------------------------
# Scatterplot
# ----------------------------
severity_colors = {
    0: ("green", "All pilots make RAP"),
    1: ("yellow", "WG shortfall"),
    2: ("orange", "WG + FL shortfall"),
    3: ("red", "WG + FL + IP shortfall"),
}

plt.figure(figsize=(10, 6))
for sev, (color, label) in severity_colors.items():
    subset = df[df["RAP_Severity"] == sev]
    plt.scatter(
        subset["IP_Ratio"],
        subset["UTE"],
        color=color,
        label=label,
        alpha=0.7,
        edgecolors="k",
        s=80
    )

plt.xlabel("IP / Upgrade Ratio")
plt.ylabel("UTE (per month)")
plt.title("RAP Outcomes vs IP Ratio and UTE")
plt.legend(title="RAP Severity")
plt.grid(True)
plt.tight_layout()
plt.savefig("Figure_IP_UTE_RAP_Scatter.png")
plt.show()


