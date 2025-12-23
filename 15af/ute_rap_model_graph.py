from dataclasses import dataclass
from enum import Enum
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------
# Config / Enums
# ----------------------

@dataclass
class SquadronConfig:
    ute: float
    paa: int
    mqt_students: int
    flug_students: int
    ipug_students: int
    total_pilots: int
    experience_ratio: float
    ip_qty: int

class Qual(Enum):
    WG = 1
    FL = 2
    IP = 3

# ----------------------
# Syllabus / Continuation
# ----------------------

@dataclass(frozen=True)
class SyllabusBucket:
    name: str
    min_qual: Qual
    side: str
    sorties: int

@dataclass
class Syllabus:
    name: str
    buckets: list[SyllabusBucket]

@dataclass(frozen=True)
class ContinuationBucket:
    name: str
    min_qual: Qual
    side: str
    fraction: float  # fraction of leftover sorties

@dataclass
class ContinuationProfile:
    name: str
    buckets: list[ContinuationBucket]

# ----------------------
# Example Syllabus / Continuation Profile
# ----------------------

COMMON_UPGRADE_SYLLABUS = Syllabus(
    name="Common Upgrade Syllabus",
    buckets=[
        SyllabusBucket("Student Sorties", Qual.WG, "Blue", 14),
        SyllabusBucket("IP Sorties", Qual.IP, "Blue", 14),
        SyllabusBucket("Blue FL DS", Qual.FL, "Blue", 8),
        SyllabusBucket("Blue WG DS", Qual.WG, "Blue", 8),
        SyllabusBucket("Red FL DS", Qual.FL, "Red", 7),
        SyllabusBucket("Red WG DS", Qual.WG, "Red", 7),
    ]
)

CONTINUATION_PROFILE = ContinuationProfile(
    name="Continuation Training",
    buckets=[
        ContinuationBucket("Blue FL", Qual.FL, "Blue", 0.36),
        ContinuationBucket("Blue WG", Qual.WG, "Blue", 0.36),
        ContinuationBucket("Red FL", Qual.FL, "Red", 0.14),
        ContinuationBucket("Red WG", Qual.WG, "Red", 0.14),
    ]
)

# ----------------------
# Helper Functions
# ----------------------

def syllabus_demand(syllabus: Syllabus, total_students: int) -> dict[SyllabusBucket, float]:
    return {bucket: bucket.sorties * total_students for bucket in syllabus.buckets}

def pilot_breakdown(cfg: SquadronConfig) -> dict[Qual, int]:
    ip = cfg.ip_qty
    experienced = int(cfg.total_pilots * cfg.experience_ratio)  # total IP + FL
    fl = max(0, experienced - ip)
    wg = cfg.total_pilots - ip - fl
    return {Qual.WG: wg, Qual.FL: fl, Qual.IP: ip}

def total_monthly_capacity(cfg: SquadronConfig) -> float:
    return cfg.ute * cfg.paa

def total_upgrade_students(cfg: SquadronConfig) -> int:
    return cfg.mqt_students + cfg.flug_students + cfg.ipug_students

def ct_demand(profile: ContinuationProfile, remaining_sorties: float) -> dict[SyllabusBucket, float]:
    return {SyllabusBucket(b.name, b.min_qual, b.side, 0): remaining_sorties * b.fraction for b in profile.buckets}

def eligible_quals(min_qual: Qual):
    return [q for q in Qual if q.value >= min_qual.value]

# ----------------------
# Allocator
# ----------------------

def allocate_sorties(demand: dict[SyllabusBucket, float], pilots: dict[Qual, int], capacity: float) -> dict[Qual, dict[str, float]]:
    allocation = {Qual.WG: {"Blue": 0.0, "Red": 0.0},
                  Qual.FL: {"Blue": 0.0, "Red": 0.0},
                  Qual.IP: {"Blue": 0.0, "Red": 0.0}}

    remaining_capacity = capacity

    for bucket, sorties in demand.items():
        remaining = min(sorties, remaining_capacity)

        for qual in eligible_quals(bucket.min_qual):
            if remaining_capacity <= 0:
                break
            flown = min(remaining, pilots[qual] * 999)
            allocation[qual][bucket.side] += flown
            remaining -= flown
            remaining_capacity -= flown
            if remaining <= 0:
                break

    return allocation

# ----------------------
# Adjust allocations to match total exactly
# ----------------------

def adjust_for_total(allocation: dict[Qual, dict[str, float]], total_capacity: float):
    total_allocated = sum(sum(s.values()) for s in allocation.values())
    remainder = total_capacity - total_allocated
    # Assign leftover sorties starting from IP -> FL -> WG
    for qual in [Qual.IP, Qual.FL, Qual.WG]:
        for side in ["Blue", "Red"]:
            if remainder <= 0:
                return
            allocation[qual][side] += 1
            remainder -= 1
            if remainder <= 0:
                return

# ----------------------
# Per-pilot calculation
# ----------------------

def sorties_per_pilot(allocation: dict[Qual, dict[str, float]], pilots: dict[Qual,int]) -> dict[Qual, dict[str,float]]:
    result = {}
    for qual in allocation:
        total = sum(allocation[qual].values())
        blue = allocation[qual]["Blue"]
        result[qual] = {
            "total": total / pilots[qual] / 12 if pilots[qual] else 0,
            "blue": blue / pilots[qual] / 12 if pilots[qual] else 0
        }
    return result

# ----------------------
# RAP / Willsbach
# ----------------------

def rap_required_sorties(pilots: dict[Qual,int]) -> int:
    return pilots[Qual.WG]*9 + pilots[Qual.FL]*8 + pilots[Qual.IP]*8

def rap_recommendations(cfg: SquadronConfig, pilots: dict[Qual,int]) -> dict:
    required = rap_required_sorties(pilots)
    current_capacity = total_monthly_capacity(cfg) * 12
    shortfall = max(0, required - current_capacity)
    ute_required = required / cfg.paa
    paa_required = required / cfg.ute
    return {
        "rap_required_sorties": required,
        "current_capacity": current_capacity,
        "sortie_shortfall": shortfall,
        "ute_required_for_rap": ute_required,
        "additional_ute": max(0, ute_required - cfg.ute),
        "paa_required_for_rap": paa_required,
        "additional_paa": max(0, paa_required - cfg.paa)
    }

def rap_shortfall(sorties_per_pilot):
    return {
        "WG": max(0, 9 - sorties_per_pilot[Qual.WG]["total"]),
        "FL": max(0, 8 - sorties_per_pilot[Qual.FL]["total"]),
        "IP": max(0, 8 - sorties_per_pilot[Qual.IP]["total"]),
    }

def willsbach_shortfall(sorties_per_pilot):
    return {qual.name: max(0, 12 - data["total"]) for qual,data in sorties_per_pilot.items()}

# ----------------------
# Main Model
# ----------------------

def run_model(cfg: SquadronConfig):
    pilots = pilot_breakdown(cfg)
    total_capacity = cfg.ute * cfg.paa * 12

    # Upgrade
    upgrade_students = total_upgrade_students(cfg)
    upgrade_demand = syllabus_demand(COMMON_UPGRADE_SYLLABUS, upgrade_students)
    upgrade_allocation = allocate_sorties(upgrade_demand, pilots, total_capacity)
    used_capacity = sum(sum(v.values()) for v in upgrade_allocation.values())
    remaining_capacity = max(0, total_capacity - used_capacity)

    # Continuation
    continuation_demand = ct_demand(CONTINUATION_PROFILE, remaining_capacity)
    continuation_allocation = allocate_sorties(continuation_demand, pilots, remaining_capacity)

    # Combine allocations
    total_allocation = {
        qual: {
            side: upgrade_allocation[qual].get(side,0) + continuation_allocation[qual].get(side,0)
            for side in ["Blue","Red"]
        } for qual in Qual
    }

    # Adjust for rounding to ensure total = total_capacity
    adjust_for_total(total_allocation, total_capacity)

    # Per-pilot
    per_pilot = sorties_per_pilot(total_allocation, pilots)
    rap_rec = rap_recommendations(cfg, pilots)

    return {
        "capacity": total_capacity,
        "per_pilot": per_pilot,
        "rap_shortfall": rap_shortfall(per_pilot),
        "willsbach_shortfall": willsbach_shortfall(per_pilot),
        "rap_recommendations": rap_rec
    }

# ----------------------
# Scenario Grid and Graphing
# ----------------------

def run_scenarios():
    experience_ratios = [0.3, 0.5, 0.7]
    ip_counts = [2, 4, 6]
    upgrade_totals = [5, 10, 15]

    records = []

    for exp_ratio in experience_ratios:
        for ip_qty in ip_counts:
            for upgrades in upgrade_totals:
                cfg = SquadronConfig(
                    ute=12,
                    paa=10,
                    mqt_students=upgrades//3,
                    flug_students=upgrades//3,
                    ipug_students=upgrades - 2*(upgrades//3),
                    total_pilots=20,
                    experience_ratio=exp_ratio,
                    ip_qty=ip_qty
                )
                res = run_model(cfg)
                record = {
                    "experience_ratio": exp_ratio,
                    "ip_qty": ip_qty,
                    "upgrade_total": upgrades,
                    "WG_per_pilot": res["per_pilot"][Qual.WG]["total"],
                    "FL_per_pilot": res["per_pilot"][Qual.FL]["total"],
                    "IP_per_pilot": res["per_pilot"][Qual.IP]["total"],
                    "WG_rap_shortfall": res["rap_shortfall"]["WG"],
                    "FL_rap_shortfall": res["rap_shortfall"]["FL"],
                    "IP_rap_shortfall": res["rap_shortfall"]["IP"],
                    "total_capacity": res["capacity"]
                }
                records.append(record)

    df = pd.DataFrame(records)
    return df

def plot_results(df):
    # Lineplot: per-pilot sorties vs experience ratio
    plt.figure(figsize=(10,6))
    sns.lineplot(data=df, x="experience_ratio", y="WG_per_pilot",
                 hue="ip_qty", style="upgrade_total", markers=True)
    plt.title("WG Per-Pilot Monthly Sorties vs Experience Ratio")
    plt.ylabel("Sorties per pilot per month")
    plt.show()

    # Similar for FL
    plt.figure(figsize=(10,6))
    sns.lineplot(data=df, x="experience_ratio", y="FL_per_pilot",
                 hue="ip_qty", style="upgrade_total", markers=True)
    plt.title("FL Per-Pilot Monthly Sorties vs Experience Ratio")
    plt.ylabel("Sorties per pilot per month")
    plt.show()

    # Heatmap for FL RAP shortfall
    pivot = df.pivot_table(index="ip_qty", columns="experience_ratio", values="FL_rap_shortfall")
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Reds")
    plt.title("FL RAP Shortfall by IP Count and Experience Ratio")
    plt.ylabel("IP Count")
    plt.xlabel("Experience Ratio")
    plt.show()

# ----------------------
# Run
# ----------------------

if __name__ == "__main__":
    df_results = run_scenarios()
    plot_results(df_results)
