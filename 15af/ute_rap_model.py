from dataclasses import dataclass
from enum import Enum

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
    side: str
    sorties: int
    student_qual: str
    wg_required: int
    fl_required: int
    ip_required: int

@dataclass(frozen=True)
class ContinuationBucket:
    name: str
    min_qual: Qual
    side: str
    fraction: float

@dataclass
class ContinuationProfile:
    name: str
    buckets: list[ContinuationBucket]

# ----------------------
# Upgrade Syllabus with Crew Requirements
# ----------------------
COMMON_UPGRADE_SYLLABUS = [
    # MQT: student (WG), instructor (IP), FL, WG
    SyllabusBucket("MQT", "Blue", 1, wg_required=2, fl_required=1, ip_required=1),
    # FLUG: student (WG), instructor (IP), FL, WG
    SyllabusBucket("FLUG", "Blue", 1, wg_required=2, fl_required=1, ip_required=1),
    # IPUG: student (FL), instructor (IP), FL, WG
    SyllabusBucket("IPUG", "Blue", 1, wg_required=1, fl_required=2, ip_required=1),
]

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
def pilot_breakdown(cfg: SquadronConfig) -> dict[Qual, int]:
    experienced = round(cfg.total_pilots * cfg.experience_ratio)
    ip = cfg.ip_qty
    if ip > experienced:
        raise ValueError("IP quantity cannot exceed experienced pilots")
    fl = experienced - ip
    wg = cfg.total_pilots - experienced
    return {Qual.WG: wg, Qual.FL: fl, Qual.IP: ip}

def total_monthly_capacity(cfg: SquadronConfig) -> float:
    return cfg.ute * cfg.paa

def total_upgrade_students(cfg: SquadronConfig) -> int:
    return cfg.mqt_students + cfg.flug_students + cfg.ipug_students

def syllabus_demand(syllabus: list[SyllabusBucket], student_counts: dict[str,int]) -> dict[Qual,int]:
    """
    Compute total sorties demand per pilot type based on upgrade counts and crew requirements
    """
    demand = {Qual.WG: 0, Qual.FL: 0, Qual.IP: 0}
    for bucket in syllabus:
        count = student_counts.get(bucket.name, 0)
        demand[Qual.WG] += bucket.wg_required * count * bucket.sorties
        demand[Qual.FL] += bucket.fl_required * count * bucket.sorties
        demand[Qual.IP] += bucket.ip_required * count * bucket.sorties
    return demand

def ct_demand(profile: ContinuationProfile, remaining_sorties: int) -> dict[Qual,int]:
    demand = {Qual.WG:0, Qual.FL:0, Qual.IP:0}
    for b in profile.buckets:
        qty = int(remaining_sorties * b.fraction)
        for q in [q for q in Qual if q.value >= b.min_qual.value]:
            demand[q] += qty
    return demand

# ----------------------
# Allocator
# ----------------------
def allocate_sorties(demand: dict[Qual,int], pilots: dict[Qual,int], capacity: int) -> dict[Qual,dict[str,float]]:
    allocation = {Qual.WG: {"Blue": 0, "Red": 0}, Qual.FL: {"Blue": 0, "Red": 0}, Qual.IP: {"Blue": 0, "Red": 0}}
    remaining_capacity = capacity
    for qual, sorties_needed in demand.items():
        flown = min(sorties_needed, pilots[qual] * 999, remaining_capacity)
        allocation[qual]["Blue"] += flown
        remaining_capacity -= flown
    return allocation

def adjust_for_total(allocation: dict[Qual, dict[str,float]], total_capacity: int):
    total_allocated = sum(sum(s.values()) for s in allocation.values())
    remainder = total_capacity - total_allocated
    for qual in [Qual.IP, Qual.FL, Qual.WG]:
        for side in ["Blue", "Red"]:
            if remainder <= 0:
                break
            allocation[qual][side] += 1
            remainder -= 1
        if remainder <= 0:
            break

# ----------------------
# Per-Pilot Sorties
# ----------------------
def sorties_per_pilot(allocation, pilots):
    result = {}
    for qual in allocation:
        total_annual = sum(allocation[qual].values())
        blue_annual = allocation[qual].get("Blue",0)
        if pilots[qual] > 0:
            result[qual] = {
                "annual": total_annual / pilots[qual],
                "monthly": (total_annual / pilots[qual]) / 12,
                "blue_monthly": (blue_annual / pilots[qual]) / 12
            }
        else:
            result[qual] = {"annual":0,"monthly":0,"blue_monthly":0}
    return result

# ----------------------
# RAP / Willsbach
# ----------------------
def rap_required_sorties(pilots: dict[Qual,int]) -> int:
    return pilots[Qual.WG]*9 + pilots[Qual.FL]*8 + pilots[Qual.IP]*8

def rap_shortfall(per_pilot):
    return {
        "WG": max(0, 9 - per_pilot[Qual.WG]["monthly"]),
        "FL": max(0, 8 - per_pilot[Qual.FL]["monthly"]),
        "IP": max(0, 8 - per_pilot[Qual.IP]["monthly"]),
    }

def willsbach_shortfall(per_pilot):
    return {qual.name: max(0, 12 - data["monthly"]) for qual, data in per_pilot.items()}

def rap_state(per_pilot):
    wg_ok = per_pilot[Qual.WG]["monthly"] >= 9
    fl_ok = per_pilot[Qual.FL]["monthly"] >= 8
    ip_ok = per_pilot[Qual.IP]["monthly"] >= 8
    state_code = (4 if wg_ok else 0) + (2 if fl_ok else 0) + (1 if ip_ok else 0)
    state_label = {0:"None",1:"IP only",2:"FL only",3:"FL + IP",
                   4:"WG only",5:"WG + IP",6:"WG + FL",7:"WG + FL + IP"}[state_code]
    return {"rap_state_code": state_code, "rap_state_label": state_label}

def rap_recommendations(cfg: SquadronConfig, pilots: dict[Qual,int]) -> dict:
    required = rap_required_sorties(pilots)
    current_capacity = total_monthly_capacity(cfg) * 12
    shortfall = max(0, required - current_capacity)
    return {"rap_required_sorties": required, "current_capacity":current_capacity,
            "sortie_shortfall":shortfall,
            "ute_required_for_rap":required/cfg.paa,
            "additional_ute": max(0, required/cfg.paa - cfg.ute),
            "paa_required_for_rap":required/cfg.ute,
            "additional_paa": max(0, required/cfg.ute - cfg.paa)}

# ----------------------
# Main Model
# ----------------------
def run_model(cfg: SquadronConfig):
    pilots = pilot_breakdown(cfg)
    total_capacity = int(total_monthly_capacity(cfg) * 12)

    # Upgrade phase
    student_counts = {"MQT": cfg.mqt_students, "FLUG": cfg.flug_students, "IPUG": cfg.ipug_students}
    upgrade_demand = syllabus_demand(COMMON_UPGRADE_SYLLABUS, student_counts)
    upgrade_allocation = allocate_sorties(upgrade_demand, pilots, total_capacity)
    used_capacity = sum(sum(v.values()) for v in upgrade_allocation.values())
    remaining_capacity = max(0, total_capacity - used_capacity)

    # Continuation phase
    continuation_demand_dict = ct_demand(CONTINUATION_PROFILE, remaining_capacity)
    continuation_allocation = allocate_sorties(continuation_demand_dict, pilots, remaining_capacity)

    # Combine allocations
    total_allocation = {qual:{side:upgrade_allocation[qual].get(side,0)+continuation_allocation[qual].get(side,0)
                              for side in ["Blue","Red"]} for qual in Qual}
    adjust_for_total(total_allocation, total_capacity)
    per_pilot = sorties_per_pilot(total_allocation, pilots)
    rap_info = rap_state(per_pilot)
    rap_rec = rap_recommendations(cfg, pilots)

    return {
        "capacity": total_capacity,
        "per_pilot": per_pilot,
        "rap_shortfall": rap_shortfall(per_pilot),
        "willsbach_shortfall": willsbach_shortfall(per_pilot),
        "rap_state_code": rap_info["rap_state_code"],
        "rap_state_label": rap_info["rap_state_label"],
        "rap_recommendations": rap_rec
    }

# ----------------------
# Example Run
# ----------------------
if __name__ == "__main__":
    cfg = SquadronConfig(
        ute=10,
        paa=21,
        mqt_students=7,
        flug_students=4,
        ipug_students=4,
        total_pilots=30,
        experience_ratio=0.6,
        ip_qty=5
    )
    results = run_model(cfg)
    print(results)

