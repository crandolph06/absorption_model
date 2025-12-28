from dataclasses import dataclass
from enum import Enum
from typing import List
import random

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
    phase_length_days: int = 120  # ~1/3 year or 4 months

class Qual(Enum):
    WG = 'WG'
    FL = 'FL'
    IP = 'IP'

class Upgrade(Enum):
    NONE = 'None'
    MQT = 'MQT'
    FLUG = 'FLUG'
    IPUG = 'IPUG'

class EventType(Enum):
    SORTIE = "sortie"
    SIM = "sim"

# ----------------------
# Pilot Entity
# ----------------------
@dataclass
class Pilot:
    qual: Qual
    upgrade: Upgrade = Upgrade.NONE
    sortie_monthly: float = 0
    sim_monthly: float = 0
    total_monthly: float = 0
    sortie_blue_monthly: float = 0
    sortie_red_monthly: float = 0

    wg_rap: float = 0
    mqt_rap: float = 0
    fl_rap: float = 0
    flug_rap: float = 0
    ip_rap: float = 0
    ipug_rap: float = 0

    target_sorties: float = 0
    rap_shortfall: float = 0
    
    def update_total(self):
        self.total_monthly = self.sortie_monthly + self.sim_monthly
        self.rap_shortfall = max(0, self.target_sorties - self.total_monthly)
# ----------------------
# Syllabus Bucket
# ----------------------
@dataclass(frozen=True)
# class SyllabusBucket:
#     name: str
#     event_type: str         # "sortie" or "sim"
#     student_qual: str       # "WG" or "FL"
#     student_sortie: int
#     blue_wg: int
#     blue_fl: int
#     blue_ip: int
#     red_wg: int
#     red_fl: int

class SyllabusEvent: 
    name: str
    event_type: EventType
    seat_type: Qual
    num_student: int = 1
    num_instructor: int = 1
    num_blue_wg: int = 0
    num_blue_fl: int = 0
    num_red_wg: int = 0
    num_red_fl: int = 0

    def total_slots(self):
        return self.num_student + self.num_instructor + self.num_blue_wg + self.num_blue_fl + self.num_red_wg + self.num_red_fl

class UpgradeProgram:
    name: str # "MQT", "FLUG", "IPUG"
    syllabus: List[SyllabusEvent]
    student_qual: Qual
    num_students: int

# ----------------------
# MQT Syllabus
# ----------------------
MQT_SYLLABUS = [
    SyllabusEvent("OBFM", EventType.SORTIE, Qual.WG, 1,1,0,0,0,0),
    SyllabusEvent("DBFM", EventType.SORTIE, Qual.WG, 1,1,0,0,0,0),
    SyllabusEvent("HABFM", EventType.SORTIE, Qual.WG, 1,1,0,0,0,0),
    SyllabusEvent("2vX TI", EventType.SIM, Qual.WG, 1,1,0,0,0,0),
    SyllabusEvent("ACM", EventType.SORTIE, Qual.WG, 1,1,0,0,1,1),
    SyllabusEvent("4vX TI", EventType.SIM, Qual.WG, 1,1,1,1,0,0),
    SyllabusEvent("DCA", EventType.SIM, Qual.WG, 1,1,1,1,0,0),
    SyllabusEvent("DCA", EventType.SORTIE, Qual.WG, 1,1,1,1,2,2),
    SyllabusEvent("U-SEAD", EventType.SIM, Qual.WG, 1,1,1,1,0,0),
    SyllabusEvent("D-SEAD", EventType.SIM, Qual.WG, 1,1,1,1,0,0),
    SyllabusEvent("U-SEAD", EventType.SORTIE, Qual.WG, 1,1,2,2,0,0),
    SyllabusEvent("O-SEAD", EventType.SIM, Qual.WG, 1,1,1,1,0,0),
    SyllabusEvent("O-SEAD", EventType.SORTIE, Qual.WG, 1,1,2,2,2,2),
    SyllabusEvent("O-AI", EventType.SIM, Qual.WG, 1,1,1,1,0,0),
    SyllabusEvent("CAS", EventType.SORTIE, Qual.WG, 1,1,0,0,0,0),
    SyllabusEvent("BSA", EventType.SORTIE, Qual.WG, 1,1,1,1,0,0),
]

# ----------------------
# FLUG Syllabus (same as MQT)
# ----------------------
FLUG_SYLLABUS = [
    SyllabusEvent("OBFM", EventType.SORTIE, Qual.WG, 1,1,0,0,0,0),
    SyllabusEvent("DBFM", EventType.SORTIE, Qual.WG, 1,1,0,0,0,0),
    SyllabusEvent("HABFM", EventType.SORTIE, Qual.WG, 1,1,0,0,0,0),
    SyllabusEvent("2vX TI", EventType.SIM, Qual.WG, 1,1,0,0,0,0),
    SyllabusEvent("ACM", EventType.SORTIE, Qual.WG, 1,1,0,0,1,1),
    SyllabusEvent("4vX TI", EventType.SIM, Qual.WG, 1,1,1,1,0,0),
    SyllabusEvent("DCA", EventType.SIM, Qual.WG, 1,1,1,1,0,0),
    SyllabusEvent("DCA", EventType.SORTIE, Qual.WG, 1,1,1,1,2,2),
    SyllabusEvent("U-SEAD", EventType.SIM, Qual.WG, 1,1,1,1,0,0),
    SyllabusEvent("D-SEAD", EventType.SIM, Qual.WG, 1,1,1,1,0,0),
    SyllabusEvent("U-SEAD", EventType.SORTIE, Qual.WG, 1,1,2,2,0,0),
    SyllabusEvent("O-SEAD", EventType.SIM, Qual.WG, 1,1,1,1,0,0),
    SyllabusEvent("O-SEAD", EventType.SORTIE, Qual.WG, 1,1,2,2,2,2),
    SyllabusEvent("O-AI", EventType.SIM, Qual.WG, 1,1,1,1,0,0),
    SyllabusEvent("CAS", EventType.SORTIE, Qual.WG, 1,1,0,0,0,0),
    SyllabusEvent("BSA", EventType.SORTIE, Qual.WG, 1,1,1,1,0,0),
]

# ----------------------
# IPUG Syllabus
# ----------------------
IPUG_SYLLABUS = [
    SyllabusEvent("OBFM", EventType.SORTIE, Qual.FL, 1,1,0,0,0,0),
    SyllabusEvent("DBFM", EventType.SORTIE, Qual.FL, 1,1,0,0,0,0),
    SyllabusEvent("HABFM", EventType.SORTIE, Qual.FL, 1,1,0,0,0,0),
    SyllabusEvent("2vX TI", EventType.SIM, Qual.FL, 1,1,0,0,0,0),
    SyllabusEvent("ACM", EventType.SORTIE, Qual.FL, 1,1,0,0,1,1),
    SyllabusEvent("4vX TI", EventType.SIM, Qual.FL, 1,1,1,1,0,0),
    SyllabusEvent("DCA", EventType.SIM, Qual.FL, 1,1,1,1,0,0),
    SyllabusEvent("DCA", EventType.SORTIE, Qual.FL, 1,1,1,1,2,2),
    SyllabusEvent("U-SEAD", EventType.SIM, Qual.FL, 1,1,1,1,0,0),
    SyllabusEvent("D-SEAD", EventType.SIM, Qual.FL, 1,1,1,1,0,0),
    SyllabusEvent("U-SEAD", EventType.SORTIE, Qual.FL, 1,1,2,2,0,0),
    SyllabusEvent("O-SEAD", EventType.SIM, Qual.FL, 1,1,1,1,0,0),
    SyllabusEvent("O-SEAD", EventType.SORTIE, Qual.FL, 1,1,2,2,2,2),
    SyllabusEvent("O-AI", EventType.SIM, Qual.FL, 1,1,1,1,0,0),
    SyllabusEvent("CAS", EventType.SORTIE, Qual.FL, 1,1,0,0,0,0),
    SyllabusEvent("BSA", EventType.SORTIE, Qual.FL, 1,1,1,1,0,0),
]

# ----------------------
# Continuation Profile
# ----------------------
@dataclass(frozen=True)
class ContinuationBucket:
    name: str
    min_qual: Qual
    side: str
    fraction: float

@dataclass
class ContinuationProfile:
    name: str
    buckets: List[ContinuationBucket]

CONTINUATION_PROFILE = ContinuationProfile(
    name="Continuation Training",
    buckets=[
        ContinuationBucket("Blue FL", Qual.FL, "Blue", 0.425),
        ContinuationBucket("Blue WG", Qual.WG, "Blue", 0.425),
        ContinuationBucket("Red FL", Qual.FL, "Red", 0.075),
        ContinuationBucket("Red WG", Qual.WG, "Red", 0.075),
    ]
)

# ----------------------
# Pilot Creation
# ----------------------
def create_pilots(cfg: SquadronConfig) -> List[Pilot]:
    experienced = int(cfg.total_pilots * cfg.experience_ratio)

    if experienced > cfg.total_pilots:
        raise ValueError("Experienced pilots cannot exceed total pilots")
    
    ip_count = cfg.ip_qty
    if ip_count > experienced:
        raise ValueError("IP quantity cannot exceed experienced pilots")
    
    fl_count = experienced - ip_count
    wg_count = cfg.total_pilots - experienced
    
    return ([Pilot(Qual.WG) for _ in range(wg_count)] +
            [Pilot(Qual.FL) for _ in range(fl_count)] +
            [Pilot(Qual.IP) for _ in range(ip_count)])

def total_monthly_capacity(cfg: SquadronConfig) -> float:
    return cfg.ute * cfg.paa

# ----------------------
# Rules
# ----------------------
def qual_meets_requirement(pilot: Pilot, required: Qual) -> bool:
    """
    Returns True if pilot's qualification satisfies the required seat qualification.
    WG seat: WG / FL / IP
    FL seat: FL / IP
    IP seat: IP only
    """
    if required == Qual.WG:
        return pilot.qual in (Qual.WG, Qual.FL, Qual.IP)
    if required == Qual.FL:
        return pilot.qual in (Qual.FL, Qual.IP)
    if required == Qual.IP:
        return pilot.qual == Qual.IP
    return False

def can_fill_seat(pilot: Pilot, seat_qual: Qual) -> bool:
    """
    Determines whether a pilot can fill a non-student seat
    given qual + upgrade restrictions.
    """

    # MQT students are restricted to MQT-only flying
    if pilot.qual == Qual.WG and pilot.upgrade == Upgrade.MQT:
        return False

    return qual_meets_requirement(pilot, seat_qual)

def is_student_eligible(pilot: Pilot, upgrade: Upgrade) -> bool:
    """
    Determines whether a pilot can be assigned as an upgrade student.
    """
    if upgrade in (Upgrade.MQT, Upgrade.FLUG):
        return pilot.qual == Qual.WG and pilot.upgrade == Upgrade.NONE

    if upgrade == Upgrade.IPUG:
        return pilot.qual == Qual.FL and pilot.upgrade == Upgrade.NONE

    return False

def can_fill_student_seat(
    pilot: Pilot,
    event: SyllabusEvent,
    upgrade: Upgrade
) -> bool:
    """
    Returns True if pilot can serve as the student for this syllabus event.
    """
    if pilot.qual != event.seat_type:
        return False

    return is_student_eligible(pilot, upgrade)

# ----------------------
# Helper: Can Fill Seat
# ----------------------
def can_fill_seat(pilot: Pilot, seat_type: Qual, syllabus_type: Upgrade | None = None) -> bool:
    """
    Returns True if the pilot can occupy a seat of the given type, considering:
    - Upgrade restrictions for WG+MQT, WG+FLUG, FL+IPUG
    - Qualification rules: WG -> WG/FL/IP, FL -> FL/IP, IP -> IP
    """
    # Upgrade-specific restrictions
    if pilot.upgrade == Upgrade.MQT:
        if syllabus_type != Upgrade.MQT:
            return False  # WG in MQT cannot fly other syllabus
    if pilot.upgrade == Upgrade.FLUG:
        if syllabus_type not in [Upgrade.FLUG, None]:
            return False  # WG in FLUG cannot fly MQT syllabus
    if pilot.upgrade == Upgrade.IPUG:
        if syllabus_type not in [Upgrade.IPUG, None]:
            return False  # FL in IPUG cannot fly other upgrade syllabus

    # Qualification seat rules
    if seat_type == Qual.WG:
        return pilot.qual in [Qual.WG, Qual.FL, Qual.IP]
    elif seat_type == Qual.FL:
        return pilot.qual in [Qual.FL, Qual.IP]
    elif seat_type == Qual.IP:
        return pilot.qual == Qual.IP
    return False


# ----------------------
# Allocate All Sorties
# ----------------------
def allocate_all_sorties(
    pilots: List[Pilot],
    mqt_syllabus: List[SyllabusEvent],
    flug_syllabus: List[SyllabusEvent],
    ipug_syllabus: List[SyllabusEvent],
    mqt_students: int,
    flug_students: int,
    ipug_students: int,
    continuation_profile: ContinuationProfile,
    total_capacity: int,
    sim_per_month: float = 3
):
    """
    Allocate all sorties for a phase:
    1. Upgrade sorties for MQT, FLUG, and IPUG students
    2. Remaining capacity allocated to continuation training proportionally
    3. Add simulator sorties
    """
    
    # ----------------------
    # Step 1: Select upgrade students
    # ----------------------
    def select_upgrade_students(upgrade_type: Upgrade, count: int) -> List[Pilot]:
        if upgrade_type in [Upgrade.MQT, Upgrade.FLUG]:
            candidates = [p for p in pilots if p.upgrade is None and p.qual == Qual.WG]
        elif upgrade_type == Upgrade.IPUG:
            candidates = [p for p in pilots if p.upgrade is None and p.qual == Qual.FL]
        else:
            candidates = []
        selected = candidates[:count]
        for p in selected:
            p.upgrade = upgrade_type
        return selected

    mqt_students_list = select_upgrade_students(Upgrade.MQT, mqt_students)
    flug_students_list = select_upgrade_students(Upgrade.FLUG, flug_students)
    ipug_students_list = select_upgrade_students(Upgrade.IPUG, ipug_students)

    # ----------------------
    # Step 2: Allocate upgrade syllabus
    # ----------------------
    def allocate_syllabus(syllabus: List[SyllabusEvent], upgrade_students: List[Pilot], syllabus_type: Upgrade):
        for event in syllabus:
            # Allocate blue WG
            for _ in range(event.num_blue_wg):
                eligible = sorted(
                    [p for p in upgrade_students + pilots if can_fill_seat(p, Qual.WG, syllabus_type)],
                    key=lambda p: p.sortie_monthly
                )
                if eligible:
                    eligible[0].sortie_monthly += 1

            # Allocate blue FL
            for _ in range(event.num_blue_fl):
                eligible = sorted(
                    [p for p in upgrade_students + pilots if can_fill_seat(p, Qual.FL, syllabus_type)],
                    key=lambda p: p.sortie_monthly
                )
                if eligible:
                    eligible[0].sortie_monthly += 1

            # Allocate red WG
            for _ in range(event.num_red_wg):
                eligible = sorted(
                    [p for p in pilots if can_fill_seat(p, Qual.WG, syllabus_type)],
                    key=lambda p: p.sortie_monthly
                )
                if eligible:
                    eligible[0].sortie_monthly += 1

            # Allocate red FL
            for _ in range(event.num_red_fl):
                eligible = sorted(
                    [p for p in pilots if can_fill_seat(p, Qual.FL, syllabus_type)],
                    key=lambda p: p.sortie_monthly
                )
                if eligible:
                    eligible[0].sortie_monthly += 1

    allocate_syllabus(mqt_syllabus, mqt_students_list, Upgrade.MQT)
    allocate_syllabus(flug_syllabus, flug_students_list, Upgrade.FLUG)
    allocate_syllabus(ipug_syllabus, ipug_students_list, Upgrade.IPUG)

    # ----------------------
    # Step 3: Allocate continuation sorties
    # ----------------------
    used_sorties = sum(p.sortie_monthly for p in pilots)
    remaining_capacity = max(0, total_capacity - used_sorties)

    # Compute integer allocation + distribute leftover by largest remainder
    raw_qty = [(b, remaining_capacity * b.fraction) for b in continuation_profile.buckets]
    base_qty = {b: int(x) for b, x in raw_qty}
    leftover = remaining_capacity - sum(base_qty.values())
    # Distribute leftover by largest fractional remainder
    for b, val in sorted(raw_qty, key=lambda x: x[1]-int(x[1]), reverse=True)[:leftover]:
        base_qty[b] += 1

    for bucket, qty in base_qty.items():
        eligible = sorted(
            [p for p in pilots if p.qual.value >= bucket.min_qual.value],
            key=lambda p: p.sortie_monthly
        )
        for i in range(qty):
            if not eligible:
                break
            eligible[i % len(eligible)].sortie_monthly += 1

    # ----------------------
    # Step 4: Add simulator sorties and total
    # ----------------------
    for p in pilots:
        p.sim_monthly += sim_per_month
        p.total_monthly = p.sortie_monthly + p.sim_monthly


# ----------------------
# RAP Shortfall / State
# ----------------------
def rap_shortfall(pilots: List[Pilot]):
    wg = max(0, 9 - sum(p.sortie_monthly for p in pilots if p.qual==Qual.WG)/len([p for p in pilots if p.qual==Qual.WG]))
    fl = max(0, 8 - sum(p.sortie_monthly for p in pilots if p.qual==Qual.FL)/len([p for p in pilots if p.qual==Qual.FL]))
    ip = max(0, 8 - sum(p.sortie_monthly for p in pilots if p.qual==Qual.IP)/len([p for p in pilots if p.qual==Qual.IP]))
    return {"WG": wg, "FL": fl, "IP": ip}

def rap_state_label(pilots: List[Pilot]):
    s = rap_shortfall(pilots)
    wg_ok, fl_ok, ip_ok = s["WG"]==0, s["FL"]==0, s["IP"]==0
    state_code = (4 if wg_ok else 0) + (2 if fl_ok else 0) + (1 if ip_ok else 0)
    labels = {0:"None",1:"IP only",2:"FL only",3:"FL+IP",4:"WG only",5:"WG+IP",6:"WG+FL",7:"WG+FL+IP"}
    return labels[state_code]

# ----------------------
# Run Phase Model
# ----------------------
def run_phase(cfg: SquadronConfig):
    pilots = create_pilots(cfg)
    total_capacity = int(total_monthly_capacity(cfg) * cfg.phase_length_days)  # scale to phase

    allocate_all_sorties(
        pilots=pilots,
        mqt_syllabus=MQT_SYLLABUS,
        flug_syllabus=FLUG_SYLLABUS,
        ipug_syllabus=IPUG_SYLLABUS,
        mqt_students=cfg.mqt_students,
        flug_students=cfg.flug_students,
        ipug_students=cfg.ipug_students,
        continuation_profile=CONTINUATION_PROFILE,
        total_capacity=total_capacity
    )

    return pilots

# ----------------------
# Example Run
# ----------------------
if __name__ == "__main__":
    cfg = SquadronConfig(
        ute=10,
        paa=21,
        mqt_students=4,
        flug_students=4,
        ipug_students=2,
        total_pilots=30,
        experience_ratio=0.6,
        ip_qty=5
    )
    # Print summary per qual

    def print_phase_summary(pilots: List[Pilot], cfg: SquadronConfig):
        for qual in [Qual.WG, Qual.FL, Qual.IP]:
            p_list = [p for p in pilots if p.qual == qual]
            print(f"{qual.value} (n={len(p_list)}):")
            
            if qual == Qual.WG:
                for upgrade in [Upgrade.MQT, Upgrade.FLUG, Upgrade.NONE]:
                    sub_list = [p for p in p_list if p.upgrade == upgrade]
                    if not sub_list:
                        continue
                    avg_phase_sortie = sum(p.sortie_monthly for p in sub_list)/len(sub_list)
                    avg_annual_sortie = avg_phase_sortie * 4
                    avg_monthly_sortie = avg_phase_sortie / (cfg.phase_length_days/30)
                    wg_rap_shortfall = max(0, 9 - avg_monthly_sortie)
                    wg_willsbach_shortfall = max(0, 12 - avg_monthly_sortie)
                    print(f"  {upgrade.value} Avg annual sortie: {avg_annual_sortie:.2f}")
                    print(f"  {upgrade.value} Avg monthly sortie: {avg_monthly_sortie:.2f}")
                    print(f"    RAP shortfall: {wg_rap_shortfall:.2f}")
                    print(f"    Willsbach shortfall: {wg_willsbach_shortfall:.2f}")
            
            elif qual == Qual.FL:
                for upgrade in [Upgrade.IPUG, Upgrade.NONE]:
                    sub_list = [p for p in p_list if p.upgrade == upgrade]
                    if not sub_list:
                        continue
                    avg_phase_sortie = sum(p.sortie_monthly for p in sub_list)/len(sub_list)
                    avg_annual_sortie = avg_phase_sortie * 4
                    avg_monthly_sortie = avg_phase_sortie / (cfg.phase_length_days/30)
                    fl_rap_shortfall = max(0, 8 - avg_monthly_sortie)
                    fl_willsbach_shortfall = max(0, 12 - avg_monthly_sortie)
                    print(f"  {upgrade.value} Avg annual sortie: {avg_annual_sortie:.2f}")
                    print(f"  {upgrade.value} Avg monthly sortie: {avg_monthly_sortie:.2f}")
                    print(f"    RAP shortfall: {fl_rap_shortfall:.2f}")
                    print(f"    Willsbach shortfall: {fl_willsbach_shortfall:.2f}")
            
            elif qual == Qual.IP:
                sub_list = [p for p in p_list]
                avg_phase_sortie = sum(p.sortie_monthly for p in sub_list)/len(sub_list)
                avg_annual_sortie = avg_phase_sortie * 4
                avg_monthly_sortie = avg_phase_sortie / (cfg.phase_length_days/30)
                ip_rap_shortfall = max(0, 8 - avg_monthly_sortie)
                ip_willsbach_shortfall = max(0, 12 - avg_monthly_sortie)
                print(f"  Avg annual sortie: {avg_annual_sortie:.2f}")
                print(f"  Avg monthly sortie: {avg_monthly_sortie:.2f}")
                print(f"    RAP shortfall: {ip_rap_shortfall:.2f}")
                print(f"    Willsbach shortfall: {ip_willsbach_shortfall:.2f}")


    # ----------------------
    # Run Phase
    # ----------------------
    pilots = run_phase(cfg)
    print_phase_summary(pilots, cfg)


