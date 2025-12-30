from dataclasses import dataclass
from enum import Enum
from typing import List
import random
from models import SquadronConfig, Pilot, Qual, Upgrade
from syllabi import SyllabusEvent, UpgradeProgram, ContinuationProfile

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
    sim_per_month: float = 3,
    allocation_noise: float = 0.0
):
    import random

    # ----------------------
    # Step 1: Select upgrade students
    # ----------------------
    def select_upgrade_students(upgrade_type: Upgrade, count: int) -> List[Pilot]:
        if upgrade_type in [Upgrade.MQT, Upgrade.FLUG]:
            candidates = [p for p in pilots if p.upgrade == Upgrade.NONE and p.qual == Qual.WG]
        elif upgrade_type == Upgrade.IPUG:
            candidates = [p for p in pilots if p.upgrade == Upgrade.NONE and p.qual == Qual.FL]
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
    # Step 2: Allocate upgrade syllabus sorties
    # ----------------------
    def allocate_syllabus(
        syllabus: List[SyllabusEvent],
        upgrade_students: List[Pilot],
        pilots: List[Pilot],
        allocation_noise: float = 0.0
    ):
        """
        Allocate sorties for a syllabus with STRICT seat rules:
        - num_student → ONLY the assigned upgrade student
        - num_instructor → ONLY IPs
        - blue slots → equitable across eligible pilots
        - red slots → equitable across eligible pilots
        """
        for pilot in pilots:
            print(f'Pilot: {pilot.qual}/{pilot.upgrade}, Sortie monthly: {pilot.sortie_monthly}')
            
        for event in syllabus:
            print(f'Event name: {event.name}')
            # ----------------------
            # STUDENT SORTIES (Upgrade Students Only)
            # ----------------------
            for student in upgrade_students:
                print(f'Student: {student.qual}/{student.upgrade}')
                for _ in range(event.num_student):
                    student.sortie_monthly += 1
                    student.sortie_blue_monthly += 1
                    print(f"Student {student.qual}/{student.upgrade} flies student slot → "
                      f"Total: {student.sortie_monthly}, Blue: {student.sortie_blue_monthly}")
                    
            print(f'Student Sorties Allocated!')
            for pilot in pilots:
                print(f'Pilot: {pilot.qual}/{pilot.upgrade}, Sortie monthly: {pilot.sortie_monthly}')

            # ----------------------
            # INSTRUCTOR SORTIES (IP Only)
            # ----------------------
            for _ in range(event.num_instructor):
                eligible_ips = [p for p in pilots if p.qual == Qual.IP]
                count = 1
                for ip in eligible_ips:
                    print(f'IP # {count} Sortie Monthly: {ip.sortie_monthly}')
                    count += 1
                if not eligible_ips:
                    continue
                # Pick the IP with fewest sorties (add random noise for tie-break)
                eligible_ips.sort(
                    key=lambda p: p.sortie_monthly + random.uniform(0, allocation_noise)
                )
                ip = eligible_ips[0]
                print(f'IP Sortie Monthly: {eligible_ips[0].sortie_monthly}')
                ip.sortie_monthly += 1
                ip.sortie_blue_monthly += 1
                print(f"{ip.qual}/{ip.upgrade} flies instructor slot → "
                  f"Total: {ip.sortie_monthly}, Blue: {ip.sortie_blue_monthly}")

        # ----------------------
        # BLUE SEATS (equitable across eligible pilots)
        # ----------------------
        for _ in range(event.num_blue_wg):
            eligible_wg = [p for p in pilots if p.upgrade != Upgrade.MQT]
            if not eligible_wg:
                continue
            eligible_wg.sort(key=lambda p: p.sortie_monthly + random.uniform(0, allocation_noise))
            selected = eligible_wg[0]
            selected.sortie_monthly += 1
            selected.sortie_blue_monthly += 1
            print(f"{selected.qual}/{selected.upgrade} flies blue WG → "
                  f"Total: {selected.sortie_monthly}, Blue: {selected.sortie_blue_monthly}")
        
        for _ in range(event.num_blue_fl):
            eligible_fl = [p for p in pilots if p.qual == Qual.FL]
            if not eligible_fl:
                continue
            eligible_fl.sort(key=lambda p: p.sortie_monthly + random.uniform(0, allocation_noise))
            #TODO way too many sorties are being allocated here.
            selected = eligible_fl[0]
            selected.sortie_monthly += 1
            selected.sortie_blue_monthly += 1
            print(f"{selected.qual}/{selected.upgrade} flies blue FL → "
                  f"Total: {selected.sortie_monthly}, Blue: {selected.sortie_blue_monthly}")

        # ----------------------
        # RED SEATS (equitable across eligible pilots)
        # ----------------------
        for _ in range(event.num_red_wg):
            eligible_wg = [p for p in pilots if p.upgrade != Upgrade.MQT]
            if not eligible_wg:
                continue
            eligible_wg.sort(key=lambda p: p.sortie_monthly + random.uniform(0, allocation_noise))
            selected = eligible_wg[0]
            selected.sortie_red_monthly += 1
            print(f"{selected.qual}/{selected.upgrade} flies red WG → "
                  f"Total: {selected.sortie_monthly}, Red: {selected.sortie_red_monthly}")
        
        for _ in range(event.num_red_fl):
            eligible_fl = [p for p in pilots if p.qual == Qual.FL]
            if not eligible_fl:
                continue
            eligible_fl.sort(key=lambda p: p.sortie_monthly + random.uniform(0, allocation_noise))
            selected = eligible_fl[0]
            selected.sortie_monthly += 1
            selected.sortie_red_monthly += 1
            print(f"{selected.qual}/{selected.upgrade} flies red FL → "
                  f"Total: {selected.sortie_monthly}, Red: {selected.sortie_red_monthly}")

    # ----------------------
    # Step 3: Allocate CT sorties
    # ----------------------
    used_sorties = sum(p.sortie_monthly for p in pilots)
    remaining_capacity = max(0, total_capacity - used_sorties)

    # Only CT-eligible pilots (exclude MQT students)
    ct_candidates = [p for p in pilots if p.upgrade != Upgrade.MQT]

    # Allocate continuation sorties according to profile fractions
    raw_qty = [(b, remaining_capacity * b.fraction) for b in continuation_profile.buckets]
    base_qty = {b: int(x) for b, x in raw_qty}
    leftover = remaining_capacity - sum(base_qty.values())

    # Distribute leftover by largest fractional remainder
    for b, val in sorted(raw_qty, key=lambda x: x[1]-int(x[1]), reverse=True)[:leftover]:
        base_qty[b] += 1

    for bucket, qty in base_qty.items():
        # Filter CT-eligible pilots meeting the min_qual
        eligible = [p for p in ct_candidates if p.qual.value >= bucket.min_qual.value]
        if not eligible:
            continue
        for i in range(qty):
            selected = eligible[i % len(eligible)]
            selected.sortie_monthly += 1
            if bucket.side == "Blue":
                selected.sortie_blue_monthly += 1
            else:
                selected.sortie_red_monthly += 1

    # ----------------------
    # Step 4: Add simulator sorties and total
    # ----------------------
    for p in pilots:
        p.sim_monthly += sim_per_month
        p.total_monthly = p.sortie_monthly + p.sim_monthly

    for pilot in pilots:
        print(f'Starting value! Pilot: {pilot.qual}/{pilot.upgrade}, Sortie monthly: {pilot.sortie_monthly}')
    allocate_syllabus(mqt_syllabus, mqt_students_list, pilots, allocation_noise=0.0)
    for pilot in pilots:
        print(f'MQT allocated! Pilot: {pilot.qual}/{pilot.upgrade}, Sortie monthly: {pilot.sortie_monthly}')
    allocate_syllabus(flug_syllabus, flug_students_list, pilots, allocation_noise=0.0)
    for pilot in pilots:
        print(f'FLUG allocated! Pilot: {pilot.qual}/{pilot.upgrade}, Sortie monthly: {pilot.sortie_monthly}')
    allocate_syllabus(ipug_syllabus, ipug_students_list, pilots, allocation_noise=0.0)
    for pilot in pilots:
        print(f'IPUG allocated! Pilot: {pilot.qual}/{pilot.upgrade}, Sortie monthly: {pilot.sortie_monthly}')

# ----------------------
# RAP Shortfall / State
# ----------------------
def rap_shortfall(pilots: List[Pilot]):
    wg_pilots = [p for p in pilots if p.qual == Qual.WG]
    wg = max(0, 9 - sum(p.sortie_monthly for p in wg_pilots)/len(wg_pilots)) if wg_pilots else 0
    fl_pilots = [p for p in pilots if p.qual == Qual.FL]
    fl = max(0, 8 - sum(p.sortie_monthly for p in fl_pilots)/len(fl_pilots)) if fl_pilots else 0
    ip_pilots = [p for p in pilots if p.qual == Qual.IP]
    ip = max(0, 8 - sum(p.sortie_monthly for p in ip_pilots)/len(ip_pilots)) if ip_pilots else 0
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
def run_phase(cfg: SquadronConfig, allocation_noise: float = 0.0):
    pilots = create_pilots(cfg)
    for pilot in pilots:
        print(f'Qual:{pilot.qual} Upgrade:{pilot.upgrade} Starting sortie monthly: {pilot.sortie_monthly}')
    total_capacity = int(total_monthly_capacity(cfg) * (cfg.phase_length_days / 30))  # scale to phase

    allocate_all_sorties(
        pilots=pilots,
        # mqt_syllabus=MQT_SYLLABUS,
        # flug_syllabus=FLUG_SYLLABUS,
        # ipug_syllabus=IPUG_SYLLABUS,
        mqt_syllabus=TEST_MQT_SYLLABUS,
        flug_syllabus=TEST_FLUG_SYLLABUS,
        ipug_syllabus=TEST_IPUG_SYLLABUS,
        mqt_students=cfg.mqt_students,
        flug_students=cfg.flug_students,
        ipug_students=cfg.ipug_students,
        continuation_profile=CONTINUATION_PROFILE,
        total_capacity=total_capacity,
        allocation_noise=allocation_noise
    )

    for p in pilots:
        p.update_total()
        print(f'{p.qual}/{p.upgrade} Monthly Sorties: {p.sortie_monthly}, Blue Monthly: {p.sortie_blue_monthly}, Red Monthly: {p.sortie_red_monthly}')

    return pilots

# ----------------------
# Print Phase Summary
# ----------------------
def print_phase_summary(pilots: List[Pilot], cfg: SquadronConfig):
    # Define the groups you want
    wingmen_groups = [
        ("MQT WG", Qual.WG, Upgrade.MQT, 9),
        ("WG", Qual.WG, Upgrade.NONE, 9),
        ("FLUG WG", Qual.WG, Upgrade.FLUG, 9)
    ]
    flight_leads_groups = [
        ("FL", Qual.FL, Upgrade.NONE, 8),
        ("IPUG FL", Qual.FL, Upgrade.IPUG, 8)
    ]
    instructor_pilots_group = [
        ("IP", Qual.IP, None, 8)  # All IPs regardless of upgrade
    ]

    # ----------------------
    # Helper to calculate average monthly sorties
    # ----------------------
    def avg_monthly_sortie(p_list):
        if not p_list:
            return 0
        return sum(p.sortie_monthly for p in p_list)/len(p_list)

    # ----------------------
    # Print summary
    # ----------------------
    print("=== Monthly RAP Summary ===")
    for name, qual, upgrade, target in wingmen_groups + flight_leads_groups + instructor_pilots_group:
        subgroup = [p for p in pilots if p.qual == qual and (upgrade is None or p.upgrade == upgrade)]
        avg_monthly = avg_monthly_sortie(subgroup)
        shortfall = max(0, target - avg_monthly)
        print(f"{name}: {avg_monthly:.1f} ({shortfall:.1f})")


# ----------------------
# Example Run
# ----------------------
if __name__ == "__main__":
    cfg = SquadronConfig(
        ute=5,
        paa=4,
        mqt_students=2,
        flug_students=1,
        ipug_students=1,
        total_pilots=10,
        experience_ratio=0.5,
        ip_qty=2,
        phase_length_days=30
    )

    # ----------------------
    # Run Phase
    # ----------------------
    pilots = run_phase(cfg, allocation_noise=0.0)

    # ----------------------
    # Print Monthly RAP Summary for key sub-groups
    # ----------------------
    print_phase_summary(pilots, cfg)



