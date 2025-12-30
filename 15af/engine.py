import random
from typing import List, Dict
from models import SquadronConfig, Pilot, Qual, Upgrade
from syllabi import SyllabusEvent, ContinuationProfile, UpgradeProgram
import rules  # Ensure rules.py is created as discussed

# ----------------------
# Pilot Creation
# ----------------------
def create_pilots(cfg: SquadronConfig) -> List[Pilot]:
    """
    Generates the initial list of pilots based on configuration.
    """
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
# Selection Phase
# ----------------------
def select_upgrade_students(pilots: List[Pilot], upgrade_type: Upgrade, count: int) -> List[Pilot]:
    """
    Identifies eligible pilots and marks them for upgrade.
    """
    candidates = [p for p in pilots if rules.can_start_upgrade(p, upgrade_type)]
    
    # Simple selection: take the first available 
    # (Future improvement: Sort by experience/seniority)
    selected = candidates[:count]
    
    for p in selected:
        p.upgrade = upgrade_type
        
    return selected

# ----------------------
# Allocation Helpers
# ----------------------
def assign_sortie(candidates: List[Pilot], side: str = "Blue", noise: float = 0.0) -> bool:
    """
    Selects the best candidate (lowest utilization) to fly a sortie.
    Returns True if a pilot was found and assigned, False otherwise.
    """
    if not candidates:
        return False
    
    # Sort by current monthly sorties + random noise for distribution
    candidates.sort(key=lambda p: p.sortie_monthly + random.uniform(0, noise))
    
    winner = candidates[0]
    
    # Use helper method if available in models.py, otherwise update manually
    if hasattr(winner, 'add_sortie'):
        winner.add_sortie(side)
    else:
        winner.sortie_monthly += 1
        if side == "Blue":
            winner.sortie_blue_monthly += 1
        elif side == "Red":
            winner.sortie_red_monthly += 1
        
    return True

# ----------------------
# Syllabus Execution
# ----------------------
def process_syllabus_event(
    event: SyllabusEvent, 
    upgrade_students: List[Pilot], 
    all_pilots: List[Pilot], 
    syllabus_upgrade_type: Upgrade,
    noise: float
):
    """
    Allocates sorties for a specific syllabus event.
    CRITICAL FIX: Support sorties are now generated PER student sortie.
    """
    for student in upgrade_students:
        # 1. Student Sorties (The student flies 'num_student' times for this event)
        for _ in range(event.num_student):
            
            # -- Student flies --
            if hasattr(student, 'add_sortie'):
                student.add_sortie("Blue")
            else:
                student.sortie_monthly += 1
                student.sortie_blue_monthly += 1
            
            # -- Instructor flies (Per student sortie) --
            for _ in range(event.num_instructor):
                # Only IPs can instruct
                ips = [p for p in all_pilots if rules.can_fill_seat(p, Qual.IP, syllabus_upgrade_type)]
                assign_sortie(ips, "Blue", noise)

            # -- Blue Wingmen (Per student sortie) --
            for _ in range(event.num_blue_wg):
                candidates = [p for p in all_pilots if rules.can_fill_seat(p, Qual.WG, syllabus_upgrade_type)]
                # Filter out the student themselves if they are in the candidate list
                candidates = [p for p in candidates if p is not student]
                assign_sortie(candidates, "Blue", noise)

            # -- Blue Flight Leads (Per student sortie) --
            for _ in range(event.num_blue_fl):
                candidates = [p for p in all_pilots if rules.can_fill_seat(p, Qual.FL, syllabus_upgrade_type)]
                candidates = [p for p in candidates if p is not student]
                assign_sortie(candidates, "Blue", noise)

            # -- Red Wingmen (Per student sortie) --
            for _ in range(event.num_red_wg):
                candidates = [p for p in all_pilots if rules.can_fill_seat(p, Qual.WG, syllabus_upgrade_type)]
                candidates = [p for p in candidates if p is not student]
                assign_sortie(candidates, "Red", noise)

            # -- Red Flight Leads (Per student sortie) --
            for _ in range(event.num_red_fl):
                candidates = [p for p in all_pilots if rules.can_fill_seat(p, Qual.FL, syllabus_upgrade_type)]
                candidates = [p for p in candidates if p is not student]
                assign_sortie(candidates, "Red", noise)

def run_upgrade_program(
    syllabus: List[SyllabusEvent],
    students: List[Pilot],
    all_pilots: List[Pilot],
    upgrade_type: Upgrade,
    noise: float
):
    for event in syllabus:
        process_syllabus_event(event, students, all_pilots, upgrade_type, noise)

# ----------------------
# Continuation Training (CT)
# ----------------------
def allocate_continuation_training(
    pilots: List[Pilot],
    profile: ContinuationProfile,
    total_capacity: int,
    noise: float
):
    # Calculate how much capacity is left
    used_sorties = sum(p.sortie_monthly for p in pilots)
    remaining_capacity = max(0, total_capacity - used_sorties)

    if remaining_capacity <= 0:
        return

    # Identify CT candidates (anyone NOT in an active upgrade)
    ct_candidates = [p for p in pilots if p.upgrade == Upgrade.NONE]

    if not ct_candidates:
        return

    # Calculate bucket sizes
    raw_qty = [(b, remaining_capacity * b.fraction) for b in profile.buckets]
    base_qty = {b: int(x) for b, x in raw_qty}
    
    # Distribute leftover "fractional" sorties
    leftover = remaining_capacity - sum(base_qty.values())
    sorted_remainders = sorted(raw_qty, key=lambda x: x[1]-int(x[1]), reverse=True)
    
    for i in range(leftover):
        bucket = sorted_remainders[i % len(sorted_remainders)][0]
        base_qty[bucket] += 1

    # Execute allocation per bucket
    for bucket, qty in base_qty.items():
        # Find eligible pilots for this specific CT bucket
        # We access the internal hierarchy check from rules since CT doesn't have a syllabus upgrade type
        eligible = [p for p in ct_candidates if rules._qual_hierarchy_check(p.qual, bucket.min_qual)]
        
        for _ in range(qty):
            assign_sortie(eligible, bucket.side, noise)

# ----------------------
# Main Simulation Phase
# ----------------------
def run_phase_simulation(cfg: SquadronConfig, pilots: List[Pilot], allocation_noise: float = 0.0):
    
    # 1. Reset Phase Counters
    for p in pilots:
        if hasattr(p, 'reset_counters'):
            p.reset_counters()
        else:
            p.sortie_monthly = 0 
            p.sortie_blue_monthly = 0
            p.sortie_red_monthly = 0
            p.sim_monthly = 0

    # 2. Select Students
    # Note: If pilots were already assigned upgrades in a previous phase, 
    # you might want to adjust this logic.
    mqt_students = select_upgrade_students(pilots, Upgrade.MQT, cfg.mqt_students)
    flug_students = select_upgrade_students(pilots, Upgrade.FLUG, cfg.flug_students)
    ipug_students = select_upgrade_students(pilots, Upgrade.IPUG, cfg.ipug_students)

    # 3. Execute Syllabi
    # Import these from your syllabi file
    from syllabi import TEST_MQT_SYLLABUS, TEST_FLUG_SYLLABUS, TEST_IPUG_SYLLABUS, CONTINUATION_PROFILE

    run_upgrade_program(TEST_MQT_SYLLABUS, mqt_students, pilots, Upgrade.MQT, allocation_noise)
    run_upgrade_program(TEST_FLUG_SYLLABUS, flug_students, pilots, Upgrade.FLUG, allocation_noise)
    run_upgrade_program(TEST_IPUG_SYLLABUS, ipug_students, pilots, Upgrade.IPUG, allocation_noise)

    # 4. Continuation Training
    # Scale capacity to phase length (e.g. 1 month vs 4 months)
    phase_months = cfg.phase_length_days / 30.0
    total_capacity = int(total_monthly_capacity(cfg) * phase_months)
    
    allocate_continuation_training(pilots, CONTINUATION_PROFILE, total_capacity, allocation_noise)

    # 5. Finalize Stats
    for p in pilots:
        # Assuming sim is flat per month
        p.sim_monthly += (3 * phase_months) 
        p.update_total()

    return pilots

# ----------------------
# Reporting
# ----------------------
def print_phase_summary(pilots: List[Pilot], cfg: SquadronConfig):
    print("\n=== Phase Summary ===")
    
    groups = {
        "MQT Students": [p for p in pilots if p.upgrade == Upgrade.MQT],
        "FLUG Students": [p for p in pilots if p.upgrade == Upgrade.FLUG],
        "IPUG Students": [p for p in pilots if p.upgrade == Upgrade.IPUG],
        "Line Wingmen": [p for p in pilots if p.qual == Qual.WG and p.upgrade == Upgrade.NONE],
        "Line FLs": [p for p in pilots if p.qual == Qual.FL and p.upgrade == Upgrade.NONE],
        "IPs": [p for p in pilots if p.qual == Qual.IP]
    }

    for name, group in groups.items():
        if not group:
            print(f"{name}: None")
            continue
            
        avg_sorties = sum(p.sortie_monthly for p in group) / len(group)
        print(f"{name} ({len(group)}): Avg Sorties {avg_sorties:.1f}")