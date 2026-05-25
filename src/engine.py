import random
from typing import List
from src.models import (
    EventType,
    SquadronConfig,
    Pilot,
    Qual,
    Upgrade,
    SIM_RAP_MONTHLY,
    PhaseUpgradeHandoff,
    DeferredLine,
)
from src.syllabi import SyllabusEvent, ContinuationProfile
from src import rules
from src.syllabi import MQT_SYLLABUS, FLUG_SYLLABUS, IPUG_SYLLABUS, CONTINUATION_PROFILE

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

    if cfg.mqt_students + cfg.flug_students > wg_count:
        raise ValueError ("WG upgrade quantity cannot exceed WG quantity")
    
    return ([Pilot(Qual.WG) for _ in range(wg_count)] +
            [Pilot(Qual.FL) for _ in range(fl_count)] +
            [Pilot(Qual.IP) for _ in range(ip_count)])

def total_phase_capacity(cfg: SquadronConfig) -> float:
    return cfg.ute * cfg.paa


def build_phase_upgrade_handoff(pilots: List[Pilot]) -> PhaseUpgradeHandoff:
    def syllabus_complete(upgrade: Upgrade) -> bool:
        students = [p for p in pilots if p.upgrade == upgrade]
        return all(not p.incomplete_syllabus_items for p in students) if students else True

    deferred = [
        DeferredLine(upgrade=p.upgrade)
        for p in pilots
        for _ in p.incomplete_syllabus_items
    ]
    return PhaseUpgradeHandoff(
        mqt_syllabus_complete=syllabus_complete(Upgrade.MQT),
        flug_syllabus_complete=syllabus_complete(Upgrade.FLUG),
        ipug_syllabus_complete=syllabus_complete(Upgrade.IPUG),
        deferred_requirements=deferred,
    )


# ----------------------
# Selection Phase
# ----------------------
def select_upgrade_students(pilots: List[Pilot], upgrade_type: Upgrade, count: int) -> List[Pilot]:
    """Pick up to ``count`` new students (``upgrade == NONE``). Carryover students stay in upgrade."""
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
def assign_sortie(cfg: SquadronConfig, candidates: List[Pilot], side: str = "Blue", noise: float = 0.0) -> bool:
    """
    Selects the best candidate (lowest utilization) to fly a sortie.
    Returns True if a pilot was found and assigned, False otherwise.
    """
    if not candidates:
        return False
    
    # Sort by current monthly sorties + random noise for distribution
    candidates.sort(key=lambda p: p.sortie_phase + random.uniform(0, noise))
    
    winner = candidates[0]
    
    # Use helper method if available in models.py, otherwise update manually
    if hasattr(winner, 'add_sortie'):
        winner.add_sortie(avg_sortie_dur=cfg.avg_sortie_dur, side=side)
    else:
        winner.sortie_phase += 1
        if side == "Blue":
            winner.sortie_blue_phase += 1
        elif side == "Red":
            winner.sortie_red_phase += 1
        
    return True

def assign_sim(cfg: SquadronConfig, candidates: List[Pilot], noise: float = 0.0) -> bool:
    """
    Selects the best candidate (lowest utilization) to fly a simulator event.
    Returns True if a pilot was found and assigned, False otherwise.
    """
    if not candidates:
        return False

    candidates.sort(key=lambda p: p.sim_phase + random.uniform(0, noise))
    candidates[0].add_sim(cfg.avg_sortie_dur)
    return True

def check_syllabus_resources(
    event: SyllabusEvent,
    all_pilots: List[Pilot],
    syllabus_upgrade_type: Upgrade,
    total_capacity: int,
) -> bool:
    """Enough support pilots and (for sorties) iron capacity for one student line."""
    if len([p for p in all_pilots if rules.can_fill_seat(p, Qual.IP, syllabus_upgrade_type)]) < event.num_instructor:
        return False
    wg_pool = len([p for p in all_pilots if rules.can_fill_seat(p, Qual.WG, syllabus_upgrade_type)])
    if wg_pool < event.num_blue_wg or wg_pool < event.num_red_wg:
        return False
    fl_pool = len([p for p in all_pilots if rules.can_fill_seat(p, Qual.FL, syllabus_upgrade_type)])
    if fl_pool < event.num_blue_fl or fl_pool < event.num_red_fl:
        return False
    if event.event_type == EventType.SORTIE:
        slots = 1 + event.num_instructor + event.num_blue_wg + event.num_blue_fl + event.num_red_wg + event.num_red_fl
        if sum(p.sortie_phase for p in all_pilots) + slots > total_capacity:
            return False
    return True
# ----------------------
# Syllabus Execution
# ----------------------
def process_syllabus_event(
    event: SyllabusEvent, 
    upgrade_students: List[Pilot], 
    all_pilots: List[Pilot], 
    syllabus_upgrade_type: Upgrade,
    noise: float,
    cfg: SquadronConfig,
    total_capacity: int,
):
    """
    Allocates sorties for a specific syllabus event.
    CRITICAL FIX: Support sorties are now generated PER student sortie.
    """
    for student in upgrade_students:
        for _ in range(event.num_student):
            if not check_syllabus_resources(event, all_pilots, syllabus_upgrade_type, total_capacity):
                # Already deferred: leave list unchanged (no duplicate)
                if event not in student.incomplete_syllabus_items:
                    student.incomplete_syllabus_items.append(event)
                continue
            # Retry succeeded: drop from deferral queue before crediting the line
            if event in student.incomplete_syllabus_items:
                student.incomplete_syllabus_items.remove(event)
            if event.event_type == EventType.SIM:
                student.add_sim(cfg.avg_sortie_dur)
                for _ in range(event.num_instructor):
                    ips = [p for p in all_pilots if rules.can_fill_seat(p, Qual.IP, syllabus_upgrade_type)]
                    assign_sim(cfg, ips, noise)
                for _ in range(event.num_blue_wg):
                    candidates = [p for p in all_pilots if rules.can_fill_seat(p, Qual.WG, syllabus_upgrade_type)]
                    candidates = [p for p in candidates if p is not student]
                    assign_sim(cfg, candidates, noise)
                for _ in range(event.num_blue_fl):
                    candidates = [p for p in all_pilots if rules.can_fill_seat(p, Qual.FL, syllabus_upgrade_type)]
                    candidates = [p for p in candidates if p is not student]
                    assign_sim(cfg, candidates, noise)
                for _ in range(event.num_red_wg):
                    candidates = [p for p in all_pilots if rules.can_fill_seat(p, Qual.WG, syllabus_upgrade_type)]
                    candidates = [p for p in candidates if p is not student]
                    assign_sim(cfg, candidates, noise)
                for _ in range(event.num_red_fl):
                    candidates = [p for p in all_pilots if rules.can_fill_seat(p, Qual.FL, syllabus_upgrade_type)]
                    candidates = [p for p in candidates if p is not student]
                    assign_sim(cfg, candidates, noise)
            else:
                student.add_sortie(cfg.avg_sortie_dur, "Blue")
                for _ in range(event.num_instructor):
                    ips = [p for p in all_pilots if rules.can_fill_seat(p, Qual.IP, syllabus_upgrade_type)]
                    assign_sortie(cfg, ips, "Blue", noise)
                for _ in range(event.num_blue_wg):
                    candidates = [p for p in all_pilots if rules.can_fill_seat(p, Qual.WG, syllabus_upgrade_type)]
                    candidates = [p for p in candidates if p is not student]
                    assign_sortie(cfg, candidates, "Blue", noise)
                for _ in range(event.num_blue_fl):
                    candidates = [p for p in all_pilots if rules.can_fill_seat(p, Qual.FL, syllabus_upgrade_type)]
                    candidates = [p for p in candidates if p is not student]
                    assign_sortie(cfg, candidates, "Blue", noise)
                for _ in range(event.num_red_wg):
                    candidates = [p for p in all_pilots if rules.can_fill_seat(p, Qual.WG, syllabus_upgrade_type)]
                    candidates = [p for p in candidates if p is not student]
                    assign_sortie(cfg, candidates, "Red", noise)
                for _ in range(event.num_red_fl):
                    candidates = [p for p in all_pilots if rules.can_fill_seat(p, Qual.FL, syllabus_upgrade_type)]
                    candidates = [p for p in candidates if p is not student]
                    assign_sortie(cfg, candidates, "Red", noise)

def run_upgrade_program(
    syllabus: List[SyllabusEvent],
    students: List[Pilot],
    all_pilots: List[Pilot],
    upgrade_type: Upgrade,
    noise: float,
    cfg: SquadronConfig,
    total_capacity: int,
):
    for event in syllabus:
        process_syllabus_event(event, students, all_pilots, upgrade_type, noise, cfg=cfg, total_capacity=total_capacity)

# ----------------------
# Continuation Training (CT)
# ----------------------
def allocate_continuation_training(
    pilots: List[Pilot],
    profile: ContinuationProfile,
    total_capacity: int,
    noise: float,
    cfg: SquadronConfig
):
    # Calculate how much capacity is left
    used_sorties = sum(p.sortie_phase for p in pilots)
    remaining_capacity = max(0, total_capacity - used_sorties)

    if remaining_capacity <= 0:
        return

    # Identify CT candidates (anyone NOT in an active upgrade)
    ct_candidates = [p for p in pilots if p.upgrade != Upgrade.MQT]

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
            assign_sortie(cfg=cfg, candidates=eligible, side=bucket.side, noise=noise)

# ----------------------
# Main Simulation Phase
# ----------------------
def run_phase_simulation(cfg: SquadronConfig, pilots: List[Pilot], allocation_noise: float = 0.0):
    
    # 1. Reset Phase Counters
    for p in pilots:
        if hasattr(p, 'reset_counters'):
            p.reset_counters()
        else:
            p.sortie_phase = 0 
            p.sortie_blue_phase = 0
            p.sortie_red_phase = 0
            p.sim_phase = 0

    # 2. New students only (carryover already have upgrade MQT/FLUG/IPUG)
    mqt_students = select_upgrade_students(pilots, Upgrade.MQT, cfg.mqt_students)
    flug_students = select_upgrade_students(pilots, Upgrade.FLUG, cfg.flug_students)
    ipug_students = select_upgrade_students(pilots, Upgrade.IPUG, cfg.ipug_students)

    phase_months = cfg.phase_length_days / 30.0
    total_capacity = int(total_phase_capacity(cfg) * phase_months)

    # 3. Carryover: incomplete lines only (not a full new syllabus)
    for upgrade_type in (Upgrade.MQT, Upgrade.FLUG, Upgrade.IPUG):
        for student in [p for p in pilots if p.upgrade == upgrade_type and p.incomplete_syllabus_items]:
            for event in list(student.incomplete_syllabus_items):
                process_syllabus_event(
                    event, [student], pilots, upgrade_type, allocation_noise, cfg, total_capacity
                )

    # 4. Full syllabus for new students only
    run_upgrade_program(MQT_SYLLABUS, mqt_students, pilots, Upgrade.MQT, allocation_noise, cfg=cfg, total_capacity=total_capacity)
    run_upgrade_program(FLUG_SYLLABUS, flug_students, pilots, Upgrade.FLUG, allocation_noise, cfg=cfg, total_capacity=total_capacity)
    run_upgrade_program(IPUG_SYLLABUS, ipug_students, pilots, Upgrade.IPUG, allocation_noise, cfg=cfg, total_capacity=total_capacity)

    # 5. Continuation Training
    allocate_continuation_training(pilots, CONTINUATION_PROFILE, total_capacity, allocation_noise, cfg=cfg)

    # 6. Finalize Stats (no graduation here — incomplete students keep upgrade status)
    for p in pilots:
        months = cfg.phase_length_days / 30.0
        p.sim_phase = SIM_RAP_MONTHLY * months
        p.update_total()
        p.update_monthly(cfg.phase_length_days)

    cfg.last_phase_upgrade_handoff = build_phase_upgrade_handoff(pilots)
    return pilots

# ----------------------
# Reporting
# ----------------------
def print_phase_summary(pilots: List[Pilot], cfg: SquadronConfig, verbose: bool = True):
    print("\n=== Phase Summary ===")
    
    groups = {
        "MQT Students": [p for p in pilots if p.upgrade == Upgrade.MQT],
        "Incomplete MQT Students": [p for p in pilots if p.upgrade == Upgrade.MQT and p.incomplete_syllabus_items],
        "FLUG Students": [p for p in pilots if p.upgrade == Upgrade.FLUG],
        "Incomplete FLUG Students": [p for p in pilots if p.upgrade == Upgrade.FLUG and p.incomplete_syllabus_items],
        "IPUG Students": [p for p in pilots if p.upgrade == Upgrade.IPUG],
        "Incomplete IPUG Students": [p for p in pilots if p.upgrade == Upgrade.IPUG and p.incomplete_syllabus_items],
        "Line Wingmen": [p for p in pilots if p.qual == Qual.WG and p.upgrade == Upgrade.NONE],
        "Line FLs": [p for p in pilots if p.qual == Qual.FL and p.upgrade == Upgrade.NONE],
        "IPs": [p for p in pilots if p.qual == Qual.IP]
    }

    for name, group in groups.items():
        if not group:
            print(f"{name}: None")
            continue
            
        avg_sorties = sum(p.sortie_monthly for p in group) / len(group)
        avg_blue_sorties = sum(p.sortie_blue_monthly for p in group) / len(group)
        avg_red_sorties = sum(p.sortie_red_monthly for p in group) / len(group)
        print(f"{name} ({len(group)}): Avg Mo. Sorties {avg_sorties:.1f}, Avg Mo. Blue Sorties {avg_blue_sorties:.1f}, Avg Mo. Red Sorties {avg_red_sorties:.1f}")

    if verbose:
        for name, group in groups.items():
            for p in group:
                print(f'{p.qual}/{p.upgrade}: Monthly {p.sortie_monthly}, Blue {p.sortie_blue_monthly}, Red {p.sortie_red_monthly}')