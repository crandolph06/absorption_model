from models import Pilot, Qual, Upgrade

# ---------------------------------------------------------
# 1. ELIGIBILITY RULES (Used during "Selection Phase")
# ---------------------------------------------------------
def can_start_upgrade(pilot: Pilot, upgrade_type: Upgrade) -> bool:
    """
    Determines if a pilot is eligible to BEGIN a specific upgrade.
    Replaces: is_student_eligible
    """
    # Pilots already in an upgrade cannot start another
    if pilot.upgrade != Upgrade.NONE:
        return False

    if upgrade_type in (Upgrade.MQT, Upgrade.FLUG):
        return pilot.qual == Qual.WG

    if upgrade_type == Upgrade.IPUG:
        return pilot.qual == Qual.FL

    return False

# ---------------------------------------------------------
# 2. SEAT RULES (Used during "Sortie Generation Phase")
# ---------------------------------------------------------
def _qual_hierarchy_check(pilot_qual: Qual, seat_required: Qual) -> bool:
    """
    Internal helper: Returns True if pilot rank >= seat rank.
    Replaces: qual_meets_requirement
    """
    # IP can fly everything
    if pilot_qual == Qual.IP:
        return True
    
    # FL can fly FL and WG seats
    if pilot_qual == Qual.FL:
        return seat_required in [Qual.FL, Qual.WG]
    
    # WG can only fly WG seats
    if pilot_qual == Qual.WG:
        return seat_required == Qual.WG
        
    return False

def can_fill_seat(pilot: Pilot, seat_type: Qual, syllabus_upgrade: Upgrade = None) -> bool:
    """
    Master rule: Can this pilot sit in this seat for this specific syllabus event?
    """
    # 1. UPGRADE RESTRICTIONS (The "Lock-in" Rule)
    # If pilot is in MQT, they cannot fly FLUG/IPUG/CT sorties, only MQT.
    if pilot.upgrade == Upgrade.MQT and syllabus_upgrade != Upgrade.MQT:
        return False
    # If pilot is in FLUG, they cannot fly MQT sorties.
    if pilot.upgrade == Upgrade.FLUG and syllabus_upgrade not in [Upgrade.FLUG, None]:
        return False
    # If pilot is in IPUG, they cannot fly other upgrades.
    if pilot.upgrade == Upgrade.IPUG and syllabus_upgrade not in [Upgrade.IPUG, None]:
        return False

    # 2. HIERARCHY CHECK
    return _qual_hierarchy_check(pilot.qual, seat_type)