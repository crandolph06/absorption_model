from src.models import Pilot, Qual, Upgrade


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

def can_fill_seat(pilot: Pilot, min_qual: Qual) -> bool:
    """
    Master rule: Can this pilot sit in this seat for this specific syllabus event?
    """
    # Backup -- no MQT flying anything other than MQT upgrade sorties.
    if pilot.upgrade == Upgrade.MQT:
        return False

    # IP requirements
    if min_qual == Qual.IP and pilot.qual == Qual.IP:
        return True

    # FLUG requirements
    if min_qual == Qual.FL and pilot.qual == Qual.FL:
        return True
    if min_qual == Qual.FL and pilot.qual == Qual.IP:
        return True

    # WG requirements
    if min_qual == Qual.WG and pilot.qual == Qual.WG:
        return True
    if min_qual == Qual.WG and pilot.qual == Qual.FL:
        return True
    if min_qual == Qual.WG and pilot.qual == Qual.IP:
        return True

    return False