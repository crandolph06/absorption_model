from src.models import Pilot, Qual, Upgrade


def can_start_upgrade(pilot: Pilot, upgrade_type: Upgrade) -> bool:
    # Pilots already in an upgrade cannot start another
    if pilot.upgrade is not Upgrade.NONE:
        return False

    if upgrade_type in (Upgrade.MQT, Upgrade.FLUG):
        return pilot.qual is Qual.WG

    if upgrade_type is Upgrade.IPUG:
        return pilot.qual is Qual.FL

    return False


def can_fill_seat(pilot: Pilot, min_qual: Qual) -> bool:
    """
    Master rule: Can this pilot sit in this seat for this specific syllabus event?
    """
    # Backup -- no MQT flying anything other than MQT upgrade sorties.
    if pilot.upgrade is Upgrade.MQT:
        return False

    qual = pilot.qual
    if min_qual is Qual.IP:
        return qual is Qual.IP
    if min_qual is Qual.FL:
        return qual is Qual.FL or qual is Qual.IP
    if min_qual is Qual.WG:
        return qual is Qual.WG or qual is Qual.FL or qual is Qual.IP
    return False
