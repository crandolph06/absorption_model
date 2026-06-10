from src.models import SquadronConfig, Qual, Pilot, Assignment, Upgrade
import random

IP_YEAR_RANGE = (2008, 2016); IP_HOUR_RANGE = (300, 1500); IP_SORTIE_RANGE = (270, 1200)
FL_YEAR_RANGE = (2016, 2024); FL_HOUR_RANGE = (200, 500);  FL_SORTIE_RANGE = (150, 400)
WG_YEAR_RANGE = (2024, 2026); WG_HOUR_RANGE = (50, 350);   WG_SORTIE_RANGE = (50, 325)

SQUADRON_DATA = [
    (14, 18, 7, 27, 10.0, (0, 0, 0)), (493, 24, 9, 36, 10.0, (0, 0, 0)), (495, 24, 9, 36, 10.0, (0, 0, 0)), (95, 24, 9, 36, 10.0, (0, 0, 0)),
    (355, 24, 9, 36, 10.0, (0, 0, 0)), (356, 24, 9, 36, 10.0, (0, 0, 0)), (4, 24, 9, 36, 10.0, (0, 0, 0)), (34, 24, 9, 36, 10.0, (0, 0, 0)),
    (421, 24, 9, 36, 10.0, (0, 0, 0)), (27, 24, 9, 36, 10.0, (0, 0, 0)), (94, 24, 9, 36, 10.0, (0, 0, 0)), (90, 24, 9, 36, 10.0, (0, 0, 0)),
    (525, 24, 9, 36, 10.0, (0, 0, 0)), (35, 18, 8, 27, 10.0, (0, 0, 0)), (80, 18, 8, 27, 10.0, (0, 0, 0)), (55, 21, 8, 32, 10.0, (0, 0, 0)),
    (77, 21, 8, 32, 10.0, (0, 0, 0)), (79, 21, 8, 32, 10.0, (0, 0, 0)), (510, 20, 8, 30, 10.0, (0, 0, 0)), (555, 20, 8, 30, 10.0, (0, 0, 0)),
    (13, 18, 7, 27, 10.0, (0, 0, 0)), (36, 21, 8, 32, 10.0, (0, 0, 0)), (480, 23, 9, 35, 10.0, (0, 0, 0)), (18, 18, 7, 27, 10.0, (0, 0, 0)),
    (335, 21, 8, 32, 10.0, (0, 0, 0)), (336, 21, 8, 32, 10.0, (0, 0, 0)), (492, 21, 8, 32, 10.0, (0, 0, 0)), (494, 21, 8, 32, 10.0, (0, 0, 0)),
    (389, 18, 7, 27, 10.0, (0, 0, 0)), (391, 24, 9, 36, 10.0, (0, 0, 0))
]

TEST_SQUADRON_DATA = [
    (14, 10, 2, 10, 3.0, (2, 0, 0))#, (493, 10, 2, 10), (495, 10, 2, 10), (95, 10, 2, 10) # All units 10 PAA, 2 IPs, 10 pilots total
]

def get_initial_squadrons(current_year: int, squadron_data: list | None = None):
    if squadron_data is None:
        squadron_data = SQUADRON_DATA
    squadrons = []
    for sq_id, paa, ip_qty, target_total, ute, upgradees in squadron_data:
        sq = SquadronConfig(id=sq_id, paa=paa, ute=ute, ip_qty=ip_qty, pilots=[])
        
        # 1. Seed IPs
        while sum(1 for p in sq.pilots if p.qual == Qual.IP) < ip_qty:
            yg = random.randint(*IP_YEAR_RANGE)
            sq.pilots.append(Pilot(qual=Qual.IP, upgrade=Upgrade.NONE, year_group=yg, adsc_remaining=max(0, 120-((current_year-yg-2)*12)), 
                             sorties_flown=random.randint(*IP_SORTIE_RANGE), flight_hours_flown=random.randint(*IP_HOUR_RANGE), 
                             squadron_id=sq_id, current_assignment=Assignment.LINE, active=True))
        
        # 2. Seed FLs (target 50% experience ratio)
        target_exp = int(target_total * 0.5)
        while sum(1 for p in sq.pilots if p.qual in [Qual.IP, Qual.FL]) < target_exp:
            yg = random.randint(*FL_YEAR_RANGE)
            sq.pilots.append(Pilot(qual=Qual.FL, upgrade=Upgrade.NONE, year_group=yg, adsc_remaining=max(0, 120-((current_year-yg-2)*12)), 
                             sorties_flown=random.randint(*FL_SORTIE_RANGE), flight_hours_flown=random.randint(*FL_HOUR_RANGE), 
                             squadron_id=sq_id, current_assignment=Assignment.LINE, active=True))

        # 3. Seed WGs to hit total
        while len(sq.pilots) < target_total:
            yg = random.randint(*WG_YEAR_RANGE)
            sq.pilots.append(Pilot(qual=Qual.WG, year_group=yg, adsc_remaining=max(0, 120-((current_year-yg-2)*12)), 
                             sorties_flown=random.randint(*WG_SORTIE_RANGE), flight_hours_flown=random.randint(*WG_HOUR_RANGE), 
                             squadron_id=sq_id, current_assignment=Assignment.LINE, active=True))
        
        # 4. Seed upgrades within current manning
        mqt, flug, ipug = upgradees if upgradees else (0, 0, 0)

        if mqt > 0: # Default to youngest wingmen
            mqt_eligible_wingmen = [p for p in sq.pilots if p.qual == Qual.WG] 
            mqt_eligible_wingmen.sort(key=lambda x: x.year_group)
            if len(mqt_eligible_wingmen) < mqt:
                raise ValueError(f"Not enough wingmen eligible for MQT upgrades: {len(mqt_eligible_wingmen)} available, {mqt} requested")
            for i in range(mqt):
                mqt_eligible_wingmen[i].upgrade = Upgrade.MQT
        if flug > 0: # Default to oldest wingmen
            flug_eligible_wingmen = [p for p in sq.pilots if p.qual == Qual.WG and p.upgrade == Upgrade.NONE] 
            if len(flug_eligible_wingmen) < flug:
                raise ValueError(f"Not enough wingmen eligible for FLUG upgrades: {len(flug_eligible_wingmen)} available, {flug} requested")
            flug_eligible_wingmen.sort(key=lambda x: x.year_group, reverse=True)
            for i in range(flug):
                flug_eligible_wingmen[i].upgrade = Upgrade.FLUG
        if ipug > 0: # Default to oldest flight leads
            ipug_eligible_flight_leads = [p for p in sq.pilots if p.qual == Qual.FL] 
            if len(ipug_eligible_flight_leads) < ipug:
                raise ValueError(f"Not enough flight leads eligible for IPUG upgrades: {len(ipug_eligible_flight_leads)} available, {ipug} requested")
            ipug_eligible_flight_leads.sort(key=lambda x: x.year_group, reverse=True)
            for i in range(ipug):
                ipug_eligible_flight_leads[i].upgrade = Upgrade.IPUG
        
        sq.update_stats()
        squadrons.append(sq)
    return squadrons