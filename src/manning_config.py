from src.models import SquadronConfig, Qual, Pilot, Assignment
import random

IP_YEAR_RANGE = (2010, 2014); IP_HOUR_RANGE = (400, 1500); IP_SORTIE_RANGE = (300, 1200)
FL_YEAR_RANGE = (2015, 2023); FL_HOUR_RANGE = (200, 500);  FL_SORTIE_RANGE = (180, 400)
WG_YEAR_RANGE = (2022, 2026); WG_HOUR_RANGE = (50, 250);   WG_SORTIE_RANGE = (50, 300)

SQUADRON_DATA = [
    (14, 18, 7, 27), (493, 24, 9, 36), (495, 24, 9, 36), (95, 24, 9, 36),
    (355, 24, 9, 36), (356, 24, 9, 36), (4, 24, 9, 36), (34, 24, 9, 36),
    (421, 24, 9, 36), (27, 24, 9, 36), (94, 24, 9, 36), (90, 24, 9, 36),
    (525, 24, 9, 36), (35, 18, 8, 27), (80, 18, 8, 27), (55, 21, 8, 32),
    (77, 21, 8, 32), (79, 21, 8, 32), (510, 20, 8, 30), (555, 20, 8, 30),
    (13, 18, 7, 27), (36, 21, 8, 32), (480, 23, 9, 35), (18, 18, 7, 27),
    (335, 21, 8, 32), (336, 21, 8, 32), (492, 21, 8, 32), (494, 21, 8, 32),
    (389, 18, 7, 27), (391, 24, 9, 36)
]

def get_initial_squadrons(current_year: int):
    squadrons = []
    for sq_id, paa, ip_qty, target_total in SQUADRON_DATA:
        sq = SquadronConfig(id=sq_id, paa=paa, ute=10.0, ip_qty=ip_qty, pilots=[])
        
        # 1. Seed IPs
        while sum(1 for p in sq.pilots if p.qual == Qual.IP) < ip_qty:
            yg = random.randint(*IP_YEAR_RANGE)
            sq.pilots.append(Pilot(qual=Qual.IP, year_group=yg, adsc_remaining=max(0, 120-((current_year-yg-2)*12)), 
                             sorties_flown=random.randint(*IP_SORTIE_RANGE), flight_hours_flown=random.randint(*IP_HOUR_RANGE), 
                             squadron_id=sq_id, current_assignment=Assignment.LINE, active=True))
        
        # 2. Seed FLs (target 50% experience ratio)
        target_exp = int(target_total * 0.5)
        while sum(1 for p in sq.pilots if p.qual in [Qual.IP, Qual.FL]) < target_exp:
            yg = random.randint(*FL_YEAR_RANGE)
            sq.pilots.append(Pilot(qual=Qual.FL, year_group=yg, adsc_remaining=max(0, 120-((current_year-yg-2)*12)), 
                             sorties_flown=random.randint(*FL_SORTIE_RANGE), flight_hours_flown=random.randint(*FL_HOUR_RANGE), 
                             squadron_id=sq_id, current_assignment=Assignment.LINE, active=True))

        # 3. Seed WGs to hit total
        while len(sq.pilots) < target_total:
            yg = random.randint(*WG_YEAR_RANGE)
            sq.pilots.append(Pilot(qual=Qual.WG, year_group=yg, adsc_remaining=max(0, 120-((current_year-yg-2)*12)), 
                             sorties_flown=random.randint(*WG_SORTIE_RANGE), flight_hours_flown=random.randint(*WG_HOUR_RANGE), 
                             squadron_id=sq_id, current_assignment=Assignment.LINE, active=True))
        
        sq.update_stats()
        squadrons.append(sq)
    return squadrons