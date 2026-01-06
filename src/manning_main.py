from src.models import SquadronConfig, Pilot, Qual, Upgrade
from src.manning_engine import CAFSimulation
import random
from typing import Optional

IP_YEAR_START = 2010
IP_YEAR_END = 2014
IP_HOUR_START = 400
IP_HOUR_END = 1500
IP_SORTIE_START = 300
IP_SORTIE_END = 1200

FL_YEAR_START = 2015
FL_YEAR_END = 2023
FL_HOUR_START = 200
FL_HOUR_END = 500
FL_SORTIE_START = 180
FL_SORTIE_END = 400

WG_YEAR_START = 2022
WG_YEAR_END = 2025
WG_HOUR_START = 50
WG_HOUR_END = 250
WG_SORTIE_START = 50
WG_SORTIE_END = 300

path = 'outputs/simulation_results.parquet'

def setup_simulation(sim_upgrades: bool = False):
    sim = CAFSimulation(path, sim_upgrades)

    squadron_manning_targets = [
        {"total": 27, "exp": 0.5}, # Get Exp Ratio from FR1/2
        {"total": 36, "exp": 0.5},
        {"total": 36, "exp": 0.5}, 
        {"total": 36, "exp": 0.5}, 
        {"total": 36, "exp": 0.5},
        {"total": 36, "exp": 0.5},
        {"total": 36, "exp": 0.5}, 
        {"total": 36, "exp": 0.5},
        {"total": 36, "exp": 0.5},
        {"total": 36, "exp": 0.5}, 
        {"total": 36, "exp": 0.5},
        {"total": 36, "exp": 0.5},
        {"total": 36, "exp": 0.5},
        {"total": 27, "exp": 0.5},
        {"total": 27, "exp": 0.5},
        {"total": 32, "exp": 0.5}, 
        {"total": 32, "exp": 0.5},
        {"total": 32, "exp": 0.5},
        {"total": 30, "exp": 0.5}, 
        {"total": 30, "exp": 0.5},
        {"total": 27, "exp": 0.5},
        {"total": 32, "exp": 0.5}, 
        {"total": 35, "exp": 0.5},
        {"total": 27, "exp": 0.5},
        {"total": 32, "exp": 0.5}, 
        {"total": 32, "exp": 0.5},
        {"total": 32, "exp": 0.5},
        {"total": 32, "exp": 0.5}, 
        {"total": 27, "exp": 0.5},
        {"total": 36, "exp": 0.5}
    ]

    # Used 1.5 CCR for all units
    # Used 50% of exp pilots as starting IP value 

    squadrons = [
        SquadronConfig(id=14, paa=18, ute=10.0, ip_qty=7, pilots=[],
                       mqt_students=0, flug_students=0, ipug_students=0), 
        SquadronConfig(id=493, paa=24, ute=10.0, ip_qty=9, pilots=[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=495, paa=24, ute=10.0, ip_qty=9, pilots=[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=95, paa=24, ute=10.0, ip_qty = 9, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=355, paa=24, ute=10.0, ip_qty = 9, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=356, paa=24, ute=10.0, ip_qty = 9, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=4, paa=24, ute=10.0, ip_qty = 9, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=34, paa=24, ute=10.0, ip_qty = 9, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=421, paa=24, ute=10.0, ip_qty = 9, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=27, paa=24, ute=10.0, ip_qty = 9, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=94, paa=24, ute=10.0, ip_qty = 9, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=90, paa=24, ute=10.0, ip_qty = 9, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=525, paa=24, ute=10.0, ip_qty = 9, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=35, paa=18, ute=10.0, ip_qty = 8, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=80, paa=18, ute=10.0, ip_qty = 8, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=55, paa=21, ute=10.0, ip_qty = 8, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=77, paa=21, ute=10.0, ip_qty = 8, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=79, paa=21, ute=10.0, ip_qty = 8, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=510, paa=20, ute=10.0, ip_qty = 8, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=555, paa=20, ute=10.0, ip_qty = 8, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=13, paa=18, ute=10.0, ip_qty = 7, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=36, paa=21, ute=10.0, ip_qty = 8, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=480, paa=23, ute=10.0, ip_qty = 9, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=18, paa=18, ute=10.0, ip_qty = 7, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=335, paa=21, ute=10.0, ip_qty = 8, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=336, paa=21, ute=10.0, ip_qty = 8, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=492, paa=21, ute=10.0, ip_qty = 8, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=494, paa=21, ute=10.0, ip_qty = 8, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=389, paa=18, ute=10.0, ip_qty = 7, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0),
        SquadronConfig(id=391, paa=24, ute=10.0, ip_qty = 9, pilots =[],
                       mqt_students=0, flug_students=0, ipug_students=0)
    ]

    # 3. Manually seed squadrons with some initial pilots to prevent div-by-zero
    for sq, tgt in zip(squadrons, squadron_manning_targets): 
        target_total = tgt["total"]
        target_exp_count = int(target_total * tgt['exp'])

        while sum(1 for p in sq.pilots if p.qual == Qual.IP) < sq.ip_qty:
            year_group = random.randint(IP_YEAR_START, IP_YEAR_END)
            sq.pilots.append(Pilot(
                qual=Qual.IP,
                year_group=year_group,
                adsc_remaining=max(0, 120 - ((sim.current_year - year_group - 2) * 12)),
                sorties_flown=random.randint(IP_SORTIE_START, IP_SORTIE_END), 
                hours_flown=random.randint(IP_HOUR_START, IP_HOUR_END), 
                squadron_id=sq.id
            ))
        
        while sum(1 for p in sq.pilots if p.qual in [Qual.IP, Qual.FL]) < target_exp_count:
            year_group = random.randint(FL_YEAR_START, FL_YEAR_END)
            sq.pilots.append(Pilot(
                qual=Qual.FL,
                year_group=year_group,
                adsc_remaining=max(0, 120 - ((sim.current_year - year_group - 2) * 12)),
                sorties_flown=random.randint(FL_SORTIE_START, FL_SORTIE_END),
                hours_flown=random.randint(FL_HOUR_START, FL_HOUR_END),
                squadron_id=sq.id
            ))

        while len(sq.pilots) < tgt["total"]:
            year_group = random.randint(WG_YEAR_START, WG_YEAR_END)
            sq.pilots.append(Pilot(
                qual=Qual.WG,
                year_group=year_group,
                adsc_remaining=max(0, 120 - ((sim.current_year - year_group - 2) * 12)),
                sorties_flown=random.randint(WG_SORTIE_START, WG_SORTIE_END),
                hours_flown=random.randint(WG_HOUR_START, WG_HOUR_END),
                squadron_id=sq.id
            ))

    return sim, squadrons

if __name__ == "__main__":
    sim, squadrons = setup_simulation()

    # 4. Run for 10 years
    # 15 annual intake, 70% retention
    results_df = sim.run_simulation(
        years_to_run=10, 
        annual_intake=12, 
        retention_rate=0.5, 
        squadron_configs=squadrons
    )

    # 5. Quick Debug Output
    print("--- Simulation Complete ---")
    print(results_df.head(10))
    print(results_df.tail(10))