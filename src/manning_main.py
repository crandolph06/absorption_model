from models import SquadronConfig, Pilot, Qual, Upgrade
from manning_engine import CAFSimulation
import random

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


def setup_debug_simulation():
    sim = CAFSimulation()

    # Sq 1: Large, high-utilization
    # Sq 2: Medium, standard ops
    # Sq 3: Small, low-utilization
    squadron_manning_targets = [
        {"total": 40, "exp": 0.5},
        {"total": 30, "exp": 0.4},
        {"total": 25, "exp": 0.35}

    ]
    squadrons = [
        SquadronConfig(id=1, paa=24, ute=13.0, mqt_students=0, flug_students=0, 
                       ip_qty=8, ipug_students=0, pilots=[]),
        SquadronConfig(id=2, paa=18, ute=10.0, mqt_students=0, flug_students=0, 
                       ip_qty=6, ipug_students=0, pilots=[]),
        SquadronConfig(id=3, paa=12, ute=7.0, mqt_students=0, flug_students=0, 
                       ip_qty=4, ipug_students=0, pilots=[])
    ]

    # 3. Manually seed squadrons with some initial pilots to prevent div-by-zero
    for sq, tgt in zip(squadrons, squadron_manning_targets):
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
        
        while (len(sq.pilots) / sq.total_pilots) < sq.experience_ratio:
            year_group = random.randint(FL_YEAR_START, FL_YEAR_END)
            sq.pilots.append(Pilot(
                qual=Qual.FL,
                year_group=year_group,
                adsc_remaining=max(0, 120 - ((sim.current_year - year_group - 2) * 12)),
                sorties_flown=random.randint(FL_SORTIE_START, FL_SORTIE_END),
                hours_flown=random.randint(FL_HOUR_START, FL_HOUR_END),
                squadron_id=sq.id
            ))

        while len(sq.pilots) < sq.total_pilots:
            year_group = random.randint(WG_YEAR_START, WG_YEAR_END)
            sq.pilots.append(Pilot(
                qual=Qual.WG,
                year_group=year_group,
                adsc_remaining=max(120 - ((sim.current_year - year_group - 2) * 12)),
                sorties_flown=random.randint(WG_SORTIE_START, WG_SORTIE_END),
                hours_flown=random.randint(WG_HOUR_START, WG_HOUR_END),
                squadron_id=sq.id
            ))

    return sim, squadrons

if __name__ == "__main__":
    sim, squadrons = setup_debug_simulation()

    # 4. Run for 10 years
    # 15 annual intake, 70% retention
    results_df = sim.run_simulation(
        years_to_run=10, 
        annual_intake=50, 
        retention_rate=0.5, 
        squadron_configs=squadrons
    )

    # 5. Quick Debug Output
    print("--- Simulation Complete ---")
    print(results_df.head(10))
    print(results_df.tail(10))