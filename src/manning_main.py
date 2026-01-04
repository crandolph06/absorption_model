from models import SquadronConfig, Pilot, Qual, Upgrade
from manning_engine import CAFSimulation

def setup_debug_simulation():
    sim = CAFSimulation()

    # Sq 1: Large, high-utilization
    # Sq 2: Medium, standard ops
    # Sq 3: Small, low-utilization
    squadrons = [
        SquadronConfig(id=1, paa=24, ute=13.0, total_pilots=40, experience_ratio=0.5, 
                       ip_qty=8, mqt_students=0, flug_students=0, ipug_students=0, pilots=[]),
        SquadronConfig(id=2, paa=18, ute=10.0, total_pilots=30, experience_ratio=0.4, 
                       ip_qty=5, mqt_students=0, flug_students=0, ipug_students=0, pilots=[]),
        SquadronConfig(id=3, paa=12, ute=7.0, total_pilots=25, experience_ratio=0.6, 
                       ip_qty=4, mqt_students=0, flug_students=0, ipug_students=0, pilots=[])
    ]

    # 3. Manually seed squadrons with some initial pilots to prevent div-by-zero
    for sq in squadrons:
        while sum(1 for p in sq.pilots if p.qual == Qual.IP) < sq.ip_qty:
            sq.pilots.append(Pilot(
                qual=Qual.IP,
                year_group=2020, # Need to vary these
                sorties_flown=600, # Need to vary these
                hours_flown=800, # Need to vary these
                squadron_id=sq.id
            ))
        
        # HERE! 
        is_exp = (len(sq.pilots) / sq.total_pilots) < sq.experience_ratio


        for _ in range(sq.total_pilots):
            # Roughly split pilots between WG/FL/IP based on experience_ratio
            has_ips = (sum(1 for p in sq.pilots if p.qual == Qual.IP)) < sq.ip_qty 
            sq.pilots.append(Pilot(
                qual=Qual.IP if is_exp else Qual.WG,
                year_group=2020,
                sorties_flown=300 if is_exp else 100,
                hours_flown=400 if is_exp else 120,
                squadron_id=sq.id
            ))

    return sim, squadrons

if __name__ == "__main__":
    sim, squadrons = setup_debug_simulation()

    # 4. Run for 10 years
    # 15 annual intake, 70% retention
    results_df = sim.run_simulation(
        years_to_run=10, 
        annual_intake=15, 
        retention_rate=0.7, 
        squadron_configs=squadrons
    )

    # 5. Quick Debug Output
    print("--- Simulation Complete ---")
    print(results_df.head(10))
    print(results_df.tail(10))