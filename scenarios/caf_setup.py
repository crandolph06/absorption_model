from engine import CAFSimulation
from models import Qual, Pilot, YearGroup

def initialize_force(sim: CAFSimulation, inventory_data: list):
    """
    inventory_data: list of dicts like 
    [{'year': 2020, 'size': 20, 'qual': Qual.IP, 'hours': 800, 'sorties': 600, 'adsc': 4}, ...]
    """
    for entry in inventory_data:
        pilots = []
        for _ in range(entry['size']):
            p = Pilot(
                year_group=entry['year'],
                sorties_flown=entry['sorties'],
                hours_flown=entry['hours'],
                qual=entry['qual'],
                adsc_remaining=entry['adsc']
            )
            pilots.append(p)
        
        yg = YearGroup(year=entry['year'], pilots=pilots)
        sim.year_groups.append(yg)

def run_scenario():
    sim = CAFSimulation()

    # Define your current real-world spread
    # Year: The year they entered service
    # Size: How many are still in that year group
    starting_inventory = [
        {'year': 2015, 'size': 15, 'qual': Qual.IP, 'hours': 1200, 'sorties': 900, 'adsc': 0},
        {'year': 2018, 'size': 25, 'qual': Qual.FL, 'hours': 600,  'sorties': 450, 'adsc': 2},
        {'year': 2022, 'size': 40, 'qual': Qual.WG, 'hours': 150,  'sorties': 100, 'adsc': 6},
    ]

    # Initialize the force
    for entry in starting_inventory:
        sim.add_existing_year_group(
            year=entry['year'],
            size=entry['size'],
            qual=entry['qual'],
            hours=entry['hours'],
            sorties=entry['sorties'],
            adsc=entry['adsc']
        )

    # Run for 10 years
    for current_year in range(2025, 2035):
        # You can vary the intake and retention per year here
        sim.run_year(current_year, intake_size=30, retention_rate=0.65)

    # Get results
    df = sim.get_dataframe()
    print(df)