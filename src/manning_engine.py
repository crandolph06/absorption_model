import pandas as pd
from typing import List
from models import Pilot, Qual, SquadronConfig, Upgrade, AgingRate


class CAFSimulation:
    def __init__(self):
        self.history = []
        self.current_year = 2024
        self.squadrons: List[SquadronConfig] = []

    @property
    def all_pilots(self):
        return [p for sq in self.squadrons for p in sq.pilots]
    
    @property
    def active_pilots(self):
        return[p for sq in self.squadrons for p in sq.pilots if p.active]
    
    @property
    def total_pilot_count(self):
        return len(self.all_pilots)

    def add_new_bcourse_graduates(self, year: int, count: int):
        num_sq = len(self.squadrons)
        if num_sq == 0:
            return

        for i in range(count):
            target_sq = self.squadrons[i % num_sq]
            
            new_pilot = (Pilot(
                qual=Qual.WG,
                upgrade=Upgrade.MQT,
                year_group=year,
                adsc_remaining=120, 
                active=True,
                squadron_id=target_sq.id,
                hours_flown=50,
                sorties_flown=50 
            ))

            target_sq.pilots.append(new_pilot)


    def run_simulation(self, years_to_run: int, annual_intake: int, retention_rate: float, squadron_configs: List[SquadronConfig]):
        """
        squadron_configs: list -> [Config(id=1, paa=12...), Config(id=2, paa=24...)]
        """
        self.squadrons = squadron_configs
        end_year = self.current_year + years_to_run

        for year in range(self.current_year, end_year):
            phase_intake = annual_intake // 3
            remainder = annual_intake % 3

            for phase_num in range(1, 4): 
                current_batch = phase_intake + (remainder if phase_num == 3 else 0)
                self.add_new_bcourse_graduates(year, current_batch)

                for config in squadron_configs:
                    self.run_phase(config)
            
            self.process_end_of_phase(year, phase_num, retention_rate, sim_upgrades=False)
            
        return pd.DataFrame(self.history)

    def process_end_of_phase(self, year: int, phase_num: int, retention_rate: float, sim_upgrades: bool = False):
        pilots_retained = 0
        pilots_separated = 0
        
        for p in self.pilots:
            if not p.active:
                continue

            if p.adsc_remaining <= 0:
                p.check_retention(retention_rate)

                if not p.active:
                    pilots_separated += 1
                    continue

                else:
                    pilots_retained += 1
            
            self.check_for_upgrades(p, sim_upgrades)
        
        active_pilots = [p for p in self.pilots if p.active]

        current_stats = {
            'year': year,
            'phase': phase_num,
            'wg_count': sum(1 for p in active_pilots if p.qual == Qual.WG),
            'fl_count': sum(1 for p in active_pilots if p.qual == Qual.FL),
            'ip_count': sum (1 for p in active_pilots if p.qual == Qual.IP),
            'total_pilots': sum(1 for p in active_pilots),
            'retained': pilots_retained,
            'separated': pilots_separated
        }

        if current_stats['total_pilots'] > 0:
            exp_pilots = current_stats['fl_count'] + current_stats['ip_count']
            current_stats['exp_rat'] = exp_pilots / current_stats['total_pilots']
        else:
            current_stats['exp_rat'] = 0

        self.history.append(current_stats)

        for p in active_pilots:
            p.graduate()
            p.reset_phase_counters()