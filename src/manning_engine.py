import pandas as pd
from typing import List
from models import Pilot, Qual, SquadronConfig, Upgrade

FLUG_WINDOW_START = 250 # Sorties
FLUG_WINDOW_END = 265 # Sorties
IPUG_WINDOW_START = 400 # Hours
IPUG_WINDOW_END = 430 # Hours

class CAFSimulation:
    def __init__(self, path):
        self.pilots: List[Pilot] = []
        self.history = []
        self.current_year = 2024
        self.squadrons: List[SquadronConfig] = []
        self.lookup_table = pd.read_csv(path)

    def add_new_bcourse_graduates(self, year: int, count: int):
        """Distributes a total 'count' of WG pilots across all active squadrons."""
        num_sq = len(self.squadrons)
        if num_sq == 0:
            return

        for i in range(count):
            target_sq = self.squadrons[i % num_sq]
            
            self.pilots.append(Pilot(
                qual=Qual.WG,
                upgrade=Upgrade.MQT,
                year_group=year,
                adsc_remaining=120, 
                active=True,
                squadron_id=target_sq.id 
            ))

    def calculate_upgrade_demand(self, sqdn_pilots: List[Pilot]):
        mqt_qty = sum(1 for p in sqdn_pilots if p.qual == Qual.WG and p.upgrade == Upgrade.MQT)
        flug_qty = sum(1 for p in sqdn_pilots if FLUG_WINDOW_START <= p.sorties_flown <= FLUG_WINDOW_END)
        ipug_qty = sum(1 for p in sqdn_pilots if IPUG_WINDOW_START <= p.hours_flown <= IPUG_WINDOW_END)

        return mqt_qty, flug_qty, ipug_qty

    def check_for_upgrades(self, p: Pilot, sim_upgrades: bool = False):
        if not sim_upgrades:
            if p.qual == Qual.WG and p.sorties_flown >= FLUG_WINDOW_START:
                p.Qual = Qual.FL

        elif p.qual == Qual.FL and p.hours_flown >= IPUG_WINDOW_START:
            p.qual = Qual.IP

        else:
            # Placeholder for syllabus logic
            # Take number of upgradees and input to SquadronConfig
            # Assess upgrade completion and input to SquadronConfig
            pass

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

    def run_phase(self, config: SquadronConfig):
        sqdn_pilots = [p for p in self.pilots if p.active and p.squadron_id == config.id]
        
        if not sqdn_pilots:
            return
        
        mqt, flug, ipug = self.calculate_upgrade_demand(sqdn_pilots)
        total_p = len(sqdn_pilots)
        exp_ratio = config.experience_ratio

        wg_rate = 6.0 # TODO Point to PySR Function or CSV lookup
        fl_rate = 8.0 # TODO Point to PySR Function or CSV lookup
        ip_rate = 10.0 # TODO Point to PySR Function or CSV lookup

        asd = config.avg_sortie_dur

        for p in sqdn_pilots:
            p_rate = wg_rate
            p.age_one_phase_with_rates(p_rate, p_rate * asd)

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
            p.reset_phase_counters()