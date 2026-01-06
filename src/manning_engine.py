import pandas as pd
from typing import List
from src.models import Pilot, Qual, SquadronConfig, Upgrade, Assignment
import os


class CAFSimulation:
    def __init__(self, path: str, sim_upgrades: bool = False):
        self.history = []
        self.current_year = 2025
        self.squadrons: List[SquadronConfig] = []

        if not os.path.exists(path):
            raise FileNotFoundError(
            'Lookup File Not Found. Simulation Stopped. ' \
            'Please check the file and try again.'
            )    
        
        self.df = pd.read_parquet(path)

        self.sim_upgrades = sim_upgrades

    @property
    def all_pilots(self):
        return [p for sq in self.squadrons for p in sq.pilots]
    
    @property
    def total_pilot_count(self):
        return len(self.all_pilots)
    
    @property
    def active_pilots(self):
        return [p for p in self.all_pilots if p.active]
    
    @property
    def total_active_pilot_count(self):
        return len(self.active_pilots)
    
    @property
    def line_pilots(self):
        return [p for p in self.active_pilots if p.current_assignment == Assignment.LINE]
    
    @property
    def total_line_pilot_count(self):
        return len(self.line_pilots)

    @property
    def staff_pilots(self):
        return [p for p in self.active_pilots if p.current_assignment == Assignment.STAFF]
    
    @property
    def total_staff_pilot_count(self):
        return len(self.staff_pilots)

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


    def run_simulation(self, years_to_run: int, annual_intake: int, retention_rate: float, squadron_configs: List[SquadronConfig], ute: float = 10.0):
        """
        squadron_configs: list -> [Config(id=1, paa=12...), Config(id=2, paa=24...)]
        """
        self.squadrons = squadron_configs
        for year in range(self.current_year, self.current_year + years_to_run):
            phase_intake = annual_intake // 3
            remainder = annual_intake % 3

            for phase_num in range(1, 4): 
                current_batch = phase_intake + (remainder if phase_num == 3 else 0)
                self.add_new_bcourse_graduates(year, current_batch)

                for sq in self.squadrons:
                    sq.ute = ute
                    rates = sq.calculate_aging_rates()
                    sq.apply_phase_aging(rates)
            
                self.process_end_of_phase(year, phase_num, retention_rate)
            
        return pd.DataFrame(self.history)

    def process_end_of_phase(self, year: int, phase_num: int, retention_rate: float):
        for p in self.active_pilots:
            p.check_retention(year, phase_num, retention_rate)

        for sq in self.squadrons:

            # current_rates = sq.calculate_aging_rates()

            sq_params = {
                'paa': sq.paa, 'ute': sq.ute, 'total_pilots': len(sq.pilots), 
                'ip_qty': sq.ip_qty, 'exp_ratio': sq.experience_ratio}
            
            current_rates = sq.lookup_aging_rate(sq_params, self.df, self.sim_upgrades)

            months = sq.phase_length_days / 30

            num_sep = sum(1 for p in sq.pilots if not p.active and p.separation_date == (year, phase_num))
            retained_list = [p for p in sq.pilots if p.adsc_remaining == 24.1]
            num_ret = len(retained_list)

            for p in retained_list:
                p.adsc_remaining = 24
            
            staff_ips = sum(1 for p in sq.pilots if p.active and p.current_assignment == Assignment.STAFF and p.qual == Qual.IP)
            staff_fls = sum(1 for p in sq.pilots if p.active and p.current_assignment == Assignment.STAFF and p.qual == Qual.FL)

            limit = sq.manning_limit
            line_roster = [p for p in sq.pilots if p.active and p.current_assignment == Assignment.LINE]
            total_line = sum(1 for p in line_roster if p.active and p.current_assignment == Assignment.LINE)

            current_stats = {
                'year': year,
                'phase': phase_num,
                'squadron_id': sq.id,
                'wg_count': sum(1 for p in line_roster if p.qual == Qual.WG),
                'fl_count': sum(1 for p in line_roster if p.qual == Qual.FL),
                'ip_count': sum(1 for p in line_roster if p.qual == Qual.IP),
                'percent_manned': total_line / limit,
                'total_pilots': total_line,
                'staff_ips': staff_ips,
                'staff_fls': staff_fls,
                'separated': num_sep,
                'retained': num_ret,
                'wg_rate_mo': current_rates.wg_phase / months,
                'fl_rate_mo': current_rates.fl_phase / months,
                'ip_rate_mo': current_rates.ip_phase / months,
            }
        
            if current_stats['total_pilots'] > 0:
                exp_pilots = current_stats['fl_count'] + current_stats['ip_count']
                current_stats['exp_rat'] = exp_pilots / current_stats['total_pilots']

            else:
                current_stats['exp_rat'] = 0
            
            self.history.append(current_stats)

            sq.graduate_current_upgrades()

            if total_line > limit:
                excess_count = total_line - limit

                ips = [p for p in line_roster if p.qual == Qual.IP]
                fls = [p for p in line_roster if p.qual == Qual.FL]
                wgs = [p for p in line_roster if p.qual == Qual.WG]

                ips.sort(key=lambda x: x.year_group)

                protected_ips = ips[:3] # Sq/CC, DO, and WO
                eligible_ips = ips[3:]

                funnel_queue = eligible_ips + sorted(fls, key=lambda x: x.year_group)
                movers = min(excess_count, len(funnel_queue))
            
                for i in range(int(movers)): 
                    funnel_queue[i].move_to_staff()

            for p in sq.pilots:
                p.reset_phase_counters()
                
            sq.pilots = [p for p in sq.pilots if p.active]
            num_ips = sum(1 for p in sq.pilots if p.active and p.qual == Qual.IP)
            sq.ip_qty = num_ips
            

