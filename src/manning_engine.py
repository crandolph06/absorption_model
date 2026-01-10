import pandas as pd
from typing import List
from src.models import Pilot, Qual, SquadronConfig, Upgrade, Assignment, AgingRate
import os
from debug_lookup import diagnose_lookup


class CAFSimulation:
    def __init__(self, path: str, sim_upgrades: bool, flug_window_start: int = 250, ipug_window_start: int = 400):
        self.history = []
        self.current_year = 2025
        self.squadrons: List[SquadronConfig] = []
        self.flug_window_start = flug_window_start # Sorties for FLUG auto-start
        self.ipug_window_start = ipug_window_start # Hours for IPUG auto-start

        if not os.path.exists(path):
            raise FileNotFoundError(f'Lookup File Not Found at {path}.')    
        
        self.df = pd.read_parquet(path)
        self.sim_upgrades = sim_upgrades

        self.base_cols = ['paa', 'ute', 'total_pilots', 'ip_qty', 'exp_ratio']
        self.student_cols = ['mqt_count', 'flug_count', 'ipug_count']

        self.valid_base_cols = [c for c in self.base_cols if c in self.df.columns]
        self.valid_stud_cols = [c for c in self.student_cols if c in self.df.columns]

        base_data = self.df[self.valid_base_cols].values 
        base_std = base_data.std(axis=0)
        base_std[base_std == 0] = 1.0 # Prevent div/0
        self.norm_base_matrix = base_data / base_std 
        self.base_std = base_std 

        if self.sim_upgrades and self.valid_stud_cols:
            stud_data = self.df[self.valid_stud_cols].values
            stud_std = stud_data.std(axis=0)
            stud_std[stud_std == 0] = 1.0
            self.norm_stud_matrix = stud_data / stud_std
            self.stud_std = stud_std
        else:
            self.norm_stud_matrix = None
            self.stud_std = None

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

        for sq in self.squadrons:
            mqt_count = sum(1 for p in sq.pilots if p.active and p.upgrade == Upgrade.MQT)
            sq.mqt_students = mqt_count
            sq.total_pilots = sum(1 for p in sq.pilots if p.active and p.current_assignment == Assignment.LINE)


    # def run_simulation(self, years_to_run: int, annual_intake: int, retention_rate: float, squadron_configs: List[SquadronConfig], ute: float = 10.0):
    def run_simulation(self, years_to_run: int, annual_intake: int, retention_rate: float, squadron_configs: List[SquadronConfig], PATH, priority_vars, ute: float = 10.0):    
        """
        squadron_configs: list -> [Config(id=1, paa=12...), Config(id=2, paa=24...)]
        """
        self.history = []
        self.squadrons = squadron_configs

        for sq in self.squadrons:
            sq.ute = ute # TODO UTE not behaving correctly throughout simulation in Streamlit

        for year in range(self.current_year, self.current_year + years_to_run):
            phase_intake = annual_intake // 3
            remainder = annual_intake % 3

            for phase_num in range(1, 4): 
                current_batch = phase_intake + (remainder if phase_num == 3 else 0)
                self.add_new_bcourse_graduates(year, current_batch)

                for sq in self.squadrons:
                    sq_params = {
                        'paa': sq.paa, 'ute': sq.ute, 'total_pilots': sq.total_pilots, 
                        'ip_qty': sq.ip_qty, 'exp_ratio': sq.experience_ratio
                    }
                    if self.sim_upgrades:
                        sq_params['mqt_qty'] = sq.mqt_students
                        sq_params['flug_qty'] = sq.flug_students
                        sq_params['ipug_qty'] = sq.ipug_students

                    mqt_count, flug_count, ipug_count = sq.new_phase_upgrades(self.flug_window_start, self.ipug_window_start)

                    if sq.flug_students != 0:
                        raise AssertionError(f'Critical Data Mismatch in Squadron Pilots')
                    
                    sq.mqt_students = mqt_count
                    sq.flug_students = flug_count
                    sq.ipug_students = ipug_count
                    
                    rates = sq.lookup_aging_rate(
                        sq_params, 
                        self.df, 
                        self.norm_base_matrix, self.base_std, self.valid_base_cols,
                        self.norm_stud_matrix, self.stud_std, self.valid_stud_cols,
                        self.sim_upgrades
                    )

                    # rates = sq.calc_aging_rate(sim_upgrades=False)

                    sq.apply_phase_aging(rates)

                    # print(f'Phase {year}, {phase_num} | Sq: {sq.id}')

                    # diagnose_lookup(PATH, sq_params,  priority_vars)
            
                    self.process_end_of_phase(sq, year, phase_num, retention_rate, rates) # TODO aging rates and manning percentage not populating correctly in Streamlit
            
        return pd.DataFrame(self.history)

    def process_end_of_phase(self, sq: SquadronConfig, year: int, phase_num: int, retention_rate: float, rates: AgingRate):
        for p in self.active_pilots:
            p.check_retention(year, phase_num, retention_rate)

        months = sq.phase_length_days / 30
        limit = sq.manning_limit

        wg_count = 0
        fl_count = 0
        ip_count = 0
        staff_ips = 0
        staff_fls = 0
        separated_count = 0
        retained_count = 0
        line_pilot_count = 0

        for p in sq.pilots:
            if not p.active:
                if p.separation_date == (year, phase_num):
                    separated_count += 1
                continue

            if p.adsc_remaining == 24.1:
                p.adsc_remaining = 24
                retained_count += 1

            if p.current_assignment == Assignment.STAFF:
                if p.qual == Qual.IP: staff_ips += 1
                elif p.qual == Qual.FL: staff_fls += 1 

            elif p.current_assignment == Assignment.LINE:
                line_pilot_count += 1
                if p.qual == Qual.WG: wg_count += 1
                elif p.qual == Qual.FL: fl_count += 1
                elif p.qual == Qual.IP: ip_count += 1
        
        exp_ratio = 0
        if line_pilot_count > 0:
            exp_ratio = (fl_count + ip_count) / line_pilot_count

        current_stats = {
            'year': year,
            'phase': phase_num,
            'squadron_id': sq.id,
            'wg_count': wg_count,
            'fl_count': fl_count,
            'ip_count': ip_count,
            'percent_manned': line_pilot_count / limit,
            'total_pilots': line_pilot_count,
            'exp_rat': exp_ratio,
            'staff_ips': staff_ips,
            'staff_fls': staff_fls,
            'separated': separated_count,
            'retained': retained_count,
            'wg_rate_mo': rates.wg_phase / months,
            'fl_rate_mo': rates.fl_phase / months,
            'ip_rate_mo': rates.ip_phase / months,
            'wg_rate_blue': rates.wg_blue_phase / months,
            'fl_rate_blue': rates.fl_blue_phase / months,
            'ip_rate_blue': rates.ip_blue_phase / months
        }
    
        self.history.append(current_stats)

        sq.graduate_current_upgrades()

        current_line_pilots = []
        for p in sq.pilots:
            if p.active and p.current_assignment == Assignment.LINE:
                current_line_pilots.append(p)

        if len(current_line_pilots) > limit:
            excess_count = len(current_line_pilots) - limit

            ips = []
            fls = []
            for p in current_line_pilots:
                if p.qual == Qual.IP: ips.append(p)
                elif p.qual == Qual.FL: fls.append(p)

            ips.sort(key=lambda x: x.year_group)
            fls.sort(key=lambda x: x.year_group)

            eligible_ips = ips[3:] if len(ips) > 3 else [] # Protects Sq/CC, DO, and WO

            funnel_queue = eligible_ips + fls
            movers_count = min(excess_count, len(funnel_queue))
        
            for i in range(int(movers_count)): # Not sure why streamlit thinks this is a float
                funnel_queue[i].move_to_staff()

        active_pilots_only = []
        for p in sq.pilots:
            p.reset_phase_counters()
            if p.active:
                active_pilots_only.append(p)
            
        sq.pilots = active_pilots_only
            

