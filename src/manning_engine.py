import pandas as pd
from typing import List
from src.models import Pilot, Qual, SquadronConfig, Upgrade, Assignment, AgingRate
import os
import joblib


class CAFSimulation:
    def __init__(self, path: str, sim_upgrades: bool, flug_window_start: int = 250, ipug_window_start: int = 400):
        self.history = []
        self.current_year = 2025
        self.squadrons: List[SquadronConfig] = []
        self.flug_window_start = flug_window_start # Sorties for FLUG auto-start
        self.ipug_window_start = ipug_window_start # Hours for IPUG auto-start

        if not os.path.exists(path):
            raise FileNotFoundError(f'Lookup File Not Found at {path}.')    
        
        brain_path = "sortie_brain.pkl"
        if os.path.exists(brain_path):
            print(f"🧠 Loading Sortie Brain from {brain_path}...")
            self.brain = joblib.load(brain_path)
        else:
            raise FileNotFoundError(f"Could not find {brain_path}. Please run train_brain.py first.")

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
    
    def reset(self):
        self.history = []
        self.current_year = 2025

    def add_new_bcourse_graduates(self, year: int, count: int): # TODO Consider sorting squadrons based on experience ratio and not distributing B-Coursers equally. 
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
            exp_pilots = sum(1 for p in sq.pilots if p.active and p.current_assignment == Assignment.LINE and p.qual != Qual.WG)
            sq.experience_ratio = exp_pilots / sq.total_pilots


    def run_simulation(self, years_to_run: int, annual_intake: int, retention_rate: float, squadron_configs: List[SquadronConfig], ute: float = 10.0):
        """
        squadron_configs: list -> [Config(id=1, paa=12...), Config(id=2, paa=24...)]
        """
        self.history = []
        self.squadrons = squadron_configs

        for sq in self.squadrons:
            sq.ute = ute # TODO UTE not behaving correctly throughout simulation in Streamlit
            phase_length_months = sq.phase_length_days / 30

        for year in range(self.current_year, self.current_year + years_to_run):
            phase_intake = annual_intake // 3
            remainder = annual_intake % 3

            for phase_num in range(1, 4): 
                current_batch = phase_intake + (remainder if phase_num == 3 else 0)
                self.add_new_bcourse_graduates(year, current_batch)

                for sq in self.squadrons:

                    mqt_count, flug_count, ipug_count = sq.new_phase_upgrades(self.flug_window_start, self.ipug_window_start)

                    if sq.flug_students != 0:
                        raise AssertionError(f'Critical Data Mismatch in Squadron Pilots')
                    
                    sq.mqt_students = mqt_count
                    sq.flug_students = flug_count
                    sq.ipug_students = ipug_count
                    
                    if self.sim_upgrades:
                        rates = sq.predict_aging_rate(self.brain)
                                       
                    else:
                        rates = sq.calc_aging_rate(self.sim_upgrades)

                    sq.apply_phase_aging(rates)

                    current_stats = sq.store_stats(year, phase_num, rates)

                    self.process_end_of_phase(sq, year, phase_num, retention_rate, current_stats) # TODO aging rates and manning percentage not populating correctly in Streamlit
            
        return pd.DataFrame(self.history)
    

    def process_end_of_phase(self, sq: SquadronConfig, year: int, phase_num: int, retention_rate: float, current_stats: dict):
        
        staff_ips = 0
        staff_fls = 0
        separated_count = 0
        retained_count = 0

        sq.graduate_current_upgrades()

        sq.send_to_staff()

        for p in sq.pilots:
            p.check_retention(year, phase_num, retention_rate)

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
                if p.upgrade != Upgrade.NONE:
                    raise AssertionError(f'Pilots are moving to staff in an upgrade status. Check pilot logic.')

        current_stats['staff_ips'] = staff_ips
        current_stats['staff_fls'] = staff_fls
        current_stats['separated'] = separated_count
        current_stats['retained'] = retained_count
        
        self.history.append(current_stats)

        active_pilots_only = []
        for p in sq.pilots:
            p.reset_phase_counters()
            if p.active:
                active_pilots_only.append(p)
            
        sq.pilots = active_pilots_only

        # print(f'{year}-P{phase_num} -- {sq.id} FS: {sq.total_pilots} pilots, {sq.ip_qty} IPs, {sq.experience_ratio}')
            

