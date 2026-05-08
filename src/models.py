from dataclasses import dataclass, field
from enum import Enum
import random
from typing import List
import pandas as pd

# ----------------------
# Math 
# ----------------------
def inv(x): return 1/x if x != 0 else 0
def square(x): return x**2

# ----------------------
# Enums & Simple Classes
# ----------------------

class Qual(Enum):
    WG = 'WG'
    FL = 'FL'
    IP = 'IP'

class Upgrade(Enum):
    NONE = 'None'
    MQT = 'MQT'
    FLUG = 'FLUG'
    IPUG = 'IPUG'

class EventType(Enum):
    SORTIE = "sortie"
    SIM = "sim"

class Assignment(Enum):
    LINE = 'LINE'
    STAFF = 'STAFF'
    TRAINING = 'TRAINING'

class PriorityMode(Enum):
    FL_FIRST = 'fl_first'
    IP_FIRST = 'ip_first'
    RANDOM = 'random'



@dataclass 
class AgingRate:
    mqt_phase: float = 0.0
    wg_phase: float = 0.0
    fl_phase: float = 0.0
    ip_phase: float = 0.0

    mqt_blue_phase: float = 0.0
    wg_blue_phase: float = 0.0
    fl_blue_phase: float = 0.0
    ip_blue_phase: float = 0.0

    def monthly_to_phase(self, phase_length_days):
        phase_length_months = phase_length_days / 30

        return AgingRate(
            mqt_phase=self.mqt_phase * phase_length_months,
            wg_phase=self.wg_phase * phase_length_months,
            fl_phase=self.fl_phase * phase_length_months,
            ip_phase=self.ip_phase * phase_length_months,

            mqt_blue_phase=self.mqt_blue_phase * phase_length_months,
            wg_blue_phase=self.wg_blue_phase * phase_length_months,
            fl_blue_phase=self.fl_blue_phase * phase_length_months,
            ip_blue_phase=self.ip_blue_phase * phase_length_months
        )
    
    def phase_to_monthly(self, phase_length_days):
        phase_length_months = phase_length_days / 30

        return AgingRate(
            mqt_phase=self.mqt_phase / phase_length_months,
            wg_phase=self.wg_phase / phase_length_months,
            fl_phase=self.fl_phase / phase_length_months,
            ip_phase=self.ip_phase / phase_length_months,

            mqt_blue_phase=self.mqt_blue_phase / phase_length_months,
            wg_blue_phase=self.wg_blue_phase / phase_length_months,
            fl_blue_phase=self.fl_blue_phase / phase_length_months,
            ip_blue_phase=self.ip_blue_phase / phase_length_months
        )
# ----------------------
# Pilot Entity
# ----------------------
@dataclass
class Pilot:
    qual: Qual = Qual.WG 
    upgrade: Upgrade = Upgrade.NONE
    sortie_phase: float = 0 
    hours_phase: float = 0
    sim_phase: float = 0 
    total_phase: float = 0 
    sortie_blue_phase: float = 0 
    sortie_red_phase: float = 0 

    target_sorties: float = 0
    rap_shortfall: float = 0

    sortie_monthly: float = 0
    sim_monthly: float = 0
    sortie_blue_monthly: float = 0
    sortie_red_monthly: float = 0

    year_group: int = 9999
    squadron_id: int = 99
    sorties_flown: int = 0
    hours_flown: int = 0
    adsc_remaining: int = 120 # Measured in months
    active: bool = True
    separation_date: tuple = (9999, 0)
    current_assignment: Assignment = Assignment.LINE
    
    def set_rap_requirement(self):
        if self.qual == Qual.WG:
            self.target_sorties = 9
        elif self.qual == Qual.FL or Qual.IP:
            self.target_sorties = 8
    
    def update_total(self):
        self.total_phase = self.sortie_phase + self.sim_phase
        self.rap_shortfall = max(0, self.target_sorties - self.total_phase) 

    def update_monthly(self, phase_length_days: int):
        months = phase_length_days / 30
        if months > 0:
            self.sortie_monthly = self.sortie_phase / months
            self.sim_monthly = self.sim_phase / months
            self.sortie_blue_monthly = self.sortie_blue_phase / months
            self.sortie_red_monthly = self.sortie_red_phase / months

    def reset_phase_counters(self):
        self.sortie_phase = 0
        self.hours_phase = 0
        self.sortie_blue_phase = 0
        self.sortie_red_phase = 0
        self.sim_phase = 0

    def add_sortie(self, avg_sortie_dur: float, side: str = "Blue"):
        self.sortie_phase += 1
        if side == "Blue":
            self.sortie_blue_phase += 1
        elif side == "Red":
            self.sortie_red_phase += 1

        self.hours_phase += avg_sortie_dur

    def graduate(self):
        if self.upgrade == Upgrade.MQT:
            self.qual = Qual.WG
        elif self.upgrade == Upgrade.FLUG:
            self.qual = Qual.FL
        elif self.upgrade == Upgrade.IPUG:
            self.qual = Qual.IP
            
        self.upgrade = Upgrade.NONE

    def age_one_phase_with_rates(self, aging_rate: float, asd: float, phase_length_months: int):  
        if not self.active:
            return
            
        self.sorties_flown += aging_rate
        self.hours_flown += aging_rate * asd
        
        if self.adsc_remaining > 0:
            self.adsc_remaining -= phase_length_months
    
    def check_retention(self, current_year, current_phase, retention_pct: float):
        """
        If ADSC is 0 or less, roll to see if the pilot stays.
        retention_pct: float (e.g., 0.65 for 65% retention)
        """
        if self.active and self.adsc_remaining <= 0:
            # random.random() returns a float between 0.0 and 1.0
            if random.random() > retention_pct:
                self.active = False  # The pilot separates
                self.separation_date = (current_year, current_phase)

            else: 
                self.adsc_remaining += 24.1 # Assumes additional 2-year ADSC; .1 is a flag for logic elsewhere in the code

    def move_to_staff(self):
        self.current_assignment = Assignment.STAFF
        self.squadron_id = 0
    
# ----------------------
# Squadron Config 
# ----------------------
@dataclass
class SquadronConfig:
    ute: float
    paa: int
    id: int
    ccr: float = 1.5
    manning_pct: float = 1.5

    mqt_students: int = 0 
    flug_students: int = 0
    ipug_students: int = 0
    wg_qty: int = 0
    fl_qty: int = 0
    ip_qty: int = 0
    total_pilots: int = 0
    line_pilots: int = 0
    experience_ratio: float = 0.0
    phase_length_days: int = 120  
    avg_sortie_dur: float = 1.3

    pilots: List[Pilot] = field(default_factory=list)

    @property
    def desired_manning(self) -> int:
        return int(self.ccr * self.paa)

    @property
    def max_manning(self) -> int:
        return int(self.desired_manning * self.manning_pct)
    
    def get_feature_vector(self) -> list:
        """Returns the ordered list of features expected by the AI Brain."""

        return [
            self.paa,
            self.ute,
            self.experience_ratio,
            self.line_pilots,       
            self.mqt_students,
            self.flug_students,
            self.ipug_students,
            self.ip_qty
        ]
    
    def update_stats(self):
        # 1. Filter for Active Line Pilots (The only ones who count for stats)
        line_pilots = [p for p in self.pilots if p.active and p.current_assignment == Assignment.LINE]
        total_pilots = [p for p in self.pilots if p.active]
        
        # 2. Update Counts
        self.line_pilots = len(line_pilots)
        self.total_pilots = len(total_pilots)
        self.ip_qty = sum(1 for p in line_pilots if p.qual == Qual.IP)
        self.fl_qty = sum(1 for p in line_pilots if p.qual == Qual.FL)
        self.wg_qty = sum(1 for p in line_pilots if p.qual == Qual.WG)
        
        # 3. Update Experience Ratio
        if self.line_pilots > 0:
            self.experience_ratio = (self.ip_qty + self.fl_qty) / self.line_pilots
        else:
            self.experience_ratio = 0.0

        # 4. Update Student Counts
        self.mqt_students = sum(1 for p in line_pilots if p.upgrade == Upgrade.MQT)
        self.flug_students = sum(1 for p in line_pilots if p.upgrade == Upgrade.FLUG)
        self.ipug_students = sum(1 for p in line_pilots if p.upgrade == Upgrade.IPUG)

    def graduate_current_upgrades(self):
        dirty = False 

        for pilot in self.pilots:
            if pilot.upgrade != Upgrade.NONE:
                pilot.graduate()
                dirty = True
        
        if dirty:
            self.update_stats()
        if self.mqt_students > 0 or self.flug_students > 0 or self.ipug_students > 0:
            raise AssertionError(f'Graduation logic not functioning properly.')

    def new_phase_upgrades(self, flug_window_start:int, ipug_window_start:int,
                           use_upgrade_quotas: bool = False, flug_quota: int = 999,
                           ipug_quota: int = 999):
        flug_eligible = [
            p for p in self.pilots if p.qual == Qual.WG and p.upgrade == Upgrade.NONE 
            and flug_window_start <= p.sorties_flown 
        ]

        if use_upgrade_quotas:
            flug_eligible.sort(key=lambda x: x.sorties_flown, reverse=True)
            flug_limit = flug_quota
        else:
            flug_limit = len(flug_eligible)

        for i in range(min(len(flug_eligible), flug_limit)):
            p = flug_eligible[i]
            p.upgrade = Upgrade.FLUG

        ipug_eligible = [
            p for p in self.pilots if p.qual == Qual.FL and p.upgrade == Upgrade.NONE 
            and ipug_window_start <= p.hours_flown 
        ]
        
        if use_upgrade_quotas:
            ipug_eligible.sort(key=lambda x: x.hours_flown, reverse=True)
            ipug_limit = ipug_quota
        else:
            ipug_limit = len(ipug_eligible)

        for i in range(min(len(ipug_eligible), ipug_limit)):
            p = ipug_eligible[i]
            p.upgrade = Upgrade.IPUG

        self.update_stats()
        
    def apply_phase_aging(self, rates: AgingRate):
        "Ages pilots by adding phase aging rate in hours/sorties and subtracts phase length from ADSC remaining."
        phase_length_months = self.phase_length_days / 30

        for p in self.pilots:
            if p.qual == Qual.IP:
                p_rate = rates.ip_phase
            elif p.qual == Qual.FL:
                p_rate = rates.fl_phase
            elif p.upgrade == Upgrade.MQT:
                p_rate = rates.mqt_phase
            else:
                p_rate = rates.wg_phase

            p.age_one_phase_with_rates(p_rate, self.avg_sortie_dur, phase_length_months)

    def calc_aging_rate(self, sim_upgrades: bool):
        phase_months = self.phase_length_days / 30
        
        ute = self.ute
        paa = self.paa

        if not sim_upgrades:
            wg_rate = ((ute * paa) / 2) / self.wg_qty
            exp_rate = ((ute * paa) / 2) / ((self.fl_qty + self.ip_qty) / 2)

            return AgingRate(
                mqt_phase=4.0 * phase_months,
                wg_phase=wg_rate * phase_months,
                fl_phase=exp_rate * phase_months,
                ip_phase=exp_rate * phase_months,
                mqt_blue_phase=4.0 * phase_months,
                wg_blue_phase=None, # TODO figure out this proportion...
                fl_blue_phase=None,
                ip_blue_phase=None
            )
    
    # def predict_aging_rate(self, brain: dict) -> AgingRate:
    #     """
    #     Args:
    #         brain: Dictionary containing the trained sklearn models 
    #                (wg_monthly, fl_monthly, ip_monthly, etc.)
    #     """
    #     # 1. CALCULATE INPUTS (Must match training order EXACTLY)
    #     # Features: ['paa', 'ute', 'exp_ratio', 'total_pilots', 'mqt_qty', 'flug_qty', 'ipug_qty', 'ip_qty']
        
    #     # Ensure we are using Line Pilots (Cockpit Strength)
    #     line_pilots = len([p for p in self.pilots if p.current_assignment == Assignment.LINE])
        
    #     # Construct Input Vector (2D Array for sklearn)
    #     feature_names = ['paa', 'ute', 'exp_ratio', 'total_pilots', 'mqt_qty', 'flug_qty', 'ipug_qty', 'ip_qty']
        
    #     # 3. Construct Input Vector
    #     input_data = pd.DataFrame([[
    #         self.paa,
    #         self.ute,
    #         self.experience_ratio,
    #         self.line_pilots,       
    #         self.mqt_students,
    #         self.flug_students,
    #         self.ipug_students,
    #         self.ip_qty
    #     ]], columns=feature_names)

    #     # 2. GET PREDICTIONS (Monthly Rates)
    #     try:
    #         wg_mo = brain['wg_monthly'].predict(input_data)[0]
    #         fl_mo = brain['fl_monthly'].predict(input_data)[0]
    #         ip_mo = brain['ip_monthly'].predict(input_data)[0]
            
    #         # Blue Air Predictions
    #         wg_blue_mo = brain['wg_blue_monthly'].predict(input_data)[0]
    #         fl_blue_mo = brain['fl_blue_monthly'].predict(input_data)[0]
    #         ip_blue_mo = brain['ip_blue_monthly'].predict(input_data)[0]
    #     except KeyError as e:
    #         print(f"🚨 Brain Missing Model: {e}")
    #         return AgingRate() # Return empty/zero rate on failure

    #     # 3. CONVERT TO PHASE OUTPUT (Sorties per Phase)
    #     months_per_phase = self.phase_length_days / 30.0

    #     return AgingRate(
    #         mqt_phase=4.0 * months_per_phase, # Fixed allocation for MQT
    #         wg_phase=max(0, wg_mo * months_per_phase),
    #         fl_phase=max(0, fl_mo * months_per_phase),
    #         ip_phase=max(0, ip_mo * months_per_phase),
            
    #         # Blue Air Support Requirements
    #         mqt_blue_phase=4.0 * months_per_phase, # Fixed allocation for MQT
    #         wg_blue_phase=max(0, wg_blue_mo * months_per_phase),
    #         fl_blue_phase=max(0, fl_blue_mo * months_per_phase),
    #         ip_blue_phase=max(0, ip_blue_mo * months_per_phase)
    #     )
        
    def store_stats(self, year: int, phase_num: int, rates: AgingRate):
        months = self.phase_length_days / 30

        self.update_stats()

        current_stats = {
            'year': year,
            'phase': phase_num,
            'squadron_id': self.id,
            'wg_qty': self.wg_qty,
            'fl_qty': self.fl_qty,
            'ip_qty': self.ip_qty,
            'mqt_qty': self.mqt_students,
            'flug_qty': self.flug_students,
            'ipug_qty': self.ipug_students,
            'percent_manned': self.line_pilots / self.desired_manning,
            'line_pilots': self.line_pilots,
            'total_pilots': self.total_pilots,
            'exp_rat': self.experience_ratio,
            'staff_ips': 0,
            'staff_fls': 0,
            'separated': 0,
            'retained': 0,
            'wg_rate_mo': rates.wg_phase / months,
            'fl_rate_mo': rates.fl_phase / months,
            'ip_rate_mo': rates.ip_phase / months,
            'wg_rate_blue': rates.wg_blue_phase / months,
            'fl_rate_blue': rates.fl_blue_phase / months,
            'ip_rate_blue': rates.ip_blue_phase / months,
            'wg_rap_shortfall': 9 - (rates.wg_phase / months), 
            'fl_rap_shortfall': 8 - (rates.fl_phase / months),
            'ip_rap_shortfall': 8 - (rates.ip_phase / months),
            'wg_blue_shortfall': 9 - (rates.wg_blue_phase / months),
            'fl_blue_shortfall': 8 - (rates.fl_blue_phase / months),
            'ip_blue_shortfall': 8 - (rates.ip_blue_phase / months)
        }
    
        return current_stats
    
    def send_to_staff(self, priority_mode: PriorityMode = PriorityMode.RANDOM, min_ips: int = 3):
        current_line_pilots = []
        limit = self.max_manning

        for p in self.pilots:
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

            eligible_ips = ips[min_ips:] if len(ips) > min_ips else [] # Min of 3 protects Sq/CC, DO, and WO

            if priority_mode == PriorityMode.FL_FIRST:
                funnel_queue = fls + eligible_ips
            elif priority_mode == PriorityMode.IP_FIRST:
                funnel_queue = eligible_ips + fls
            elif priority_mode == PriorityMode.RANDOM:
                funnel_queue = eligible_ips + fls
                random.shuffle(funnel_queue)

            movers_count = min(excess_count, len(funnel_queue))
        
            for i in range(int(movers_count)): # Not sure why streamlit thinks this is a float
                funnel_queue[i].move_to_staff()

            self.update_stats()