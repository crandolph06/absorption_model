from dataclasses import dataclass, field
from enum import Enum
import random
from typing import List, Optional
from math import sqrt

# ----------------------
# Simple Upgrade Logic 
# ----------------------
FLUG_WINDOW_START = 250 # Sorties
FLUG_WINDOW_END = 265 # Sorties
IPUG_WINDOW_START = 400 # Hours
IPUG_WINDOW_END = 430 # Hours

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
    current_assignment: Assignment = Assignment.LINE
    
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

    def age_one_phase_with_rates(self, phase_sorties: float, phase_hours: float): # TODO Update as model develops
        """Updates pilot experience based on the calculated environment."""
        if not self.active:
            return
            
        self.sorties_flown += phase_sorties
        self.hours_flown += phase_hours
        
        if self.adsc_remaining > 0:
            self.adsc_remaining -= 4

        if self.upgrade == Upgrade.MQT:
            self.upgrade = Upgrade.NONE
    
    def check_retention(self, retention_pct: float):
        """
        If ADSC is 0 or less, roll to see if the pilot stays.
        retention_pct: float (e.g., 0.65 for 65% retention)
        """
        if self.active and self.adsc_remaining <= 0:
            # random.random() returns a float between 0.0 and 1.0
            if random.random() > retention_pct:
                self.active = False  # The pilot separates

            else: 
                self.adsc_remaining += 2 # Assumes additional 2-year ADSC
    

# ----------------------
# Squadron Config 
# ----------------------
@dataclass
class SquadronConfig:
    ute: float
    paa: int
    mqt_students: int
    flug_students: int
    ipug_students: int
    ip_qty: int
    phase_length_days: int = 120  # ~1/3 year or 4 months
    avg_sortie_dur: float = 1.3
    id: int = 99

    _total_pilots: Optional[int] = None
    _experience_ratio: Optional[float] = None

    pilots: List[Pilot] = field(default_factory=list)

    @property
    def total_pilots(self) -> int:
        if self._total_pilots is not None:
            return self._total_pilots
        return sum(1 for p in self.pilots if p.active)
    
    @total_pilots.setter
    def total_pilots(self, value: int):
        self._total_pilots = value

    @property
    def experience_ratio(self) -> float:
        if self._experience_ratio is not None:
            return self._experience_ratio
        
        tp = self.total_pilots
        if tp == 0: return 0.0
        exp_count = sum(1 for p in self.pilots if p.active and p.qual in [Qual.FL, Qual.IP])
        return exp_count/tp
    
    @experience_ratio.setter
    def experience_ratio(self, value:float):
        self._experience_ratio = value

    def graduate_current_upgrades(self):
        for pilot in self.pilots:
            if pilot.upgrade != Upgrade.NONE:
                pilot.graduate()

    def new_phase_upgrades(self, sim_upgrades: bool = False):
        mqt_count = sum(1 for p in self.pilots if p.upgrade == Upgrade.MQT)

        if not sim_upgrades:
            flug_eligible = [
                p for p in self.pilots if p.qual == Qual.WG and p.upgrade == Upgrade.NONE 
                and FLUG_WINDOW_START <= p.sorties_flown # <= FLUG_WINDOW_END
            ]
            for p in flug_eligible:
                p.upgrade = Upgrade.FLUG

            ipug_eligible = [
                p for p in self.pilots if p.qual == Qual.FL and p.upgrade == Upgrade.NONE 
                and IPUG_WINDOW_START <= p.hours_flown # <= IPUG_WINDOW_END
            ]
            for p in ipug_eligible:
                p.upgrade = Upgrade.IPUG

            return mqt_count, len(flug_eligible), len(ipug_eligible)
        
        else:
            # Sim more complex upgrade logic and IP Availability
            return
        
    def apply_phase_aging(self, rates: AgingRate):
        phase_months = self.phase_length_days / 30

        for p in self.pilots:
            if p.qual == Qual.IP:
                p_rate = rates.ip_phase
            elif p.qual == Qual.FL:
                p_rate = rates.fl_phase
            elif p.upgrade == Upgrade.MQT:
                p_rate = rates.mqt_phase
            else:
                p_rate = rates.wg_phase

            p.sorties_flown += p_rate
            p.hours_flown += p_rate * self.avg_sortie_dur

            p.adsc_remaining -= phase_months
        
    def calculate_aging_rates(self): 
        mqt, flug, ipug = self.new_phase_upgrades()
        sum_students = max(mqt + flug + ipug, 0.01)

        total_pilots = self.total_pilots
        exp_ratio = self.experience_ratio
        ip_ratio = self.ip_qty/total_pilots
        upg_pct = (sum_students)/total_pilots
        ute_per_pilot = self.ute/total_pilots
        ac_per_pilot = self.paa/total_pilots
        ip_to_stud_ratio = self.ip_qty/sum_students
        ip_load = sum_students/self.ip_qty

        ip_rate = max(0, sqrt(inv(exp_ratio) + (square(ac_per_pilot + (ute_per_pilot * (0.8983698 + sqrt(ip_to_stud_ratio)))) + (ip_load / 0.80686635))))
        fl_rate_inner_num = (((sqrt(total_pilots) + 3.5092452) + (0.16958028 / (ip_ratio - exp_ratio + 1e-6))) * (upg_pct + ute_per_pilot))
        fl_rate_inner_denom = ((exp_ratio / max(ac_per_pilot, 0.01)) + upg_pct)
        fl_rate = sqrt(max(0, fl_rate_inner_num / max(fl_rate_inner_denom, 1e-6)))
        mqt_rate = 4.0

        num_fls = int(exp_ratio * total_pilots)
        num_wg = total_pilots - num_fls - mqt

        phase_months = self.phase_length_days / 30

        total_capacity = self.ute * self.paa
        ip_sorties = self.ip_qty * ip_rate * phase_months
        fl_sorties = num_fls * fl_rate 
        mqt_sorties = mqt * mqt_rate
        remaining_for_wg = total_capacity - ip_sorties - fl_sorties - mqt_sorties
        wg_rate = remaining_for_wg / num_wg if num_wg > 0 else 0

        return AgingRate(
            mqt_phase = mqt_rate * phase_months,
            wg_phase=wg_rate * phase_months,
            fl_phase=fl_rate * phase_months,
            ip_phase=ip_rate * phase_months,
        )

