from dataclasses import dataclass, field
from enum import Enum
import random
from typing import List, Optional
from math import sqrt
import pandas as pd
import numpy as np

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
    separation_date: tuple = (9999, 0)
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
                self.adsc_remaining += 24.1 # Assumes additional 2-year ADSC

    def move_to_staff(self):
        self.current_assignment = Assignment.STAFF
        self.squadron_id = None
    

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

    @property
    def manning_limit(self) -> int:
        return 1.5 * self.paa

    def graduate_current_upgrades(self):
        for pilot in self.pilots:
            if pilot.upgrade != Upgrade.NONE:
                pilot.graduate()

        self.mqt_students = 0
        self.flug_students = 0
        self.ipug_students = 0
        self.ip_qty = sum(1 for p in self.pilots if p.active and p.qual == Qual.IP)
        self.total_pilots = sum(1 for p in self.pilots if p.active and p.current_assignment == Assignment.LINE)

    def new_phase_upgrades(self, flug_window_start:int, ipug_window_start:int):
        mqt_count = sum(1 for p in self.pilots if p.upgrade == Upgrade.MQT)

        flug_eligible = [
            p for p in self.pilots if p.qual == Qual.WG and p.upgrade == Upgrade.NONE 
            and flug_window_start <= p.sorties_flown 
        ]
        for p in flug_eligible:
            p.upgrade = Upgrade.FLUG

        ipug_eligible = [
            p for p in self.pilots if p.qual == Qual.FL and p.upgrade == Upgrade.NONE 
            and ipug_window_start <= p.hours_flown 
        ]
        for p in ipug_eligible:
            p.upgrade = Upgrade.IPUG

        return mqt_count, len(flug_eligible), len(ipug_eligible)
        
    def apply_phase_aging(self, rates: AgingRate):
        "Ages pilots by adding phase aging rate in hours/sorties and subtracts phase length from ADSC remaining."
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


    def lookup_aging_rate(self, params: dict, original_df: pd.DataFrame, 
                                  base_matrix, base_std, base_cols, 
                                  stud_matrix, stud_std, stud_cols, 
                                  sim_upgrades: bool) -> 'AgingRate':
        """
        Finds the row in lookup_df most similar to current_params and returns an AgingRate.
        """
        input_base = np.array([params.get(c, 0) for c in base_cols])
        norm_input_base = input_base / base_std
        dists = np.sum((base_matrix - norm_input_base)**2, axis=1)

        if sim_upgrades and stud_matrix is not None:
            input_stud = np.array([params.get(c, 0) for c in stud_cols])
            norm_input_stud = input_stud / stud_std
            dists += np.sum((stud_matrix - norm_input_stud)**2, axis=1)

        closest_idx = np.argmin(dists)
        closest_row = original_df.iloc[closest_idx]

        phase_months = self.phase_length_days / 30

        return AgingRate(
            mqt_phase=4.0 * phase_months, # Assumes 16 sortie upgrade over 4 months.
            wg_phase=(closest_row['wg_monthly'] * phase_months),
            fl_phase=(closest_row['fl_monthly'] * phase_months),
            ip_phase=(closest_row['ip_monthly'] * phase_months),
            mqt_blue_phase=4.0,
            wg_blue_phase=(closest_row['wg_blue_monthly'] * phase_months),
            fl_blue_phase=(closest_row['fl_blue_monthly'] * phase_months),
            ip_blue_phase=(closest_row['ip_blue_monthly'] * phase_months)
        )
        