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

    def age_one_phase_with_rates(self, aging_rate: float, asd: float): 
        """Updates pilot experience based on the calculated environment."""
        if not self.active:
            return
            
        self.sorties_flown += aging_rate
        self.hours_flown += aging_rate * asd
        
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
        self.ip_qty = sum(1 for p in self.pilots if p.active and p.qual == Qual.IP and p.current_assignment == Assignment.LINE)
        self.total_pilots = sum(1 for p in self.pilots if p.active and p.current_assignment == Assignment.LINE)
        fl_count = sum(1 for p in self.pilots if p.active and p.current_assignment == Assignment.LINE and p.qual == Qual.FL)

        self.experience_ratio = (self.ip_qty + fl_count) / self.total_pilots


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

        for p in self.pilots:
            if p.qual == Qual.IP:
                p_rate = rates.ip_phase
            elif p.qual == Qual.FL:
                p_rate = rates.fl_phase
            elif p.upgrade == Upgrade.MQT:
                p_rate = rates.mqt_phase
            else:
                p_rate = rates.wg_phase

            p.age_one_phase_with_rates(p_rate, self.avg_sortie_dur)

    def calc_aging_rate(self, sim_upgrades: bool):
        phase_months = self.phase_length_days / 30
        
        ute = self.ute
        paa = self.paa
        wg_count = sum(1 for p in self.pilots if p.active and p.current_assignment == Assignment.LINE and p.qual == Qual.WG)
        fl_count = sum(1 for p in self.pilots if p.active and p.current_assignment == Assignment.LINE and p.qual == Qual.FL)
        ip_count = sum(1 for p in self.pilots if p.active and p.current_assignment == Assignment.LINE and p.qual == Qual.IP)
        exp_pilots = fl_count + ip_count # TODO Where do we re-hack experience ratio? Must just include LINE pilots

        if not sim_upgrades:
            wg_rate = (ute * paa) / wg_count
            fl_rate = ((ute * paa)/ exp_pilots) / 2
            ip_rate = fl_rate

            return AgingRate(
                mqt_phase=4.0 * phase_months,
                wg_phase=wg_rate * phase_months,
                fl_phase=fl_rate * phase_months,
                ip_phase=ip_rate * phase_months,
                mqt_blue_phase=4.0 * phase_months,
                wg_blue_phase=None,
                fl_blue_phase=None,
                ip_blue_phase=None
            )



    def lookup_aging_rate(self, params: dict, original_df: pd.DataFrame, 
                                  base_matrix, base_std, base_cols, 
                                  stud_matrix, stud_std, stud_cols, 
                                  sim_upgrades: bool) -> 'AgingRate':
        """
        Sequential Drill-Down Lookup:
        1. PAA/UTE (Exact)
        2. IP Qty (Closest)
        3. Exp Ratio (Closest)
        4. Total Pilots (Closest)
        5. Student Counts (Distance Tie-Breaker)
        """
        # --- 0. Handle Non-Upgrade Logic ---
        # If upgrades are OFF, we force the target student counts to 0.
        # This makes the lookup find "healthy" rows (low student load).
        current_stud_params = {}
        if sim_upgrades:
            current_stud_params = {c: params.get(c, 0) for c in stud_cols}
        else:
            current_stud_params = {c: 0 for c in stud_cols}

        # --- 1. Environmental Lock (Exact Match) ---
        target_paa = params.get('paa')
        target_ute = params.get('ute')
        
        mask = (original_df['paa'].values == target_paa) & (original_df['ute'].values == target_ute)
        
        # Safety fallback
        if not np.any(mask):
            mask = np.ones(len(original_df), dtype=bool)

        # --- 2. Sequential Soft Locks (The "Drill Down") ---
        # We define the priority order of variables to lock down
        priority_vars = ['exp_ratio', 'ip_qty', 'total_pilots']
        
        for var in priority_vars:
            target_val = params.get(var, 0)
            # Get values only from the currently surviving rows
            valid_values = original_df[var].values[mask]
            
            if len(valid_values) > 0:
                # Find the smallest difference available in the current subset
                min_diff = np.min(np.abs(valid_values - target_val))
                
                # Update mask: Keep only rows that are this close to the target
                # We use a small epsilon for float comparison safety on exp_ratio
                is_closest = np.abs(original_df[var].values - target_val) <= (min_diff + 0.00001)
                
                # Intersect with current mask
                mask = mask & is_closest

        # --- 3. Final Distance Calculation (Student Bottleneck) ---
        
        # Start with zero distance (all survivors are considered "equal" on base stats now)
        dists = np.zeros(len(original_df))
        
        # Add Student Distance (using the 0-forced values if sim_upgrades=False)
        if stud_matrix is not None:
            input_stud = np.array([current_stud_params.get(c, 0) for c in stud_cols])
            norm_input_stud = input_stud / stud_std
            
            # Calculate distance only on student columns
            dists += np.sum((stud_matrix - norm_input_stud)**2, axis=1)

        # --- 4. Apply Mask & Pick Winner ---
        # Set non-survivors to infinity
        dists[~mask] = np.inf
        
        closest_idx = np.argmin(dists)
        closest_row = original_df.iloc[closest_idx]

        if params.get('exp_ratio', 1.0) < 0.30:
            print("\n🚨 LOW RATIO LOOKUP TRIGGERED")
            print(f"INPUTS:  Ratio={params.get('exp_ratio'):.2f} | Pilots={params.get('total_pilots')} | IPs={params.get('ip_qty')}")
            print(f"WINNER:  Index={closest_idx} | Ratio={closest_row['exp_ratio']:.2f} | Pilots={closest_row['total_pilots']}")
            print(f"TOTAL RATES:   WG={closest_row.get('wg_monthly'):.2f} | IP={closest_row.get('ip_monthly'):.2f}")
            print(f"BLUE RATES:   WG={closest_row.get('wg_blue_monthly'):.2f} | IP={closest_row.get('ip_blue_monthly'):.2f}")
            print(f"CONTEXT: PAA={closest_row['paa']} | UTE={closest_row['ute']}")
            print("-" * 50)

        phase_months = self.phase_length_days / 30

        return AgingRate(
            mqt_phase=4.0 * phase_months,
            wg_phase=(closest_row.get('wg_monthly', 0) * phase_months),
            fl_phase=(closest_row.get('fl_monthly', 0) * phase_months),
            ip_phase=(closest_row.get('ip_monthly', 0) * phase_months),
            mqt_blue_phase=4.0,
            wg_blue_phase=(closest_row.get('wg_blue_monthly', 0) * phase_months),
            fl_blue_phase=(closest_row.get('fl_blue_monthly', 0) * phase_months),
            ip_blue_phase=(closest_row.get('ip_blue_monthly', 0) * phase_months)
        )