from dataclasses import dataclass, field
from enum import Enum
import random

# ----------------------
# Config 
# ----------------------
@dataclass
class SquadronConfig:
    ute: float
    paa: int
    mqt_students: int
    flug_students: int
    ipug_students: int
    total_pilots: int
    experience_ratio: float
    ip_qty: int
    phase_length_days: int = 120  # ~1/3 year or 4 months
    avg_sortie_dur: float = 1.3

# ----------------------
# Enums 
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

    wg_rap: float = 0
    mqt_rap: float = 0
    fl_rap: float = 0
    flug_rap: float = 0
    ip_rap: float = 0
    ipug_rap: float = 0

    target_sorties: float = 0
    rap_shortfall: float = 0

    year_group: int = 9999
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

    def reset_phase_counters(pilots):
        for p in pilots:
            p.sortie_phase = 0
            p.hours_phase = 0
            p.sortie_blue_phase = 0
            p.sortie_red_phase = 0
            p.sim_phase = 0

    def add_sortie(self, avg_sortie_dur: float, side: str = "Blue"):
        self.sortie_phase += 1
        if side == "Blue":
            self.sortie_blue_phase += 1
        elif side == "Red":
            self.sortie_red_phase += 1

        self.hours_phase += avg_sortie_dur

    def age_one_phase_with_rates(self, phase_sorties: float, phase_hours: float): # TODO Update as model develops
        """Updates pilot experience based on the calculated environment."""
        if not self.active:
            return
            
        self.sorties_flown += phase_sorties
        self.hours_flown += phase_hours
        
        # Decrement commitment
        if self.adsc_remaining > 0:
            self.adsc_remaining -= 4
            
        # Check for upgrades
        self.best_case_flight_lead_upgrade()
        self.best_case_instructor_upgrade()

    def best_case_flight_lead_upgrade(self):
        # Updated to handle '>= 250' so they don't miss the window
        if self.qual == Qual.WG and self.sorties_flown >= 250:
            self.qual = Qual.FL

    def best_case_instructor_upgrade(self):
        if self.qual == Qual.FL and self.hours_flown >= 400:
            self.qual = Qual.IP
    
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

@dataclass
class YearGroup:
    year: int
    pilots: list[Pilot] = field(default_factory=list)

    @property
    def num_active_pilots(self):
        return len([p for p in self.pilots if p.active])
    
@dataclass 
class AgingRate:
    wg_phase: float = 0.0
    fl_phase: float = 0.0
    ip_phase: float = 0.0

    wg_blue_phase: float = 0.0
    fl_blue_phase: float = 0.0
    ip_blue_phase: float = 0.0

