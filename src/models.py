from dataclasses import dataclass
from enum import Enum

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

# ----------------------
# Pilot Entity
# ----------------------
@dataclass
class Pilot:
    qual: Qual
    upgrade: Upgrade = Upgrade.NONE
    sortie_phase: float = 0 
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
            p.sortie_blue_phase = 0
            p.sortie_red_phase = 0
            p.sim_phase = 0

    def add_sortie(self, side: str = "Blue"):
        self.sortie_phase += 1
        if side == "Blue":
            self.sortie_blue_phase += 1
        elif side == "Red":
            self.sortie_red_phase += 1

