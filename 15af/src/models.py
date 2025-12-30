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
    sortie_monthly: float = 0
    sim_monthly: float = 0
    total_monthly: float = 0
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
        self.total_monthly = self.sortie_monthly + self.sim_monthly
        self.rap_shortfall = max(0, self.target_sorties - self.total_monthly)

    def reset_monthly_counters(pilots):
        for p in pilots:
            p.sortie_monthly = 0
            p.sortie_blue_monthly = 0
            p.sortie_red_monthly = 0
            p.sim_monthly = 0

    def add_sortie(self, side: str = "Blue"):
        self.sortie_monthly += 1
        if side == "Blue":
            self.sortie_blue_monthly += 1
        elif side == "Red":
            self.sortie_red_monthly += 1

