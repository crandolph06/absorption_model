from dataclasses import dataclass, field
from enum import Enum
import random
from typing import List, Optional, Tuple

# ----------------------
# Math
# ----------------------
# Used with `phase_length_days` everywhere we scale monthly rates or RAP to a phase
# (not calendar months; matches historical 120-day ≈ 4-month convention).
PHASE_DAYS_PER_NOTIONAL_MONTH: float = 30.0
# Monthly sim RAP (WG / FL / IP): 1 EP + 2 other sims = 3 / month in ``SIM_RAP_MONTHLY``.
SIM_RAP_MONTHLY: float = 3.0
SIM_EP_MONTHLY: float = 1.0
# Simulator wing capacity: ~30 session lines / month; each session has this many bays
# (e.g. 4 bays = one 4-ship block or two 2-ship blocks in parallel).
SIM_SESSIONS_MONTHLY: float = float("inf")
SIM_BAYS_PER_SESSION: int = 4
# Max combined sorties + sims per pilot per month (single-phase allocation path).
MAX_MONTHLY_EVENTS: float = 25.0


# ----------------------
# Enums & Simple Classes
# ----------------------

class Qual(Enum):
    WG = 'WG'
    FL = 'FL'
    IP = 'IP'


# Monthly sortie RAP (same as ``Pilot.set_rap_requirement`` / ``rap_state`` / manning stats).
SORTIE_RAP_MONTHLY_WG: float = 9.0
SORTIE_RAP_MONTHLY_FL_IP: float = 8.0


def monthly_sortie_rap_target(qual: Qual) -> float:
    """Notional monthly sortie RAP requirement for ``qual``; 0 if none."""
    if qual == Qual.WG:
        return SORTIE_RAP_MONTHLY_WG
    if qual in (Qual.FL, Qual.IP):
        return SORTIE_RAP_MONTHLY_FL_IP
    return 0.0


def monthly_sim_rap_target(qual: Qual) -> float:
    """Notional monthly sim RAP for WG/FL/IP; 0 otherwise (``Pilot.set_rap_requirement``)."""
    if qual in (Qual.WG, Qual.FL, Qual.IP):
        return float(SIM_RAP_MONTHLY)
    return 0.0


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

    # Phase-level sim event totals (manning / future brain). When zero, only sortie fields are used.
    mqt_sim_phase: float = 0.0
    wg_sim_phase: float = 0.0
    fl_sim_phase: float = 0.0
    ip_sim_phase: float = 0.0
    mqt_sim_blue_phase: float = 0.0
    wg_sim_blue_phase: float = 0.0
    fl_sim_blue_phase: float = 0.0
    ip_sim_blue_phase: float = 0.0

    def monthly_to_phase(self, phase_length_days: float):
        phase_length_months = float(phase_length_days) / PHASE_DAYS_PER_NOTIONAL_MONTH

        return AgingRate(
            mqt_phase=self.mqt_phase * phase_length_months,
            wg_phase=self.wg_phase * phase_length_months,
            fl_phase=self.fl_phase * phase_length_months,
            ip_phase=self.ip_phase * phase_length_months,

            mqt_blue_phase=self.mqt_blue_phase * phase_length_months,
            wg_blue_phase=self.wg_blue_phase * phase_length_months,
            fl_blue_phase=self.fl_blue_phase * phase_length_months,
            ip_blue_phase=self.ip_blue_phase * phase_length_months,

            mqt_sim_phase=self.mqt_sim_phase * phase_length_months,
            wg_sim_phase=self.wg_sim_phase * phase_length_months,
            fl_sim_phase=self.fl_sim_phase * phase_length_months,
            ip_sim_phase=self.ip_sim_phase * phase_length_months,
            mqt_sim_blue_phase=self.mqt_sim_blue_phase * phase_length_months,
            wg_sim_blue_phase=self.wg_sim_blue_phase * phase_length_months,
            fl_sim_blue_phase=self.fl_sim_blue_phase * phase_length_months,
            ip_sim_blue_phase=self.ip_sim_blue_phase * phase_length_months,
        )
    
# ----------------------
# Pilot Entity
# ----------------------
@dataclass
class Pilot:
    qual: Qual = Qual.WG 
    upgrade: Upgrade = Upgrade.NONE
    incomplete_syllabus_items: List = field(default_factory=list)
    sortie_phase: float = 0 
    flight_hours_phase: float = 0.0
    sim_hours_phase: float = 0.0
    sim_phase: float = 0 
    ep_sim_phase: float = 0.0
    total_phase: float = 0 
    sortie_blue_phase: float = 0 
    sortie_red_phase: float = 0 

    target_sorties: float = 0
    target_sims: float = 0.0
    rap_shortfall: float = 0
    sim_rap_shortfall: float = 0.0

    sortie_monthly: float = 0
    sim_monthly: float = 0
    flight_hours_monthly: float = 0.0
    sim_hours_monthly: float = 0.0
    sortie_blue_monthly: float = 0
    sortie_red_monthly: float = 0

    year_group: int = 9999
    squadron_id: int = 99
    sorties_flown: int = 0
    sorties_at_phase_start: int = 0
    sorties_at_upgrade_start: int = 0
    sims_flown: float = 0.0
    sims_at_upgrade_start: int = 0
    flight_hours_flown: float = 0.0
    sim_hours_flown: float = 0.0
    adsc_remaining: int = 120 # Measured in months
    active: bool = True
    separation_date: tuple = (9999, 0)
    current_assignment: Assignment = Assignment.LINE

    def set_rap_requirement(self):
        """Monthly sortie and sim RAP targets (single source: ``monthly_*_rap_target``)."""
        if self.upgrade == Upgrade.MQT:
            self.target_sorties = 0.0
            self.target_sims = 0.0
            return
        self.target_sorties = monthly_sortie_rap_target(self.qual)
        self.target_sims = monthly_sim_rap_target(self.qual)

    def update_total(self, phase_length_days: Optional[float] = None):
        self.total_phase = self.sortie_phase + self.sim_phase
        if phase_length_days is not None and phase_length_days > 0:
            months = float(phase_length_days) / PHASE_DAYS_PER_NOTIONAL_MONTH
            exp_sorties = self.target_sorties * months
            self.rap_shortfall = max(0.0, exp_sorties - self.sortie_phase)
            exp_sims = self.target_sims * months
            self.sim_rap_shortfall = max(0.0, exp_sims - self.sim_phase)
        else:
            self.rap_shortfall = max(0.0, self.target_sorties - self.sortie_phase)
            self.sim_rap_shortfall = max(0.0, self.target_sims - self.sim_phase)

    def update_monthly(self, phase_length_days: float):
        months = float(phase_length_days) / PHASE_DAYS_PER_NOTIONAL_MONTH
        if months > 0:
            self.sortie_monthly = self.sortie_phase / months
            self.sim_monthly = self.sim_phase / months
            self.flight_hours_monthly = self.flight_hours_phase / months
            self.sim_hours_monthly = self.sim_hours_phase / months
            self.sortie_blue_monthly = self.sortie_blue_phase / months
            self.sortie_red_monthly = self.sortie_red_phase / months

    def reset_phase_counters(self):
        self.sortie_phase = 0
        self.flight_hours_phase = 0.0
        self.sim_hours_phase = 0.0
        self.sortie_blue_phase = 0
        self.sortie_red_phase = 0
        self.sim_phase = 0
        self.ep_sim_phase = 0.0

    def phase_events(self) -> float:
        return float(self.sortie_phase) + float(self.sim_phase)

    def has_events_capacity(self, phase_length_days: float, additional: float = 1.0) -> bool:
        """True if ``additional`` more events would stay within ``MAX_MONTHLY_EVENTS`` per month."""
        months = float(phase_length_days) / PHASE_DAYS_PER_NOTIONAL_MONTH
        if months <= 0:
            return False
        return (self.phase_events() + additional) / months <= MAX_MONTHLY_EVENTS + 1e-9

    def add_sortie(self, avg_sortie_dur: float, side: str = "Blue"):
        self.sortie_phase += 1
        if side == "Blue":
            self.sortie_blue_phase += 1
        elif side == "Red":
            self.sortie_red_phase += 1

        self.flight_hours_phase += avg_sortie_dur
        self.flight_hours_flown += avg_sortie_dur

    def add_sim(self, avg_event_dur: float, *, count_ep: bool = False) -> None:
        """One simulator event for this pilot (syllabus or sim RAP). ``count_ep`` tags EP RAP."""
        self.sim_phase += 1.0
        self.sims_flown += 1
        self.sim_hours_phase += avg_event_dur
        self.sim_hours_flown += avg_event_dur
        if count_ep:
            self.ep_sim_phase += 1.0

    def graduate(self):
        if self.upgrade == Upgrade.MQT:
            self.qual = Qual.WG
        elif self.upgrade == Upgrade.FLUG:
            self.qual = Qual.FL
        elif self.upgrade == Upgrade.IPUG:
            self.qual = Qual.IP
            
        self.upgrade = Upgrade.NONE
        self.incomplete_syllabus_items.clear()

    def age_one_phase_with_rates(self, aging_rate: float, asd: float, phase_length_months: int):  
        if not self.active:
            return
            
        self.sorties_flown += aging_rate
        self.flight_hours_flown += aging_rate * asd
        
        if self.adsc_remaining > 0:
            self.adsc_remaining -= phase_length_months

    def age_sim_phase_with_rates(self, sim_rate: float, avg_event_dur: float) -> None:
        """Fractional sim phase credit (manning when brain supplies ``*_sim_phase``), mirroring sortie aging."""
        if not self.active:
            return
        self.sims_flown += float(sim_rate)
        self.sim_hours_flown += float(sim_rate) * float(avg_event_dur)
        self.sim_phase += float(sim_rate)

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
    mqt_carry: float = 0.0
    flug_carry: float = 0.0
    ipug_carry: float = 0.0
    wg_qty: int = 0
    fl_qty: int = 0
    ip_qty: int = 0
    total_pilots: int = 0
    line_pilots: int = 0
    experience_ratio: float = 0.0
    phase_length_days: int = 120  # Default ~4 months; drive all phase scaling from this field.
    avg_sortie_dur: float = 1.3
    # Simulator wing: session lines per month (each line has ``sim_bays_per_session`` bays).
    sim_sessions_monthly: float = SIM_SESSIONS_MONTHLY
    sim_bays_per_session: int = SIM_BAYS_PER_SESSION

    pilots: List[Pilot] = field(default_factory=list)
    observed_mqt_monthly: Optional[float] = None

    @property
    def phase_length_months(self) -> float:
        """Notional months in a phase (phase_length_days / 30)."""
        return max(0.0, float(self.phase_length_days) / PHASE_DAYS_PER_NOTIONAL_MONTH)

    @property
    def desired_manning(self) -> int:
        return int(self.ccr * self.paa)

    @property
    def max_manning(self) -> int:
        return int(self.desired_manning * self.manning_pct)
    
    def mean_monthly_sim_events(self, qual: Qual) -> float:
        total_sims = sum(p.sim_phase for p in self.pilots if p.active and p.current_assignment == Assignment.LINE and p.qual == qual)
        qualified_pilots = [p for p in self.pilots if p.active and p.current_assignment == Assignment.LINE and p.qual == qual]
        if total_sims <= 0:
            return 0.0
        if len(qualified_pilots) <= 0:
            return 0.0
        if self.phase_length_months <= 0:
            return 0.0
        return(total_sims / len(qualified_pilots)) / self.phase_length_months


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

    def graduate_current_upgrades(self, deferrals: Tuple[int, int, int, int, int, int], sorties_only: bool = True):
        """Graduate upgrade students with no deferred syllabus lines. """
        
        graduated_count = 0

        if all(abs(d) < 1e-3 for d in deferrals):
            for pilot in self.pilots:
                if pilot.upgrade != Upgrade.NONE:
                    pilot.graduate()
                    graduated_count += 1
            self.mqt_carry = 0.0
            self.flug_carry = 0.0
            self.ipug_carry = 0.0

        else:
            if sorties_only:
                deferrals = deferrals[3:]
            else:
                deferrals = deferrals[:3]
            mqt_deferrals, flug_deferrals, ipug_deferrals = deferrals

            mqt_whole = int(mqt_deferrals)
            flug_whole = int(flug_deferrals)
            ipug_whole = int(ipug_deferrals)

            self.mqt_carry = mqt_deferrals - mqt_whole
            self.flug_carry = flug_deferrals - flug_whole
            self.ipug_carry = ipug_deferrals - ipug_whole

            mqt_pilots = [p for p in self.pilots if p.upgrade == Upgrade.MQT]
            flug_pilots = [p for p in self.pilots if p.upgrade == Upgrade.FLUG]
            ipug_pilots = [p for p in self.pilots if p.upgrade == Upgrade.IPUG]

            mqt_pilots.sort(
                key=lambda x: (x.sorties_flown - x.sorties_at_upgrade_start, 
                x.sorties_flown), reverse=True)
            flug_pilots.sort(
                key=lambda x: (x.sorties_flown - x.sorties_at_upgrade_start, 
                x.sorties_flown), reverse=True)
            ipug_pilots.sort( 
                key=lambda x: (x.sorties_flown - x.sorties_at_upgrade_start, 
                x.sorties_flown), reverse=True)

            mqt_limit = max(0, len(mqt_pilots) - mqt_whole)
            flug_limit = max(0, len(flug_pilots) - flug_whole)
            ipug_limit = max(0, len(ipug_pilots) - ipug_whole)

            for i in range(mqt_limit):
                mqt_pilots[i].graduate()
                graduated_count += 1
            for i in range(flug_limit):
                flug_pilots[i].graduate()
                graduated_count += 1
            for i in range(ipug_limit):
                ipug_pilots[i].graduate()
                graduated_count += 1
                
        self.update_stats()
        still_upgrade = self.mqt_students + self.flug_students + self.ipug_students
        # if still_upgrade:
        #     print(
        #         f"Squadron {self.id} graduation: {graduated_count} pilot(s) graduated; "
        #         f"{still_upgrade} still in upgrade (deferred syllabus lines remain) "
        #         f"[MQT={self.mqt_students}, FLUG={self.flug_students}, IPUG={self.ipug_students}]."
        #     )

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
            p.sorties_at_upgrade_start = p.sorties_flown
            p.sims_at_upgrade_start = p.sims_flown

        ipug_eligible = [
            p for p in self.pilots if p.qual == Qual.FL and p.upgrade == Upgrade.NONE 
            and ipug_window_start <= p.flight_hours_flown 
        ]

        # print(f"Squadron {self.id} | FLUG Eligible: {len(flug_eligible)} | IPUG Eligible: {len(ipug_eligible)}")
        
        if use_upgrade_quotas:
            ipug_eligible.sort(key=lambda x: x.flight_hours_flown, reverse=True)
            ipug_limit = ipug_quota
        else:
            ipug_limit = len(ipug_eligible)

        for i in range(min(len(ipug_eligible), ipug_limit)):
            p = ipug_eligible[i]
            p.upgrade = Upgrade.IPUG
            p.sorties_at_upgrade_start = p.sorties_flown
            p.sims_at_upgrade_start = p.sims_flown

        self.update_stats()

    def _phase_sim_rate_for_pilot(self, p: Pilot, rates: AgingRate) -> float:
        """Phase sim rate bucket mirroring ``apply_phase_aging`` sortie branch."""
        if p.qual == Qual.IP:
            return rates.ip_sim_phase
        if p.qual == Qual.FL:
            return rates.fl_sim_phase
        if p.upgrade == Upgrade.MQT:
            return rates.mqt_sim_phase
        return rates.wg_sim_phase

    def apply_phase_aging(self, rates: AgingRate):
        """Sortie and sim phase credit from ``rates`` (brain-predicted monthly rates)."""
        for p in self.pilots:
            if not p.active:
                continue

            if p.qual == Qual.IP:
                p_rate = rates.ip_phase
            elif p.qual == Qual.FL:
                p_rate = rates.fl_phase
            elif p.upgrade == Upgrade.MQT:
                p_rate = rates.mqt_phase
            else:
                p_rate = rates.wg_phase

            p.age_one_phase_with_rates(p_rate, self.avg_sortie_dur, self.phase_length_months)

            sim_rate = self._phase_sim_rate_for_pilot(p, rates)
            p.age_sim_phase_with_rates(sim_rate, self.avg_sortie_dur)

    def store_stats(self, year: int, phase_num: int, rates: AgingRate):
        months = self.phase_length_months
        if months <= 0:
            months = 1e-9

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
            'mqt_carry': self.mqt_carry,
            'flug_carry': self.flug_carry,
            'ipug_carry': self.ipug_carry,
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
            'wg_rate_sim': self.mean_monthly_sim_events(Qual.WG),
            'fl_rate_sim': self.mean_monthly_sim_events(Qual.FL),
            'ip_rate_sim': self.mean_monthly_sim_events(Qual.IP),
            'wg_rap_shortfall': monthly_sortie_rap_target(Qual.WG) - (rates.wg_phase / months),
            'fl_rap_shortfall': monthly_sortie_rap_target(Qual.FL) - (rates.fl_phase / months),
            'ip_rap_shortfall': monthly_sortie_rap_target(Qual.IP) - (rates.ip_phase / months),
            'wg_blue_shortfall': monthly_sortie_rap_target(Qual.WG) - (rates.wg_blue_phase / months),
            'fl_blue_shortfall': monthly_sortie_rap_target(Qual.FL) - (rates.fl_blue_phase / months),
            'ip_blue_shortfall': monthly_sortie_rap_target(Qual.IP) - (rates.ip_blue_phase / months),
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