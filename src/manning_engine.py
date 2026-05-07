import pandas as pd
from typing import List, Optional
from src.models import Pilot, Qual, SquadronConfig, Upgrade, Assignment, PriorityMode, AgingRate
import os
import joblib


class CAFSimulation:
    def __init__(self, sim_upgrades: bool, round_robin: bool, brain = None, flug_window_start: int = 250, ipug_window_start: int = 400, max_manning_pct: int = 150, staff_priority_mode: PriorityMode = PriorityMode.RANDOM):
        self.history = []
        self.current_year = 2026
        self.squadrons: List[SquadronConfig] = []
        self.flug_window_start = flug_window_start # Sorties for FLUG auto-start
        self.ipug_window_start = ipug_window_start # Hours for IPUG auto-start
        self.max_manning = max_manning_pct/100
        self.staff_priority = staff_priority_mode

        if brain:
            self.brain = brain
        elif os.path.exists("sortie_brain.pkl"):
            print(f"🧠 Loading Sortie Brain from disk...")
            self.brain = joblib.load("sortie_brain.pkl")
        else:
            raise FileNotFoundError(f"Could not find sortie_brain.pkl. Please run train_brain.py first.")

        self.sim_upgrades = sim_upgrades
        self.round_robin = round_robin

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
    
    @property
    def total_wg_qty(self):
        return sum(1 for sq in self.squadrons for p in sq.pilots if p.qual.name == 'WG')
    
    @property
    def total_fl_qty(self):
        return sum(1 for sq in self.squadrons for p in sq.pilots if p.qual.name == 'FL')
    
    @property
    def total_ip_qty(self):
        return sum(1 for sq in self.squadrons for p in sq.pilots if p.qual.name == 'IP')
    
    @property
    def experience_ratio(self):
        total_experienced = self.total_ip_qty + self.total_fl_qty
        return total_experienced / max(self.total_line_pilot_count, 1)

    @property
    def current_wg_shortfall(self) -> float:
        num_sq = len(self.squadrons)
        if len(self.history) < num_sq: 
            return 0.0
        return sum(max(0, s.get('wg_rap_shortfall', 0)) for s in self.history[-num_sq:]) / num_sq

    @property
    def current_fl_shortfall(self) -> float:
        num_sq = len(self.squadrons)
        if len(self.history) < num_sq: 
            return 0.0
        return sum(max(0, s.get('fl_rap_shortfall', 0)) for s in self.history[-num_sq:]) / num_sq

    @property
    def current_ip_shortfall(self) -> float:
        num_sq = len(self.squadrons)
        if len(self.history) < num_sq: 
            return 0.0
        return sum(max(0, s.get('ip_rap_shortfall', 0)) for s in self.history[-num_sq:]) / num_sq

    def reset(self):
        self.history = []
        self.current_year = 2025

    def add_new_bcourse_graduates(self, year: int, count: int, round_robin: bool): 
        num_sq = len(self.squadrons)
        if num_sq == 0:
            return
        
        for i in range(count):
            if not round_robin:
                target_sq = max(self.squadrons, key=lambda s: (s.experience_ratio, -s.total_pilots))
            else:
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
            target_sq.update_stats()

        
    def run_simulation(self, years_to_run: int, annual_intake: int, retention_rate: float, squadron_configs: List[SquadronConfig], ute: float = 10.0):
        """
        squadron_configs: list -> [Config(id=1, paa=12...), Config(id=2, paa=24...)]
        """
        self.history = []
        self.squadrons = squadron_configs

        FEATURE_NAMES = ['paa', 'ute', 'exp_ratio', 'total_pilots', 'mqt_qty', 'flug_qty', 'ipug_qty', 'ip_qty']

        for sq in self.squadrons:
            sq.ute = ute # With current implementation all squadrons must have same UTE
            sq.manning_pct = self.max_manning

        for year in range(self.current_year, self.current_year + years_to_run):
            phase_intake = annual_intake // 3
            remainder = annual_intake % 3

            for phase_num in range(1, 4): 
                current_batch = phase_intake + (remainder if phase_num == 3 else 0)
                self.add_new_bcourse_graduates(year, current_batch, self.round_robin) 


                for sq in self.squadrons:
                    if sq.flug_students != 0 or sq.ipug_students != 0:
                        raise AssertionError(f'Critical Data Mismatch in Squadron Pilots')
                    sq.new_phase_upgrades(self.flug_window_start, self.ipug_window_start)

                if self.sim_upgrades:
                    FEATURE_NAMES_EXPANDED = [
                                    'paa', 'ute', 'exp_ratio', 'total_pilots', 
                                    'mqt_qty', 'flug_qty', 'ipug_qty', 'ip_qty',
                                    'ip_ratio', 'ip_to_stud_ratio'
                                ]

                    batch_data = []

                    for sq in self.squadrons:
                        vec = sq.get_feature_vector()

                        total_students = sq.mqt_students + sq.flug_students + sq.ipug_students
                        ip_ratio = sq.ip_qty / max(sq.total_pilots, 1)
                        ip_to_stud_ratio = sq.ip_qty / (total_students if total_students > 0 else 0.1)

                        vec.extend([ip_ratio, ip_to_stud_ratio])
                        batch_data.append(vec)

                    df_batch = pd.DataFrame(batch_data, columns=FEATURE_NAMES_EXPANDED)

                    wg_rates = self.brain['wg_monthly'].predict(df_batch)
                    fl_rates = self.brain['fl_monthly'].predict(df_batch)
                    ip_rates = self.brain['ip_monthly'].predict(df_batch)

                    wg_blue_rates = self.brain['wg_blue_monthly'].predict(df_batch)
                    fl_blue_rates = self.brain['fl_blue_monthly'].predict(df_batch)
                    ip_blue_rates = self.brain['ip_blue_monthly'].predict(df_batch)

                for i, sq in enumerate(self.squadrons):
                    if self.sim_upgrades:
                        monthly_rates = AgingRate(4.0, wg_rates[i], fl_rates[i], ip_rates[i],
                                          4.0, wg_blue_rates[i], fl_blue_rates[i], 
                                          ip_blue_rates[i])
                        rates =  monthly_rates.monthly_to_phase(sq.phase_length_days)

                    else:
                        rates = sq.calc_aging_rate(False)

                    sq.apply_phase_aging(rates)

                    current_stats = sq.store_stats(year, phase_num, rates)

                    self.process_end_of_phase(sq, year, phase_num, retention_rate, current_stats) 
            
        return pd.DataFrame(self.history)
    

    def process_end_of_phase(self, sq: SquadronConfig, year: int, phase_num: int, retention_rate: float, current_stats: dict):
        
        staff_ips = 0
        staff_fls = 0
        separated_count = 0
        retained_count = 0

        sq.graduate_current_upgrades()

        sq.send_to_staff(priority_mode=self.staff_priority)

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

        sq.update_stats()

    def check_stability(self, phases_per_year=3, years=2, pop_threshold=100.0):
        num_phases = phases_per_year * years
        if len(self.history) < num_phases:
            return False, None, None
        
        df = pd.DataFrame(self.history)

        # Full Series Data
        pop_series = df.groupby(['year', 'phase'])['total_pilots'].sum().reset_index()

        # Check End of Simulation Stability
        def is_window_stable(window_series, threshold):
            std_dev = window_series.std()
            is_smooth = std_dev < threshold

            drift = abs(window_series.iloc[-1] - window_series.iloc[0])
            is_flat = drift < threshold

            return is_smooth and is_flat, (std_dev, drift)
        
        recent_window = pop_series['total_pilots'].tail(num_phases)
        is_stable_at_end, (recent_std, recent_drift) = is_window_stable(recent_window, pop_threshold)

        equilibrium_point = None
        if is_stable_at_end:
            for i in range(len(pop_series) - num_phases):
                window = pop_series['total_pilots'].iloc[i : i + num_phases]
                stable, _ = is_window_stable(window, pop_threshold)

                if stable:
                    row = pop_series.iloc[i]
                    equilibrium_point = (int(row['year']), int(row['phase']))
                    break

        return is_stable_at_end, (recent_std, recent_drift), equilibrium_point

    def get_simulation_grade_card(self, phases_per_year=3, stable_years=2, pop_threshold =100.0):
        if not self.history:
            return "No data"
        
        df = pd.DataFrame(self.history)

        max_year = df['year'].max()
        max_phase = df[df['year'] == max_year]['phase'].max()

        final_snapshot = df[(df['year'] == max_year) & (df['phase'] ==max_phase)]
        recent_history = df[df['year'] > (max_year - stable_years)]
        
        total_line_pilots = final_snapshot['line_pilots'].sum()
        total_pilots = final_snapshot['total_pilots'].sum()
        total_staff_pilots = total_pilots - total_line_pilots

        # Aggregate across squadrons
        aggregated_recent_history = recent_history.groupby(['year', 'phase']).agg({
            'wg_rap_shortfall': 'mean',
            'wg_blue_shortfall': 'mean',
            'fl_rap_shortfall': 'mean',
            'fl_blue_shortfall': 'mean',
            'ip_rap_shortfall': 'mean',
            'ip_blue_shortfall': 'mean',
            'exp_rat': 'mean'
        }).reset_index()

        # Mean for last 2 years
        avg_wg_delta = aggregated_recent_history['wg_rap_shortfall'].mean()
        avg_wg_blue_delta = aggregated_recent_history['wg_blue_shortfall'].mean()
        avg_fl_delta = aggregated_recent_history['fl_rap_shortfall'].mean()
        avg_fl_blue_delta = aggregated_recent_history['fl_blue_shortfall'].mean()
        avg_ip_delta = aggregated_recent_history['ip_rap_shortfall'].mean()
        avg_ip_blue_delta = aggregated_recent_history['ip_blue_shortfall'].mean()
        avg_exp_ratio = aggregated_recent_history['exp_rat'].mean()

        is_stable_at_end, (recent_std, recent_drift), equilbrium_point = self.check_stability(phases_per_year, stable_years, pop_threshold)

        return {
            "is_stable": is_stable_at_end,
            "when_stable": equilbrium_point,
            "series_end_std": round(recent_std, 2) if recent_std is not None else 0.0,
            "series_end_drift": round(recent_drift, 2) if recent_drift is not None else 0.0,
            "avg_wg_shortfall": round(avg_wg_delta, 2),
            "avg_wg_blue_shortfall": round(avg_wg_blue_delta, 2),
            "avg_fl_shortfall": round(avg_fl_delta, 2),
            "avg_fl_blue_shortfall": round(avg_fl_blue_delta, 2),
            "avg_ip_shortfall": round(avg_ip_delta, 2),
            "avg_ip_blue_shortfall": round(avg_ip_blue_delta, 2),
            "final_exp_ratio": round(avg_exp_ratio, 2),
            "final_line_pilots": round(total_line_pilots),
            "final_total_pilots": round(total_pilots),
            "final_staff_pilots": round(total_staff_pilots)
        }
