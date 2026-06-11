import pandas as pd
from typing import List, Optional, Tuple
from src.models import Pilot, Qual, SquadronConfig, Upgrade, Assignment, PriorityMode, AgingRate
from src.simulation_config import SimulationConfig
import os
import numpy as np
import joblib


class CAFSimulation:
    def __init__(self, annual_intake: int, retention_rate: float, 
                 round_robin: bool, brain = None, flug_window_start: int = 250, 
                 ipug_window_start: int = 400, max_manning_pct: int = 150, 
                 staff_priority_mode: PriorityMode = PriorityMode.RANDOM,
                 use_upgrade_quotas: bool = False,
                 sim_config: Optional[SimulationConfig] = None,
                 use_physics_allocator: bool = False):
        self.history = []
        self.current_year = 2026
        self.current_phase = 1
        self.squadrons: List[SquadronConfig] = []
        self.flug_window_start = flug_window_start # Sorties for FLUG auto-start
        self.ipug_window_start = ipug_window_start # Hours for IPUG auto-start
        self.max_manning = max_manning_pct/100
        self.staff_priority = staff_priority_mode
        self.annual_intake = annual_intake
        self.phase_intake = annual_intake // 3 # APPROXIMATE +/- 2
        self.retention_rate = retention_rate
        self.use_upgrade_quotas = use_upgrade_quotas
        self.use_physics_allocator = use_physics_allocator
        if self.use_upgrade_quotas == False:
            self.sq_phase_flug_intake = 999
            self.sq_phase_ipug_intake = 999
        else:
            self.sq_phase_flug_intake = 3 
            self.sq_phase_ipug_intake = 2 

        if brain:
            self.brain = brain
        elif use_physics_allocator:
            self.brain = None
        else:
            for path in (
                "brains/hpc_sortie_brain_multi_output_mlp_16_out.pkl",
                "brains/hpc_sortie_brain_multi_output_mlp.pkl",
            ):
                if os.path.exists(path):
                    print(f"🧠 Loading Sortie Brain from {path}...")
                    self.brain = joblib.load(path)
                    break
            else:
                raise FileNotFoundError(
                    "Could not find brains/hpc_sortie_brain_multi_output_mlp*.pkl"
                )
        self.round_robin = round_robin
        self.sim_config = sim_config or SimulationConfig()
        self._brain_output_count: Optional[int] = None

    _DEFERRAL_NEGLIGIBLE = 0.10

    def _brain_n_outputs(self) -> int:
        if self._brain_output_count is None:
            sample = np.zeros((1, len(self._PREDICT_FEATURE_COLS)))
            preds = self.brain.predict(sample)
            self._brain_output_count = int(preds.shape[1])
        return self._brain_output_count

    def _phase_rates_from_brain_row(
        self, mqt_mo: float, row: np.ndarray, phase_length_days: float
    ) -> AgingRate:
        """Monthly ``AgingRate`` from one brain row, scaled to a phase (sorties + sim monthly)."""
        if self._brain_n_outputs() >= 16:
            sim = row[6:10]
        else:
            sim = (0.0, 0.0, 0.0, 0.0)
        monthly = AgingRate(
            mqt_mo, row[0], row[1], row[2],
            mqt_mo, row[3], row[4], row[5],
            mqt_sim_phase=sim[0],
            wg_sim_phase=sim[1],
            fl_sim_phase=sim[2],
            ip_sim_phase=sim[3],
            mqt_sim_blue_phase=sim[0],
            wg_sim_blue_phase=sim[1],
            fl_sim_blue_phase=sim[2],
            ip_sim_blue_phase=sim[3],
        )
        return monthly.monthly_to_phase(phase_length_days)

    def _clean_deferral_frac(self, value: float) -> float:
        if value < self._DEFERRAL_NEGLIGIBLE:
            return 0.0
        return max(0.0, float(value))

    def _deferrals_from_brain_row(self, row: np.ndarray) -> Tuple[float, float, float, float, float, float]:
        """
        Brain deferral outputs are syllabi fractions; convert to sortie/sim line slots.

        12-output: combined 6–8, sorties-only 9–11.
        16-output: combined 10–12, sorties-only 13–15 (6–9 are sim monthly rates).
        """
        from src.syllabi import (
            SORTIE_BURDEN_FLUG,
            SORTIE_BURDEN_IPUG,
            SORTIE_BURDEN_MQT,
            SIM_BURDEN_FLUG,
            SIM_BURDEN_IPUG,
            SIM_BURDEN_MQT,
        )

        if self._brain_n_outputs() >= 16:
            combined = row[10:13]
            sorties_only = row[13:16]
        else:
            combined = row[6:9]
            sorties_only = row[9:12]

        sortie_burdens = (SORTIE_BURDEN_MQT, SORTIE_BURDEN_FLUG, SORTIE_BURDEN_IPUG)
        sim_burdens = (SIM_BURDEN_MQT, SIM_BURDEN_FLUG, SIM_BURDEN_IPUG)

        sortie_slots = []
        sim_slots = []
        for combined_frac, sortie_frac, sortie_burden, sim_burden in zip(
            combined, sorties_only, sortie_burdens, sim_burdens
        ):
            sortie_f = self._clean_deferral_frac(sortie_frac)
            combined_f = self._clean_deferral_frac(combined_frac)
            sim_f = max(0.0, combined_f - sortie_f)
            sortie_slots.append(sortie_f * sortie_burden)
            sim_slots.append(sim_f * sim_burden)

        return (
            sortie_slots[0], sortie_slots[1], sortie_slots[2],
            sim_slots[0], sim_slots[1], sim_slots[2],
        )

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
    def line_ips(self):
        return [p for p in self.line_pilots if p.qual.name == 'IP']

    @property
    def line_fls(self):
        return [p for p in self.line_pilots if p.qual.name == 'FL']

    @property
    def total_line_pilot_count(self):
        return len(self.line_pilots)

    @property
    def line_ip_count(self):
        return len(self.line_ips)

    @property
    def line_fl_count(self):
        return len(self.line_fls)

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
        total_experienced = self.line_ip_count + self.line_fl_count
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
        self.current_year = 2026
        self.current_phase = 1

    def advance_clock(self):
        """Advance one CAF phase (RL gym calls after each ``run_phase``)."""
        self.current_phase += 1
        if self.current_phase > 3:
            self.current_phase = 1
            self.current_year += 1

    def _resolve_mqt_monthly(self, sq: SquadronConfig) -> float:
        """
        Monthly MQT sortie rate for ``AgingRate`` when blending with the sortie brain.

        Uses last phase's observed rate when available. Otherwise: no MQT students or
        no line IPs → 0. Else use 2.2 (9 sorties / 4 month phase) (first phase before
        ``observed_mqt_monthly`` is recorded).
        """
        if sq.observed_mqt_monthly is not None:
            return float(sq.observed_mqt_monthly)
        if sq.mqt_students <= 0:
            return 0.0
        if sq.ip_qty <= 0:
            return 0.0
        return 2.2 

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
                flight_hours_flown=50.0,
                sim_hours_flown=0.0,
                sorties_flown=50,
                sorties_at_upgrade_start=50,
                sims_flown=0,
                sims_at_upgrade_start=0,
            ))

            target_sq.pilots.append(new_pilot)
            target_sq.update_stats()

    def _run_squadron_physics_phase(self, sq: SquadronConfig, year: int, phase_num: int):
        """One CAF phase using ``run_phase_simulation`` instead of the sortie brain."""
        from src.engine import run_phase_simulation
        from src.rap_state import mqt_observed_sortie_metrics

        phase_days = self.sim_config.phase_length_days
        run_phase_simulation(
            sq,
            sq.pilots,
            debug_verbose=False,
            pre_seed_upgrades=False,
            sim_config=self.sim_config,
        )

        mqt_metrics = mqt_observed_sortie_metrics(sq.pilots)
        if mqt_metrics["sortie_mo"] > 0:
            sq.observed_mqt_monthly = mqt_metrics["sortie_mo"]

        current_stats = sq.store_stats_from_physics(year, phase_num, phase_days)
        self.process_end_of_phase(
            sq,
            year,
            phase_num,
            self.retention_rate,
            current_stats,
            deferrals=(0, 0, 0, 0, 0, 0),
            skip_graduation=True,
        )

    def run_phase(self, phase_num: int, year: int):

        # print(f"Running phase {phase_num} of {year}")

        self.phase_intake = self.annual_intake // 3
        remainder = self.annual_intake % 3
        current_batch = self.phase_intake + (remainder if phase_num == 3 else 0)
        self.add_new_bcourse_graduates(year, current_batch, self.round_robin) 


        for sq in self.squadrons:
            sq.manning_pct = self.max_manning
            use_upgrade_quotas = self.use_upgrade_quotas
            flug_quota = self.sq_phase_flug_intake
            ipug_quota = self.sq_phase_ipug_intake
            sq.new_phase_upgrades(
                flug_window_start=self.flug_window_start, ipug_window_start=self.ipug_window_start, 
                use_upgrade_quotas=use_upgrade_quotas, flug_quota=flug_quota, ipug_quota=ipug_quota)
            sq.update_stats()            

        if self.use_physics_allocator:
            for sq in self.squadrons:
                self._run_squadron_physics_phase(sq, year, phase_num)
            return

        preds = self.predict_rates_fast()

        for i, sq in enumerate(self.squadrons):
            row = preds[i]
            mqt_mo = self._resolve_mqt_monthly(sq)
            phase_days = self.sim_config.phase_length_days
            rates = self._phase_rates_from_brain_row(mqt_mo, row, phase_days)
            mqt_baseline = {
                id(p): p.sorties_flown
                for p in sq.pilots
                if p.upgrade == Upgrade.MQT and p.active
            }

            sq.apply_phase_aging(rates, phase_days)

            if mqt_baseline:
                months = self.sim_config.phase_length_months
                if months > 0:
                    deltas = [
                        p.sorties_flown - mqt_baseline[id(p)]
                        for p in sq.pilots
                        if p.upgrade == Upgrade.MQT and id(p) in mqt_baseline
                    ]
                    if deltas:
                        sq.observed_mqt_monthly = sum(deltas) / len(deltas) / months

            current_stats = sq.store_stats(year, phase_num, rates, phase_days)

            deferrals = self._deferrals_from_brain_row(row)
            self.process_end_of_phase(sq, year, phase_num, self.retention_rate, current_stats, deferrals) 
            
        
    def run_simulation(self, years_to_run: int, squadron_configs: List[SquadronConfig], ute: float = 10.0):
        self.history = []
        self.squadrons = squadron_configs

        for sq in self.squadrons:
            sq.ute = ute # With current implementation all squadrons must have same UTE

        for year in range(self.current_year, self.current_year + years_to_run):
            for phase_num in range(1, 4): 
                self.run_phase(phase_num, year)
            
        return pd.DataFrame(self.history)
    

    def process_end_of_phase(self, sq: SquadronConfig, year: int, phase_num: int, retention_rate, current_stats: dict, deferrals: Tuple[int, int, int, int, int, int], skip_graduation: bool = False):
        
        staff_ips = 0
        staff_fls = 0
        separated_count = 0
        retained_count = 0

        if not skip_graduation:
            sq.graduate_current_upgrades(deferrals, sorties_only=False)

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
                # if p.upgrade != Upgrade.NONE:
                #     print(f"Pilot is moving to staff in an upgrade status.")

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

    # Feature column order must match training / predict_rates (MLP pipeline).
    _PREDICT_FEATURE_COLS = [
        'paa', 'ute', 'exp_ratio', 'ip_ratio', 'fl_congestion',
        'wg_crowding', 'sorties_avail', 'pilot_to_sortie', 'ip_to_stud_ratio',
    ]

    def predict_rates_fast(self) -> np.ndarray:
        """One DataFrame build per call; same feature math as legacy predict_rates."""
        batch_records = []

        for sq in self.squadrons:
            paa = sq.paa
            ute = sq.ute
            # Training uses column name total_pilots but value is line pilot count (cockpit strength).
            total_pilots = sq.line_pilots
            exp_ratio = sq.experience_ratio
            mqt_qty = sq.mqt_students + sq.mqt_sortie_carry
            flug_qty = sq.flug_students + sq.flug_sortie_carry
            ipug_qty = sq.ipug_students + sq.ipug_sortie_carry
            wg_qty = sq.wg_qty
            fl_qty = sq.fl_qty
            ip_qty = sq.ip_qty

            fls = fl_qty if fl_qty != 0 else 1.0
            wgs = wg_qty if wg_qty != 0 else 1.0

            fl_congestion = (ipug_qty + flug_qty) / fls
            wg_crowding = (mqt_qty + flug_qty + ipug_qty) / wgs

            sorties_avail = paa * ute
            pilot_to_sortie = (
                total_pilots / sorties_avail if sorties_avail != 0 else 0.0
            )

            total_students = mqt_qty + flug_qty + ipug_qty
            denom_tp = total_pilots if total_pilots != 0 else 1
            ip_ratio = ip_qty / denom_tp
            denom_stud = total_students if total_students != 0 else 0.1
            ip_to_stud_ratio = ip_qty / denom_stud

            batch_records.append({
                'paa': paa,
                'ute': ute,
                'exp_ratio': exp_ratio,
                'ip_ratio': ip_ratio,
                'fl_congestion': fl_congestion,
                'wg_crowding': wg_crowding,
                'sorties_avail': sorties_avail,
                'pilot_to_sortie': pilot_to_sortie,
                'ip_to_stud_ratio': ip_to_stud_ratio,
            })

        X = pd.DataFrame(batch_records, columns=self._PREDICT_FEATURE_COLS)
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        return self.brain.predict(X)
