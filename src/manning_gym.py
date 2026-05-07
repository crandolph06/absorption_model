import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.manning_config import get_initial_squadrons

class ManningEnv(gym.Env):
    def __init__(self, sim_engine, run_mode="ideal", reward_mode="quantity_first"):
        super(ManningEnv, self).__init__()
        self.sim = sim_engine
        self.run_mode = run_mode
        self.reward_mode = reward_mode
        self.initial_intake = sim_engine.annual_intake
        self.initial_retention = sim_engine.retention_rate

        if run_mode == "ideal":
            # [B-Course, FLUG, IPUG, max manning, UTE, retention, PAA] -> increase, maintain, decrease for each
            self.action_space = spaces.MultiDiscrete([3, 3, 3, 3, 3, 3, 3])

        elif run_mode == "optimistic":
            # [B-Course, FLUG, IPUG, max manning, UTE, retention, PAA] -> increase, maintain, decrease for each
            self.action_space = spaces.MultiDiscrete([3, 3, 3, 3, 3, 3, 3])
            # PAA per sq capped to 30, retention capped at 65%

        elif run_mode == "pragmatic":
            # [B-Course, FLUG, IPUG, max manning, UTE, retention] -> increase, maintain, decrease for each
            self.action_space = spaces.MultiDiscrete([3, 3, 3, 3, 3, 3])
            # UTE capped to 15, retention capped at 50%

        elif run_mode == "current":
            # [B-Course, FLUG, IPUG, max manning] -> increase, maintain, decrease for each
            self.action_space = spaces.MultiDiscrete([3, 3, 3, 3])

        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(13,), dtype=np.float32 
        )

    def _apply_current_logic(self, action, run_mode):
        intake_act = action[0]
        flug_act = action[1]
        ipug_act = action [2]
        man_pct_act = action[3]
        ute_act = action [4] if len(action) > 4 else 1
        ret_act = action [5] if len(action) > 5 else 1
        paa_act = action [6] if len(action) > 6 else 1

        # B-Course Intake (Phase) 
        if intake_act == 0: 
            self.sim.annual_intake = max(10, self.sim.phase_intake - 10)
        elif intake_act ==2: 
            self.sim.annual_intake = min(350, self.sim.phase_intake + 10)

        # FLUG Intake (Phase)
        if flug_act == 0: 
            self.sim.sq_phase_flug_intake = max(0, self.sim.sq_phase_flug_intake - 1)
        elif flug_act ==2: 
            self.sim.sq_phase_flug_intake = min(10, self.sim.sq_phase_flug_intake + 1)

        # IPUG Intake (Phase) 
        if ipug_act == 0: 
            self.sim.sq_phase_ipug_intake = max(0, self.sim.sq_phase_ipug_intake - 1)
        elif ipug_act ==2: 
            self.sim.sq_phase_ipug_intake = min(10, self.sim.sq_phase_ipug_intake + 1)

        # Manning Percentage 
        if man_pct_act == 0:
            self.sim.max_manning = max(0, self.sim.max_manning - 0.05)
        elif man_pct_act == 2:
            self.sim.max_manning = min(1.5, self.sim.max_manning + 0.05)

        # UTE 
        if run_mode == "optimistic":
            max_ute = 20
        elif run_mode == "pragmatic":
            max_ute = 15
        else: max_ute = 30

        if ute_act == 0:
            for sq in self.sim.squadrons:
                sq.ute = max(1, sq.ute - 1)
        elif ute_act == 2:
            for sq in self.sim.squadrons:
                sq.ute = min(max_ute, sq.ute + 1)

        # Retention
        if run_mode == "optimistic":
            max_retention = .65
        elif run_mode == "pragmatic":
            max_retention = .50
        else: max_retention = 1.0

        if ret_act == 0:
            self.sim.retention_rate = max(.1, self.sim.retention_rate - 0.05)
        elif ret_act == 2:
            self.sim.retention_rate = min(max_retention, self.sim.retention_rate + 0.05)

        # PAA 
        if run_mode == "optimistic":
            max_paa = 30
        else: max_paa = 48
        for sq in self.sim.squadrons:
            if paa_act == 0:
                sq.paa = max(1, sq.paa - 1)
            elif paa_act == 2:
                sq.paa = min(max_paa, sq.paa + 1)

    def reset(self, seed=None):
        super().reset(seed=seed)

        self.sim.annual_intake = self.initial_intake
        self.sim.retention_rate = self.initial_retention

        self.sim.current_year = 2026
        self.sim.current_phase = 1

        self.sim.squadrons = get_initial_squadrons(self.sim.current_year)

        observation = self._get_obs()
        return observation, {}
    
    def step(self, action):
            self._apply_current_logic(action, self.run_mode)

            self.sim.run_phase() 
            observation = self._get_obs()

            reward = self._calculate_reward()

            terminated = self.sim.current_year >= 2046 
            truncated = False
            
            return observation, reward, terminated, truncated, {}
    
    def _calculate_reward(self): 
        current_total = self.sim.total_active_pilot_count
        line_count = self.sim.total_line_pilot_count
        staff_count = self.sim.total_staff_pilot_count
        wg_short = self.sim.current_wg_shortfall
        fl_short = self.sim.current_fl_shortfall
        ip_short = self.sim.current_ip_shortfall

        if self.reward_mode == "quantity_first":
            # Get to 3500 total pilots (line and staff) first, then focus on line RAP
            return self._reward_quantity(current_total, wg_short, fl_short, ip_short)
        elif self.reward_mode == "readiness_first":
            # Get to line pilot RAP first, then increase toward 3500 total pilots
            return self._reward_readiness(current_total, wg_short, fl_short, ip_short)
        elif self.reward_mode == "key_staff_first":
            # Get to 20% of staff pilot positions manned ((3500-line pilots) * .2), then focus on line RAP, then increase toward 3500 total pilots
            return self._reward_key_staff(current_total, line_count, staff_count, wg_short, fl_short, ip_short)

    
    def _reward_quantity(self, current_total, wg_short, fl_short, ip_short):
        reward = current_total * 0.1

        if current_total >= 3500:
            reward += 100.0

            if wg_short > 0 or fl_short > 0 or ip_short > 0:
                reward -= wg_short * 40
                reward -= (fl_short + ip_short) * 20

        return reward
    
    def _reward_readiness(self, current_total, wg_short, fl_short, ip_short):
        reward = 0.0

        if wg_short + fl_short + ip_short > 0.5:
            reward -= wg_short * 50.0
            reward -= (fl_short + ip_short) * 25.0
        else:
            reward += 100.0
            reward += (current_total * 0.1)

            if current_total >= 3500:
                reward += 200.0

        return reward
    
    def _reward_key_staff(self, current_total, line_count, staff_count, wg_short, fl_short, ip_short, key_staff_ratio=0.2):
        reward = 0.0
        target_staff = (3500 - line_count) * key_staff_ratio

        if staff_count < target_staff:
            reward = (staff_count / max(target_staff, 1)) * 50.0
            return reward
        
        reward += 50.0

        if sum(wg_short, fl_short, ip_short) > 0.5:
            reward -= wg_short * 50.0
            reward -= (fl_short + ip_short) * 25.0
            return reward
        
        reward += 100.0

        reward += (current_total * 0.1)
        if current_total >= 3500:
            reward += 200.0

        return reward

    def _get_obs(self):
            total_paa = sum(sq.paa for sq in self.sim.squadrons)
            avg_ute = sum(sq.ute for sq in self.sim.squadrons) / max(len(self.sim.squadrons), 1)
            
            total_pilots = self.sim.total_active_pilot_count
            staff_pilots = self.sim.total_staff_pilot_count
            line_pilots = self.sim.total_line_pilot_count
            total_ips = self.sim.total_ip_qty
            total_fls = self.sim.total_fl_qty
            total_wg = self.sim.total_wg_qty
            exp_ratio = self.sim.experience_ratio
            wg_short = self.sim.current_wg_shortfall
            fl_short = self.sim.current_fl_shortfall
            ip_short = self.sim.current_ip_shortfall
            
            current_intake = self.sim.annual_intake

            return np.array([
                total_paa,
                avg_ute,
                total_pilots,
                staff_pilots,
                line_pilots,
                total_ips,
                total_fls,
                total_wg,
                exp_ratio,
                wg_short,
                fl_short,
                ip_short,
                current_intake
            ], dtype=np.float32)