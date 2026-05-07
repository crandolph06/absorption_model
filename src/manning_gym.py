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
        self.initial_ute = sim_engine.ute

        if run_mode == "ideal":
            # [B-Course, FLUG, IPUG, UTE, retention, PAA] -> increase, maintain, decrease for each
            self.action_space = spaces.MultiDiscrete([3, 3, 3, 3, 3, 3])

        elif run_mode == "optimistic":
            # [B-Course, FLUG, IPUG, UTE, retention, PAA] -> increase, maintain, decrease for each
            self.action_space = spaces.MultiDiscrete([3, 3, 3, 3, 3, 3])
            # PAA per sq capped to 30, retention capped at 65%

        elif run_mode == "pragmatic":
            # [B-Course, FLUG, IPUG, UTE, retention] -> increase, maintain, decrease for each
            self.action_space = spaces.MultiDiscrete([3, 3, 3, 3, 3])
            # UTE capped to 15, retention capped at 50%

        elif run_mode == "current":
            # [B-Course, FLUG, IPUG] -> increase, maintain, decrease for each
            self.action_space = spaces.MultiDiscrete([3, 3, 3])

        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(10,), dtype=np.float32
        )

    def _apply_current_logic(self, action, run_mode):
        intake_act = action[0]
        flug_act = action[1]
        ipug_act = action [2]
        ute_act = action [3] if len(action) > 3 else 1
        ret_act = action [4] if len(action) > 4 else 1
        paa_act = action [5] if len(action) > 5 else 1

        # B-Course Intake (Phase) # TODO ensure engine divides annual intake by 3
        if intake_act == 0: 
            self.sim.phase_intake = max(10, self.sim.phase_intake - 10)
        elif intake_act ==2: 
            self.sim.phase_intake = min(350, self.sim.phase_intake + 10)

        # FLUG Intake (Phase) # TODO need to figure out how to encode this in sim
        if flug_act == 0: 
            self.sim.phase_flug_intake = max(0, self.sim.phase_flug_intake - 1)
        elif flug_act ==2: 
            self.sim.phase_flug_intake = min(10, self.sim.phase_flug_intake + 1)

        # IPUG Intake (Phase) # TODO need to figure out how to encode this in sim
        if ipug_act == 0: 
            self.sim.phase_ipug_intake = max(0, self.sim.phase_ipug_intake - 1)
        elif ipug_act ==2: 
            self.sim.phase_ipug_intake = min(10, self.sim.phase_ipug_intake + 1)

        # UTE 
        if run_mode == "optimistic":
            max_ute = 20
        elif run_mode == "pragmatic":
            max_ute = 15
        if ute_act == 0:
            self.sim.ute = max(1, self.sim.ute - 1)
        elif ute_act == 2:
            self.sim.ute = min(max_ute, self.sim.ute + 1)

        # Retention
        if run_mode == "optimistic":
            max_retention = .65
        elif run_mode == "pragmatic":
            max_retention = .50
        if ret_act == 0:
            self.sim.retention = max(.1, self.sim.retention - 0.05)
        elif ret_act == 2:
            self.sim.retention = min(max_retention, self.sim.retention + 0.05)

        # PAA 
        if run_mode == "optimistic":
            max_paa = 30
        for sq in self.sim.squadrons:
            if paa_act == 0:
                sq.paa = max(1, sq.paa - 1)
            elif paa_act == 2:
                sq.paa = min(max_paa, sq.paa + 1)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.sim.annual_intake = self.initial_intake
        self.sim.ute = self.initial_ute
        self.sim.current_year = 2026

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

        num_sq = len(self.sim.squadrons)
        if len(self.sim.history) >= num_sq:
            latest_stats = self.sim.history[-num_sq:]

            total_shortfall = sum(
                max(0, s['wg_rap_shortfall']) +
                max(0, s['fl_rap_shortfall']) +
                max(0, s['ip_rap_shortfall'])
                for s in latest_stats
            )
            avg_shortfall = total_shortfall / (num_sq * 3)
        
        else:
            avg_shortfall = 0.0

        if self.reward_mode == "quantity_first":
            # Get to 3500 total pilots (line and staff) first, then focus on line RAP
            return self._reward_quantity(current_total, avg_shortfall)
        elif self.reward_mode == "readiness_first":
            # Get to line pilot RAP first, then increase toward 3500 total pilots
            return self._reward_readiness(current_total, avg_shortfall)
        elif self.reward_mode == "key_staff_first":
            # Get to 20% of staff pilot positions manned ((3500-line pilots) * .2), then focus on line RAP, then increase toward 3500 total pilots
            return self._reward_key_staff(current_total, line_count, staff_count, avg_shortfall)

    
    def _reward_quantity(self, current_total, avg_shortfall):
        reward = current_total * 0.1

        if current_total >= 3500:
            reward += 100.0

            if avg_shortfall > 0:
                reward -= (avg_shortfall * 20)

        return reward
    
    def _reward_readiness(self, current_total, avg_shortfall):
        reward = 0.0

        if avg_shortfall > 0.5:
            reward -= (avg_shortfall * 50.0)
        else:
            reward += 100.0
            reward += (current_total * 0.1)

            if current_total >= 3500:
                reward += 200.0

        return reward
    
    def _reward_key_staff(self, current_total, line_count, staff_count, avg_shortfall, key_staff_ratio=0.2):
        reward = 0.0
        target_staff = (3500 - line_count) * key_staff_ratio

        if staff_count < target_staff:
            reward = (staff_count / max(target_staff, 1)) * 50.0
            return reward
        
        reward += 50.0

        if avg_shortfall > 0.5:
            reward -= (avg_shortfall * 30.0)
            return reward
        
        reward += 100.0

        rewatd += (current_total * 0.1)
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
            
            num_sq = len(self.sim.squadrons)
            if len(self.sim.history) >= num_sq:
                latest_stats = self.sim.history[-num_sq:]
                total_shortfall = sum(
                    max(0, s.get('wg_rap_shortfall', 0)) + 
                    max(0, s.get('fl_rap_shortfall', 0)) + 
                    max(0, s.get('ip_rap_shortfall', 0)) 
                    for s in latest_stats
                )
                avg_shortfall = total_shortfall / (num_sq * 3)
            else:
                avg_shortfall = 0.0

            current_intake = self.sim.annual_intake

            return np.array([
                total_paa,
                avg_ute,
                total_pilots,
                staff_pilots,
                total_ips,
                total_fls,
                total_wg,
                exp_ratio,
                avg_shortfall,
                current_intake
            ], dtype=np.float32)