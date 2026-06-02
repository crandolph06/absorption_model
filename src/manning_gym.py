import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.manning_config import get_initial_squadrons

class ManningEnv(gym.Env):
    # Stable per-step reward scale for RL (used by quantity / readiness shaping).
    TARGET_TOTAL_PILOTS = 3500
    _STEP_REWARD_CLIP = 5.0
    # Share of unfilled line slots (vs target force) treated as key staff need — see _reward_key_staff.
    KEY_STAFF_RATIO = 0.2

    def __init__(self, sim_engine, run_mode="ideal", reward_mode="quantity_first"):
        super(ManningEnv, self).__init__()
        self.sim = sim_engine
        self.run_mode = run_mode
        self.reward_mode = reward_mode
        self.initial_intake = sim_engine.annual_intake
        self.initial_retention = sim_engine.retention_rate
        self.initial_max_manning = sim_engine.max_manning
        self.initial_flug_quota = sim_engine.sq_phase_flug_intake
        self.initial_ipug_quota = sim_engine.sq_phase_ipug_intake

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

        else:
            raise ValueError(f"Invalid run mode: {run_mode}")

        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(15,), dtype=np.float32 
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
            self.sim.annual_intake = max(10, self.sim.annual_intake - 10)
        elif intake_act ==2: 
            self.sim.annual_intake = min(350, self.sim.annual_intake + 10)

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
            self.sim.max_manning = min(2.0, self.sim.max_manning + 0.05)

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

        self.sim.reset()
        self.sim.annual_intake = self.initial_intake
        self.sim.retention_rate = self.initial_retention
        self.sim.phase_intake = self.initial_intake // 3
        self.sim.max_manning = self.initial_max_manning
        self.sim.sq_phase_flug_intake = self.initial_flug_quota
        self.sim.sq_phase_ipug_intake = self.initial_ipug_quota

        self.sim.squadrons = get_initial_squadrons(self.sim.current_year)

        observation = self._get_obs()
        return observation, {}
    
    def step(self, action):
            self._apply_current_logic(action, self.run_mode)

            simulated_year = self.sim.current_year
            simulated_phase = self.sim.current_phase
            self.sim.run_phase(simulated_phase, simulated_year)
            observation = self._get_obs()

            reward = self._calculate_reward()
            self.sim.advance_clock()

            terminated = self.sim.current_year >= 2046 
            truncated = False
            
            info = {
                "simulated_year": simulated_year,
                "simulated_phase": simulated_phase,
            }
            return observation, reward, terminated, truncated, info
    
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
            # Fill KEY_STAFF_RATIO of remaining slots toward TARGET_TOTAL_PILOTS, then blend into readiness-first shaping.
            return self._reward_key_staff(current_total, line_count, staff_count, wg_short, fl_short, ip_short)

    
    def _reward_quantity(self, current_total, wg_short, fl_short, ip_short):
        """
        Grow toward TARGET_TOTAL_PILOTS first, with small steady signals each step.

        - Headcount: smooth score from 0→4 as you approach full strength (no big jumps).
        - Readiness: small penalty every step; gets stronger as you near the goal so
          quantity stays the main focus early, but line health still nudges learning.
        - Final reward is clipped so the learner does not see huge spikes.
        """
        target = float(self.TARGET_TOTAL_PILOTS)
        progress = min(current_total / target, 1.0)

        headcount_points = 4.0 * progress

        # Near full strength, care more about shortfalls (still soft, not an on/off gate).
        readiness_weight = 0.15 + 0.85 * progress
        shortfall_penalty = readiness_weight * (
            0.15 * wg_short + 0.10 * fl_short + 0.10 * ip_short
        )

        reward = headcount_points - shortfall_penalty
        return float(np.clip(reward, -self._STEP_REWARD_CLIP, self._STEP_REWARD_CLIP))

    def _reward_readiness_unclipped(self, current_total, wg_short, fl_short, ip_short):
        """Same math as readiness-first, before clipping (shared with key_staff blend)."""
        target = float(self.TARGET_TOTAL_PILOTS)
        progress = min(current_total / target, 1.0)
        shortfall_cost = (
            0.75 * wg_short + 0.52 * fl_short + 0.52 * ip_short
        )
        return -shortfall_cost + self._STEP_REWARD_CLIP * progress

    def _reward_readiness(self, current_total, wg_short, fl_short, ip_short):
        """
        Readiness-first: shortfall penalties are the main drag; population progress
        scales the positive side up to the clip (best case ≈ +clip at no shortfall, 3500+ pilots).
        """
        reward = self._reward_readiness_unclipped(current_total, wg_short, fl_short, ip_short)
        return float(np.clip(reward, -self._STEP_REWARD_CLIP, self._STEP_REWARD_CLIP))

    def _reward_key_staff(self, current_total, line_count, staff_count, wg_short, fl_short, ip_short):
        """
        Key staff first: smooth emphasis on filling key staff slots, then crossfade into
        the same readiness + population logic as _reward_readiness (stable clip).
        """
        target = float(self.TARGET_TOTAL_PILOTS)
        ratio = self.KEY_STAFF_RATIO

        remaining_slots = max(target - line_count, 0.0)
        target_staff = remaining_slots * ratio

        if target_staff <= 1e-9:
            staff_progress = 1.0
        else:
            staff_progress = min(float(staff_count) / target_staff, 1.0)

        staff_term = self._STEP_REWARD_CLIP * staff_progress
        readiness_term = self._reward_readiness_unclipped(
            current_total, wg_short, fl_short, ip_short
        )

        reward = (1.0 - staff_progress) * staff_term + staff_progress * readiness_term
        return float(np.clip(reward, -self._STEP_REWARD_CLIP, self._STEP_REWARD_CLIP))

    def _get_obs(self):
            total_paa = sum(sq.paa for sq in self.sim.squadrons)
            avg_ute = sum(sq.ute for sq in self.sim.squadrons) / max(len(self.sim.squadrons), 1)
            
            total_pilots = self.sim.total_active_pilot_count
            staff_pilots = self.sim.total_staff_pilot_count
            line_pilots = self.sim.total_line_pilot_count
            total_ips = self.sim.total_ip_qty
            total_fls = self.sim.total_fl_qty
            total_wg = self.sim.total_wg_qty
            line_ips = self.sim.line_ip_count
            line_fls = self.sim.line_fl_count
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
                line_ips,
                line_fls,
                exp_ratio,
                wg_short,
                fl_short,
                ip_short,
                current_intake
            ], dtype=np.float32)


class SingleActionManningEnv(ManningEnv):
    """
    One policy lever may change per step; all others hold.

    Action is MultiDiscrete([n_levers, 3]): lever index, then 0=decrease / 1=hold / 2=increase.
    """

    _LEVER_COUNT = {
        "ideal": 7,
        "optimistic": 7,
        "pragmatic": 6,
        "current": 4,
    }
    _ACTION_DIM = _LEVER_COUNT

    def __init__(self, sim_engine, run_mode="ideal", reward_mode="quantity_first"):
        super().__init__(sim_engine, run_mode=run_mode, reward_mode=reward_mode)
        n_levers = self._LEVER_COUNT[run_mode]
        self.action_space = spaces.MultiDiscrete([n_levers, 3])

    def _apply_single_action(self, action, run_mode):
        n_levers = self._LEVER_COUNT[run_mode]
        lever_idx = int(np.clip(action[0], 0, n_levers - 1))
        direction = int(action[1])

        n_dims = self._ACTION_DIM[run_mode]
        full_action = np.ones(n_dims, dtype=int)
        if direction != 1:
            full_action[lever_idx] = direction
        self._apply_current_logic(full_action, run_mode)

    def step(self, action):
        self._apply_single_action(action, self.run_mode)

        simulated_year = self.sim.current_year
        simulated_phase = self.sim.current_phase
        self.sim.run_phase(simulated_phase, simulated_year)
        observation = self._get_obs()

        reward = self._calculate_reward()
        self.sim.advance_clock()

        terminated = self.sim.current_year >= 2046
        truncated = False

        info = {
            "simulated_year": simulated_year,
            "simulated_phase": simulated_phase,
        }
        return observation, reward, terminated, truncated, info
