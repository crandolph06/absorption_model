import gymnasium as gym
from gynmasium import spaces
import numpy as np

class ManningEnv(gym.Env):
    def __init__(self, sim_engine, run_mode="ideal"):
        super(ManningEnv, self).__init__()
        self.sim = sim_engine
        self.run_mode = run_mode
        self.initial_intake = sim_engine.annual_intake
        self.initial_ute = sim_engine.ute
        self.initial_paa = sim_engine.paa

        if run_mode == "ideal":
            # [B-Course, FLUG, IPUG, UTE, PAA] -> increase, maintain, decrease for each
            self.action_space = spaces.MultiDiscrete([3, 3, 3, 3, 3])

        elif run_mode == "optimistic":
            # [B-Course, FLUG, IPUG, UTE, PAA] -> increase, maintain, decrease for each
            self.action_space = spaces.MultiDiscrete([3, 3, 3, 3, 3])
            # TODO Cap PAA to 20% increase

        elif run_mode == "pragmatic":
            # [B-Course, FLUG, IPUG, UTE] -> increase, maintain, decrease for each
            self.action_space = spaces.MultiDiscrete([3, 3, 3, 3])
            # TODO Cap UTE increase at 50%

        elif run_mode == "current":
            # [B-Course, FLUG, IPUG] -> increase, maintain, decrease for each
            self.action_space = spaces.MultiDiscrete([3, 3, 3])

    def _apply_current_logic(self, action, run_mode):
        intake_act = action[0]
        flug_act = action[1]
        ipug_act = action [2]
        ute_act = action [3] if len(action) > 3 else 1
        paa_act = action [4] if len(action) > 4 else 1

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
            max_ute = int(self.initial_ute * 1.5)
        if ute_act == 0:
            self.sim.ute = max(1, self.sim.ute - 1)
        elif ute_act == 2:
            self.sim.ute = min(max_ute, self.sim.ute + 1)

        # PAA 
        if run_mode == "optimistic":
            max_paa = 30
        if paa_act == 0:
            for sq in self.sim.squadrons:
                if paa_act == 0:
                    sq.paa = max(1, sq.paa - 1)
                elif paa_act == 2:
                    sq.paa = min(max_paa, sq.paa + 1)