from typing import Optional
import copy
import joblib
from src.models import PriorityMode
from src.manning_engine import CAFSimulation
from src.manning_config import get_initial_squadrons, SQUADRON_DATA

def setup_simulation(round_robin: bool, annual_intake: int,
                     ai_brain, 
                     existing_sim: Optional[CAFSimulation] = None, 
                     flug_window_start: int = 250, ipug_window_start: int = 400, 
                     max_manning_pct: int = 150, retention_rate: float = .4,
                     staff_priority_mode: PriorityMode = PriorityMode.RANDOM):
    if existing_sim:
        sim = copy.deepcopy(existing_sim)
        sim.reset()

        sim.annual_intake = annual_intake
        sim.phase_intake = annual_intake // 3 # APPROXIMATE +/- 2
        sim.round_robin = round_robin
        sim.flug_window_start = flug_window_start
        sim.ipug_window_start = ipug_window_start
        sim.max_manning = max_manning_pct / 100
        sim.staff_priority = staff_priority_mode
        sim.retention_rate = retention_rate

        sim.brain = ai_brain

        if len(sim.squadrons) > 0:
            return sim, sim.squadrons  
    
    else:
        sim = CAFSimulation(round_robin=round_robin,
                            brain = ai_brain, flug_window_start=flug_window_start, 
                            ipug_window_start=ipug_window_start, max_manning_pct=max_manning_pct, 
                            staff_priority_mode=staff_priority_mode, annual_intake=annual_intake,
                            retention_rate=retention_rate)

    # Used 1.5 CCR for all units
    # Used 50% of exp pilots as starting IP value 

    squadrons = get_initial_squadrons(2026, SQUADRON_DATA)

    for sq in squadrons:
        sq.update_stats()
    
    return sim, squadrons

# if __name__ == "__main__":
#     # brain = joblib.load('outputs/single_phase/brains') # For HPC
#     brain = joblib.load('brains/hpc_sortie_brain_multi_output_mlp.pkl') # For local
#     sim, squadrons = setup_simulation(round_robin=False, ai_brain=brain, annual_intake=150, retention_rate = .4)

#     results_df = sim.run_simulation(
#         years_to_run=1,   
#         squadron_configs=squadrons,
#     )