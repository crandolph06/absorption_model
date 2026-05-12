import os
import pandas as pd
from stable_baselines3 import PPO
from src.manning_engine import CAFSimulation
from src.manning_gym import ManningEnv
from src.models import PriorityMode
import joblib

def run_evaluation(run_mode="pragmatic", reward_mode="readiness_first"):
    print("🚀 Initializing Evaluation Engine...")
    brain_path = "brains/hpc_sortie_brain_multi_output_mlp.pkl"
    
    if not os.path.exists(brain_path):
        raise FileNotFoundError(f"Could not find brain at {brain_path}")
        
    brain = joblib.load(brain_path)
    
    sim_engine = CAFSimulation(
        sim_upgrades=True,
        annual_intake=200,
        retention_rate=0.40,
        brain=brain,
        flug_window_start=250,
        ipug_window_start=400,
        max_manning_pct=125,
        staff_priority_mode=PriorityMode.RANDOM,
        use_upgrade_quotas=True,
        round_robin=False
    )
    
    env = ManningEnv(sim_engine, run_mode=run_mode, reward_mode=reward_mode)

    # Standardized to match the exact save structure from your training loop
    model_path = f"rl_agents/ppo_manning_agent_{reward_mode}_{run_mode}"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find RL agent at {model_path}")
    print(f"🧠 Loading RL Brain from {model_path}...")
    model = PPO.load(model_path)

    obs, info = env.reset()
    terminated = False
    truncated = False
    
    history = []
    
    # Map the action indices to actual meanings for the pragmatic mode
    action_names = ["Intake", "FLUG", "IPUG", "Max Manning"]
    if run_mode in ["current", "ideal", "optimistic"]:
        action_names.append("UTE")
        action_names.append("Retention")
    if run_mode in ["ideal", "optimistic"]:
        action_names.append("PAA")
        
    while not (terminated or truncated):
        # deterministic=True forces the agent to take what it believes is the optimal path
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Calculate real shortfalls for the log
        total_shortfall = (sim_engine.current_wg_shortfall + 
                           sim_engine.current_fl_shortfall + 
                           sim_engine.current_ip_shortfall)
                           
        avg_ute = sum(sq.ute for sq in sim_engine.squadrons) / max(len(sim_engine.squadrons), 1) if sim_engine.squadrons else 0
        
        # Build the phase record
        record = {
            "Year": sim_engine.current_year,
            "Phase": sim_engine.current_phase,
            "Reward": reward,
            "Total Pilots": sim_engine.total_active_pilot_count,
            "Total Shortfall": total_shortfall,
            "Intake Target": sim_engine.annual_intake,
            "Retention Rate": sim_engine.retention_rate,
            "Max Manning": sim_engine.max_manning,
            "Avg UTE": avg_ute
        }
        
        # Map the 0,1,2 actions to -1,0,1 so they plot cleanly on a chart (-1=Decrease, 0=Hold, 1=Increase)
        for i, name in enumerate(action_names):
            record[f"Action: {name}"] = action[i] - 1 
            
        history.append(record)

    return pd.DataFrame(history)

if __name__ == "__main__":
    df_results = run_evaluation()
    print(df_results.head())