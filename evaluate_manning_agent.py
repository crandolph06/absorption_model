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

    model_path = f"saved_models/ppo_manning_agent_{reward_mode}_{run_mode}"
    print(f"🧠 Loading RL Brain from {model_path}...")
    model = PPO.load(model_path)

    obs, info = env.reset()
    terminated = False
    truncated = False
    
    history = []
    
    # Map the action indices to actual meanings
    action_names = ["Intake", "FLUG", "IPUG", "Max Manning", "UTE", "Retention"]
    if run_mode in ["ideal", "optimistic"]:
        action_names.append("PAA")
        
    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        avg_ute = sum(sq.ute for sq in sim_engine.squadrons) / max(len(sim_engine.squadrons), 1) if sim_engine.squadrons else 0
        total_paa = sum(sq.paa for sq in sim_engine.squadrons) if sim_engine.squadrons else 0
        
        # Build the phase record (Added Staff Pilots, FLUG, IPUG, and PAA)
        record = {
            "Year": sim_engine.current_year,
            "Phase": sim_engine.current_phase,
            "Reward": reward,
            "Total Pilots": sim_engine.total_active_pilot_count,
            "Total Staff Pilots": sim_engine.total_staff_pilot_count,
            "WG Shortfall": sim_engine.current_wg_shortfall,
            "FL Shortfall": sim_engine.current_fl_shortfall,
            "IP Shortfall": sim_engine.current_ip_shortfall,
            "Intake Target": sim_engine.annual_intake,
            "FLUG Intake": sim_engine.sq_phase_flug_intake,
            "IPUG Intake": sim_engine.sq_phase_ipug_intake,
            "Retention Rate": sim_engine.retention_rate,
            "Max Manning": sim_engine.max_manning,
            "Avg UTE": avg_ute,
            "Total PAA": total_paa,
            "Experience Ratio": sim_engine.experience_ratio,
            "Number of Squadrons": len(sim_engine.squadrons)
        }
        
        # Map the 0,1,2 actions to -1,0,1 so they plot cleanly on a chart
        for i, name in enumerate(action_names):
            record[f"Action: {name}"] = action[i] - 1 
            
        history.append(record)

    return pd.DataFrame(history)

if __name__ == "__main__":
    df_results = run_evaluation()
    print(df_results.head())