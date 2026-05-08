import os
import pandas as pd
from stable_baselines3 import PPO
from src.manning_engine import CAFSimulation
from src.manning_gym import ManningEnv
from src.models import PriorityMode
import joblib

def evaluate():
    # 1. Setup the Environment (Must match training config)
    print("🚀 Initializing Evaluation Engine...")
    brain = joblib.load("brains/hpc_sortie_brain_multi_output_mlp.pkl") # For PC
    # brain = joblib.load("outputs/single_phase/brains/hpc_sortie_brain_multi_output_mlp.pkl") # For HPC
    
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
    
    env = ManningEnv(sim_engine, run_mode="pragmatic", reward_mode="readiness_first")

    # 2. Load the Trained RL Agent
    model_path = "saved_models/ppo_manning_agent_pragmatic_readiness"
    print(f"🧠 Loading RL Brain from {model_path}...")
    model = PPO.load(model_path)

    # 3. Run the "20-Year Test Flight"
    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    
    print("\n📅 --- 20-YEAR OPERATIONAL LOG ---")
    print(f"{'Year':<6} | {'Phase':<6} | {'Action (PAA/UTE)':<18} | {'Avg Manning':<12} | {'Reward':<8}")
    print("-" * 70)

    while not (terminated or truncated):
        # Predict the best action (deterministic=True is key for evaluation!)
        action, _states = model.predict(obs, deterministic=True)
        
        # Take the step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Log the state
        avg_manning = sum([sq.manning_pct for sq in sim_engine.squadrons]) / len(sim_engine.squadrons)
        print(f"{sim_engine.current_year:<6} | {sim_engine.current_phase:<6} | {str(action):<18} | {avg_manning:<12.1f} | {reward:<8.1f}")

    print("-" * 70)
    print(f"✅ Evaluation Complete. Total 20-Year Score: {total_reward:.2f}")

if __name__ == "__main__":
    evaluate()