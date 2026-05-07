import os
import joblib
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from src.manning_engine import CAFSimulation
from src.models import PriorityMode
from src.manning_gym import ManningEnv

def load_ai_brain():
    if not os.path.exists("brains/hpc_sortie_brain_lite.pkl"):
        print("Warning: HPC Brain models not found. Check file path for HPC sortie brain.")
    return joblib.load("brains/hpc_sortie_brain_lite.pkl")      

cached_brain = load_ai_brain()
      
def main():
    sim_upgrades = True
    sim_engine = CAFSimulation(
        sim_upgrades=sim_upgrades,
        annual_intake=200,
        retention_rate=0.40,
        round_robin=False,
        brain=cached_brain,
        flug_window_start=250, # Likely needs to change to reality - ~150
        ipug_window_start=400, # Likely needs to change to reality - ~300
        max_manning_pct=125,
        staff_priority_mode=PriorityMode.RANDOM,
        use_upgrade_quotas=True,  
    )

    # 3. Wrap the Engine in the Gym Environment
    raw_env = ManningEnv(sim_engine, run_mode="pragmatic", reward_mode="readiness_first")
    
    # 4. Run the Stable-Baselines3 Environment Checker
    print("Running environment compliance check...")
    check_env(raw_env, warn=True)
    print("Environment check passed!")

    os.makedirs("logs", exist_ok=True)
    env = Monitor(raw_env, "logs/")

    # 5. Initialize the PPO Agent
    print("Building the Neural Network...")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_manning_tensorboard/")

    # 6. Train the Agent
    print("Starting Training Loop...")
    timesteps = 100_000 # Eventually change to 1M, then 5M
    model.learn(total_timesteps=timesteps, progress_bar=True)

    # 7. Save the Model
    os.makedirs("saved_models", exist_ok=True)
    model_path = "saved_models/ppo_manning_agent"
    model.save(model_path)
    print(f"Training complete. Model saved to {model_path}.zip")

    # TODO Come back and plot 20-year run in each of the reward/run mode pairs
    # print("Running a quick evaluation phase...")
    # obs, info = env.reset()
    # for _ in range(10): # Step through a few phases
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, truncated, info = env.step(action)
    #     print(f"Action Taken: {action} | Reward: {reward}")
    #     if done or truncated:
    #         obs, info = env.reset()

if __name__ == "__main__":
    main()