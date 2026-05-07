import os
import joblib
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from src.manning_engine import CAFSimulation
from src.models import PriorityMode
from src.manning_gym import ManningEnv

def make_env(rank, seed=0):
    def _init():
        brain = joblib.load("outputs/single_phase/brains/hpc_sortie_brain_multi_output.pkl")      
        sim_engine = CAFSimulation(
            sim_upgrades=True,
            annual_intake=200,
            retention_rate=0.40,
            round_robin=False,
            brain=brain,
            flug_window_start=250, # Likely needs to change to reality - ~150
            ipug_window_start=400, # Likely needs to change to reality - ~300
            max_manning_pct=125,
            staff_priority_mode=PriorityMode.RANDOM,
            use_upgrade_quotas=True,  
        )
        env = ManningEnv(sim_engine, run_mode="pragmatic", reward_mode="readiness_first")
        check_env(env, warn=True)
        print("Environment check passed!")

        log_sub_dir = f"logs/env_{rank}"
        os.makedirs(log_sub_dir, exist_ok=True)
        return Monitor(env, log_sub_dir)
        
def main():
    n_procs = int(os.getenv('SLURM_NTASKS', 1))
    print(f"📡 Slurm allocated {n_procs} tasks. Launching parallel environments...")

    env_functions = [make_env(i) for i in range(n_procs)]

    env = SubprocVecEnv(env_functions)

    model = PPO(
        "MlpPolicy", env, n_steps=2048, batch_size=1024, verbose=1,
        tensorboard_log="./ppo_manning_tensorboard/"
    )

    print("Starting Training Loop...")
    timesteps = 1_000_000 # Eventually change to 5M
    model.learn(total_timesteps=timesteps, progress_bar=True)

    os.makedirs("saved_models", exist_ok=True)
    model_path = "saved_models/ppo_manning_agent_prag_ready"
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