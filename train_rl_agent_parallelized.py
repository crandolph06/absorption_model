import os
import joblib
import time
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from src.manning_engine import CAFSimulation
from src.models import PriorityMode
from src.manning_gym import ManningEnv

start_time = time.time()

def load_ai_brain():
    pc_path = "brains/hpc_sortie_brain_multi_output_mlp.pkl"
    hpc_path = "outputs/single_phase/brains/hpc_sortie_brain_multi_output_mlp.pkl"

    if os.path.exists(pc_path):
        return joblib.load(pc_path)
    if os.path.exists(hpc_path):
        return joblib.load(hpc_path)

    raise FileNotFoundError(
        f"Could not find MLP multi-output brain at '{pc_path}' or '{hpc_path}'. "
        "Run 'hpc_train_brain_multi_output.py' to generate it."
    )

def parse_mode_list(env_var_name, default_modes):
    raw_value = os.getenv(env_var_name, "").strip()
    if not raw_value:
        return default_modes
    return [mode.strip() for mode in raw_value.split(",") if mode.strip()]

def make_env(rank, run_mode, reward_mode, seed=0):
    def _init():
        brain = load_ai_brain()
        sim_engine = CAFSimulation(
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
        env = ManningEnv(sim_engine, run_mode=run_mode, reward_mode=reward_mode)
        check_env(env, warn=True)
        print("Environment check passed!")

        log_sub_dir = f"logs/env_{rank}"
        os.makedirs(log_sub_dir, exist_ok=True)
        return Monitor(env, log_sub_dir)
    return _init
        
def main():
    run_modes = parse_mode_list("RUN_MODES", ["pragmatic", "optimistic", "current", "ideal"])
    reward_modes = parse_mode_list("REWARD_MODES", ["readiness_first", "quantity_first", "key_staff_first"])

    n_procs = int(os.getenv('SLURM_NTASKS', 1))
    print(f"📡 Slurm allocated {n_procs} tasks. Launching parallel environments...")
    timesteps = int(os.getenv("TIMESTEPS", 1_000_000))

    for run_mode in run_modes[:1]:
        for reward_mode in reward_modes[:1]:
            print(f"🚀 Training combination: run_mode='{run_mode}', reward_mode='{reward_mode}'")
            env_functions = [make_env(i, run_mode, reward_mode) for i in range(n_procs)]
            env = SubprocVecEnv(env_functions)

            model = PPO(
                "MlpPolicy", env, n_steps=2048, batch_size=1024, verbose=1,
                tensorboard_log=f"./ppo_manning_tensorboard/{run_mode}_{reward_mode}/"
            )

            print("Starting Training Loop...")
            model.learn(total_timesteps=timesteps, progress_bar=True)
            end_time = time.time()
            duration = str(datetime.timedelta(seconds=int(end_time - start_time)))

            print(f"✅ Learning Complete. Total duration: {duration} (HH:MM:SS)")

            os.makedirs("saved_models", exist_ok=True)
            model_path = f"saved_models/ppo_manning_agent_{reward_mode}_{run_mode}"
            model.save(model_path)
            print(f"Training complete. Model saved to {model_path}.zip")
            env.close()

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

