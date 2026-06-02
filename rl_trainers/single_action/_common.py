"""Shared PPO training loop for single-action Manning RL jobs."""

import datetime
import os
import time

import joblib
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.manning_engine import CAFSimulation
from src.manning_gym import SingleActionManningEnv
from src.models import PriorityMode

_GATE_CONFIG = {
    "book": {
        "models_subdir": "saved_models/single_action/book_gates",
        "flug_window_start": 250,
        "ipug_window_start": 400,
    },
    "real": {
        "models_subdir": "saved_models/single_action/real_gates",
        "flug_window_start": 150,
        "ipug_window_start": 300,
    },
}


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


def make_env(rank, run_mode, reward_mode, gate_type, seed=0):
    gate = _GATE_CONFIG[gate_type]

    def _init():
        brain = load_ai_brain()
        sim_engine = CAFSimulation(
            annual_intake=200,
            retention_rate=0.40,
            round_robin=False,
            brain=brain,
            flug_window_start=gate["flug_window_start"],
            ipug_window_start=gate["ipug_window_start"],
            max_manning_pct=125,
            staff_priority_mode=PriorityMode.RANDOM,
            use_upgrade_quotas=True,
        )
        env = SingleActionManningEnv(
            sim_engine, run_mode=run_mode, reward_mode=reward_mode
        )
        check_env(env, warn=True)

        log_sub_dir = f"logs/single_action/env_{rank}"
        os.makedirs(log_sub_dir, exist_ok=True)
        return Monitor(env, log_sub_dir)

    return _init


def main(run_mode: str, reward_mode: str, gate_type: str) -> None:
    if gate_type not in _GATE_CONFIG:
        raise ValueError(f"gate_type must be one of {list(_GATE_CONFIG)}; got {gate_type!r}")

    gate = _GATE_CONFIG[gate_type]
    start_time = time.time()

    n_procs = int(os.getenv("SLURM_NTASKS", 1))
    timesteps = int(os.getenv("TIMESTEPS", 1_000_000))

    print(f"📡 Slurm allocated {n_procs} tasks. Launching parallel environments...")
    print(
        f"🚀 Single-action training: gate={gate_type}, run_mode={run_mode!r}, "
        f"reward_mode={reward_mode!r}"
    )

    env_functions = [
        make_env(i, run_mode, reward_mode, gate_type) for i in range(n_procs)
    ]
    env = SubprocVecEnv(env_functions)

    tb_dir = f"./ppo_manning_tensorboard/single_action/{gate_type}/{run_mode}_{reward_mode}/"
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2048,
        batch_size=1024,
        verbose=1,
        tensorboard_log=tb_dir,
    )

    print("Starting Training Loop...")
    model.learn(total_timesteps=timesteps, progress_bar=True)

    duration = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"✅ Learning Complete. Total duration: {duration} (HH:MM:SS)")

    models_dir = gate["models_subdir"]
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(
        models_dir, f"ppo_manning_agent_{reward_mode}_{run_mode}"
    )
    model.save(model_path)
    print(f"Training complete. Model saved to {model_path}.zip")
    env.close()
