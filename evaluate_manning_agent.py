import os
import sys

import joblib
import pandas as pd
from stable_baselines3 import PPO

from src.manning_engine import CAFSimulation
from src.manning_gym import ManningEnv
from src.models import PriorityMode


def _patch_numpy_bitgenerator_ctor() -> None:
    """Some pickles call ``__bit_generator_ctor(<class MT19937>)``; NumPy expects ``'MT19937'``."""
    import numpy.random._pickle as _rp

    if getattr(_rp, "_eval_manning_bitgen_ctor_patched", False):
        return

    _orig = _rp.__bit_generator_ctor

    def _wrapped(bit_generator_name="MT19937"):
        if not isinstance(bit_generator_name, str):
            bit_generator_name = getattr(bit_generator_name, "__name__", "MT19937")
        return _orig(bit_generator_name)

    _rp.__bit_generator_ctor = _wrapped
    _rp._eval_manning_bitgen_ctor_patched = True


def _is_rng_pickle_error(msg: str) -> bool:
    return any(
        s in msg
        for s in (
            "BitGenerator",
            "MT19937",
            "legacy MT19937",
            "not a known",
        )
    )


def _joblib_load_brain(path: str) -> object:
    """
    Load ``brains/hpc_sortie_brain_multi_output_mlp.pkl``.

    Tries plain ``joblib.load`` first, then applies the BitGenerator ctor shim once.
    """
    errors: list[BaseException] = []

    for attempt in ("plain", "patched"):
        try:
            if attempt == "patched":
                _patch_numpy_bitgenerator_ctor()
            return joblib.load(path)
        except Exception as e:
            errors.append(e)
            msg = str(e)
            if attempt == "plain" and _is_rng_pickle_error(msg):
                continue
            if attempt == "patched" and _is_rng_pickle_error(msg):
                break
            raise

    import numpy as np
    import sklearn

    detail = " | ".join(repr(e) for e in errors)
    raise RuntimeError(
        "Could not unpickle the sortie brain. Details: "
        f"{detail}. "
        f"Interpreter: {sys.executable}. "
        f"Versions here: numpy=={np.__version__}, scikit-learn=={sklearn.__version__}, joblib=={joblib.__version__}. "
        "Align with ``requirements.txt`` (numpy==2.4.4, scikit-learn==1.8.0, joblib==1.4.0). "
        "If Streamlit was started from another env, use the same interpreter (e.g. "
        "``python -m streamlit run rl_app.py`` after ``conda activate``). "
        "Otherwise re-run ``python hpc_train_brain_multi_output.py`` here and copy the new ``.pkl`` to ``brains/``."
    ) from errors[-1]


def _reraise_numpy_pickle_hint(where: str, exc: BaseException) -> None:
    """NumPy 1.x vs 2.x changes BitGenerator pickling; brain/PPO pickles fail across versions."""
    msg = str(exc)
    if "BitGenerator" in msg or "MT19937" in msg:
        raise RuntimeError(
            f"{where}: NumPy / pickle mismatch ({msg}). "
            "Try, in order: (1) ``pip install numpy==2.4.4`` to match requirements.txt; "
            "(2) if it still fails, ``pip install 'numpy>=1.26,<2'`` if the brain was saved under NumPy 1; "
            "(3) re-run ``python hpc_train_brain_multi_output.py`` in this env and copy the new .pkl to "
            "``brains/``. PPO checkpoints can hit the same issue—re-save after NumPy aligns."
        ) from exc
    raise exc


def run_evaluation(run_mode="pragmatic", reward_mode="readiness_first"):
    print("🚀 Initializing Evaluation Engine...")
    brain_path = "brains/hpc_sortie_brain_multi_output_mlp.pkl"

    if not os.path.exists(brain_path):
        raise FileNotFoundError(f"Could not find brain at {brain_path}")

    brain = _joblib_load_brain(brain_path)

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
    try:
        model = PPO.load(model_path)
    except Exception as e:
        _reraise_numpy_pickle_hint("PPO.load", e)

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

        # Build the phase record (year/phase = CAF tick simulated this step, pre-advance_clock)
        record = {
            "Year": info["simulated_year"],
            "Phase": info["simulated_phase"],
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
