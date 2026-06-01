"""Manual smoke test: a few gym steps without training. Not used by CI or train scripts."""
import os

from evaluate_manning_agent import _joblib_load_brain
from src.manning_gym import ManningEnv
from src.manning_engine import CAFSimulation


def load_ai_brain():
    for path in (
        "brains/hpc_sortie_brain_multi_output_mlp.pkl",
        "outputs/single_phase/brains/hpc_sortie_brain_multi_output_mlp.pkl",
    ):
        if os.path.exists(path):
            return _joblib_load_brain(path)
    raise FileNotFoundError("No sortie brain .pkl found under brains/ or outputs/single_phase/brains/")


if __name__ == "__main__":
    sim = CAFSimulation(
        use_upgrade_quotas=True,
        annual_intake=150,
        retention_rate=0.4,
        round_robin=False,
        brain=load_ai_brain(),
    )
    env = ManningEnv(sim, run_mode="optimistic")

    obs, _ = env.reset()
    for i in range(1, 10):
        action = env.action_space.sample()
        print(f"Executing Step {i}...")
        obs, rew, term, trunc, info = env.step(action)
        print(
            f"  simulated {info['simulated_year']} P{info['simulated_phase']} "
            f"-> clock now {env.sim.current_year} P{env.sim.current_phase} reward={rew:.3f}"
        )
        if term or trunc:
            print(f"Episode ended at step {i}.")
            break
