from src.manning_gym import ManningEnv
from src.manning_engine import CAFSimulation

# Use your current engine setup
sim = CAFSimulation(
    use_upgrade_quotas=True, annual_intake=150,
    retention_rate=.4, round_robin=False) 
env = ManningEnv(sim, run_mode="optimistic")

obs, _ = env.reset()
for i in range(1, 10):
    # Take a manual action that usually causes the crash
    action = env.action_space.sample() 
    print(f"Executing Step {i}...")
    obs, rew, term, trunc, info = env.step(action)
    print(env.sim.current_year)
    if term or trunc:
        print(f"⚠️ Episode ended at step {i} without a crash error.")
        break