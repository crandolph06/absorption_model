# USAF Pilot Readiness & Manning Model

Python model for exploring how pilot production, squadron manning, and training capacity interact over time. The project has four complementary paths:

1. **Single-phase physics** — detailed sortie/sim allocation for one squadron over a 120-day training phase (syllabus events, RAP, capacity limits).
2. **Long-term manning model** — multi-squadron CAF simulation over 20+ years. Two plant options:
   - **Brain path (default):** a learned “sortie brain” predicts monthly flying rates from manning inputs.
   - **Physics path:** `run_phase_simulation` runs per squadron each phase (capacity-constrained, same rules as Layer 1).
3. **Simulation optimization** — search constant CAF levers (intake, retention, upgrade quotas, UTE, etc.) by minimizing an explicit objective on the **physics** manning rollout (`optimize_constant_policy.py`).
4. **RL policy search** — reinforcement learning on top of the manning model to explore levers phase-by-phase under clipped gym rewards (today still uses the brain plant).

Layers 1 → 2 (brain) → 4 was the original stack. **Path 3** (physics + direct optimization) is the preferred rigorous route for policy recommendations: same allocator as Layer 1, explicit cost function instead of a surrogate brain or PPO rewards. RL on the physics plant is a natural follow-on comparison.

---

## Quick start (local)

**Requirements:** Python 3.11, dependencies in `requirements.txt`.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Sortie brain:** Place the trained model at:

```
brains/hpc_sortie_brain_multi_output_mlp.pkl
```

Copy from HPC after training (`outputs/single_phase/brains/`) or use an artifact a teammate shares.

**Run Streamlit apps** (from repo root):

```bash
streamlit run app.py           # Layer 1 + brain: single-phase “what-if” dashboard
streamlit run manning_app.py   # Layer 2: 20-year CAF manning simulation
streamlit run rl_app.py        # Layer 3: evaluate trained RL policies on manning sim
```

**Run single-phase physics only** (no brain):

```bash
python -m src.main
```

**Run physics-backed policy optimization** (no brain; slow on full CAF):

```bash
# Fast iteration: 1 test squadron
python optimize_constant_policy.py --preset test --years 5 --trials 20

# Full ~30-squadron CAF (profile runtime first)
python optimize_constant_policy.py --preset full --years 10 --method de --trials 30 --output results/best_policy.json
```

---

## Repository map

```
absorption_model/
├── src/                    # Core simulation logic (start here)
│   ├── engine.py           # Single-phase: pilot creation, sortie/sim allocation, syllabi
│   ├── models.py           # Pilot, SquadronConfig, enums, RAP constants, manning aging
│   ├── simulation_config.py # SimulationConfig: phase length, allocation noise (fleet-wide variables)
│   ├── syllabi.py          # MQT / FLUG / IPUG event definitions
│   ├── rules.py            # Who can fly, upgrade eligibility
│   ├── rap_state.py        # RAP shortfall assessment (sortie + sim)
│   ├── manning_engine.py   # Layer 2: CAFSimulation (brain or physics plant)
│   ├── manning_objective.py # Explicit scalar cost J for simulation optimization
│   ├── manning_config.py   # Layer 2: initial 30-squadron CAF roster (PAA, IPs, targets)
│   ├── manning_main.py     # Layer 2: setup_simulation() helper for apps
│   └── manning_gym.py      # Layer 4: Gymnasium env (ManningEnv, SingleActionManningEnv)
│
├── optimize_constant_policy.py  # Path 3: random / scipy DE search on constant levers
├── app.py                  # Layer 1 UI (+ brain predictions for sweeps)
├── manning_app.py          # Layer 2 UI: long-horizon manning + sensitivity
├── rl_app.py               # Layer 4 UI: load PPO agent + manning charts
│
├── brains/                 # Local copy of trained MLP (not always in git)
│
├── hpc_sweepers/single_phase/
│   └── hpc_single_phase_sweeper_*.py   # Parameter sweeps → parquet batches
├── hpc_train_brain_multi_output.py     # Train multi-output MLP on sweep data
├── verify_multi_brain.py               # Sanity-check brain vs parquet
│
├── tools/
│   └── do_repartition.py   # Dask: many batch parquets → ~50 part files for training
│
├── rl_trainers/             # PPO training scripts (parallel + single-action variants)
├── slurm/                   # HPC job scripts (sweeps, train, repartition, RL)
├── evaluate_manning_agent.py
└── archive/                 # Older experiments (safe to ignore unless referenced)
```

---

## Three complementary layers

### Layer 1 — Single-phase physics (`src/engine.py`)

**What it does:** For one `SquadronConfig`, creates pilots, runs syllabus programs (MQT/FLUG/IPUG), continuation training, and sim RAP. Sorties are allocated against **hard capacity** `PAA × UTE × phase_months`, with a per-pilot monthly event cap (`MAX_MONTHLY_EVENTS` in `models.py`). Phase length comes from `SimulationConfig` (default 120 days), not squadron config. Syllabus requirements and CT makeup are based on 20 FW flying training.

**Key functions:**

| Function | Role |
|----------|------|
| `create_pilots()` | Build WG / FL / IP roster from config |
| `run_phase_simulation()` | Full phase: upgrades → syllabus → CT → sim RAP (`sim_config` optional) |
| `assign_sortie()` / `assign_sim()` | Pick lowest-utilization pilot (total events, then type-specific tie-break) |
| `rap_assess()` | Per-cohort sortie RAP vs targets (`rap_state.py`) |

**Entry points:** `src/main.py`, HPC sweepers (`hpc_sweepers/single_phase/`), `app.py` (brain predicts rates for slider sweeps; physics is separate).

**Output:** Parquet sweep data used to train the sortie brain (Layer 2).

### Layer 2 — Long-term manning model (`src/manning_engine.py`)

**What it does:** ~30 squadrons over many years. Each CAF phase:

1. Adds B-course graduates  
2. Starts upgrades (FLUG / IPUG windows and optional per-phase quotas)  
3. **Plant step** (one of two modes):
   - **Brain (default):** sortie brain → `apply_phase_aging()` with predicted rates; graduation via `graduate_current_upgrades(deferrals)` from brain deferral outputs.
   - **Physics:** `run_phase_simulation()` per squadron; allocator deferrals via `apply_deferred_burden_to_squadron`; graduation via `graduate_completed_upgrades()` inside `process_end_of_phase`, then ADSC countdown, staff funnel, and retention (same end-of-phase order as brain except plant step).
4. Staff funnel, retention, history  

Enable physics plant: `CAFSimulation(..., use_physics_allocator=True, brain=None)`.

**Brain path caveat:** Does **not** re-run `engine.py` each phase. RAP shortfall in history is `RAP_target − brain_predicted_rate`, not capacity balance.

**Physics path:** RAP shortfall comes from observed pilot rates via `store_stats_from_physics()` (WG cohort excludes MQT only; FLUG/IPUG remain in their qual cohorts).

**Entry points:** `manning_app.py`, `src/manning_main.py` (`setup_simulation()`). Apps today use the brain path; physics is used from `optimize_constant_policy.py` and programmatic rollouts.

### Path 3 — Simulation optimization (`optimize_constant_policy.py`, `src/manning_objective.py`)

**What it does:** Rolls out the **physics** manning model for N years and minimizes an explicit cost `J` (lower is better):

- Weighted mean RAP shortfall (WG / FL / IP)  
- Manning gap vs 3,500 target at horizon  
- Mean deferral burden (sortie line slots)  
- Small bonus for experience ratio  

**Policy levers searched (constant over horizon):** annual intake, retention, max manning %, FLUG/IPUG phase quotas, UTE. FLUG/IPUG **gates** (250 sorties / 400 hrs vs 150 / 300) are intended as fixed scenario conditions or a separate sensitivity sweep—not jointly optimized with quotas on the first pass.

**Methods:** `--method random` (default) or `--method de` (scipy differential evolution; falls back to random if scipy missing).

**Presets:** `--preset test` (one squadron, fast), `--preset full` (30-squadron `SQUADRON_DATA`).

### Layer 4 — RL policy search (`src/manning_gym.py`, `rl_trainers/`)

**What it does:** Wraps the Layer 2 `CAFSimulation` in a Gymnasium environment. A PPO agent adjusts policy levers each phase (which levers depend on `run_mode`) to maximize a `reward_mode` objective (headcount vs RAP shortfall vs key staff).

**Entry points:** `rl_trainers/train_rl_parallelized_*.py`, `evaluate_manning_agent.py`, `rl_app.py` (load a saved policy and roll out 20 years with charts).

**Depends on:** A trained sortie brain (Layer 2 brain path) and optionally a trained PPO checkpoint in `saved_models/`. RL on the physics plant is not wired yet; Path 3 is the recommended first policy-search baseline.

---

## Sortie brain (MLP) — bridge from Layer 1 to Layer 2 (brain path)

The brain is trained on Layer 1 sweep data and consumed by the Layer 2 manning model when `use_physics_allocator=False` (and by Layer 4 RL today).

**File:** `brains/hpc_sortie_brain_multi_output_mlp.pkl`  
**Trainer:** `hpc_train_brain_multi_output.py`  
**Verifier:** `verify_multi_brain.py`

### Inputs (9 features)

Must match `CAFSimulation._PREDICT_FEATURE_COLS` in `manning_engine.py`:

`paa`, `ute`, `exp_ratio`, `ip_ratio`, `fl_congestion`, `wg_crowding`, `sorties_avail`, `pilot_to_sortie`, `ip_to_stud_ratio`

### Outputs — **12 vs 16** (read this before changing code)

The codebase is transitioning to a **16-output** brain. Layout in training script:

| Index | Target |
|-------|--------|
| 0–2 | `wg_monthly`, `fl_monthly`, `ip_monthly` |
| 3–5 | `wg_blue_monthly`, `fl_blue_monthly`, `ip_blue_monthly` |
| 6–9 | `mqt_sim_monthly`, `wg_sim_monthly`, `fl_sim_monthly`, `ip_sim_monthly` *(16-output only)* |
| 6–11 or 10–15 | Remaining syllabus deferrals (total + sorties-only) |

**Current state (check before running):**

- **`app.py` / `manning_app.py` / `rl_app.py`** — syllabus charts use **12-output indices** (6–11), with 16-output lines commented for later.
- **`manning_engine.py`** — still maps **16-output** layout (sim at 6–9, deferrals at 10–15).

If your `.pkl` has 12 outputs, manning sim and apps must agree on indexing. Verify with:

```bash
python -c "import joblib, numpy as np; m=joblib.load('brains/hpc_sortie_brain_multi_output_mlp.pkl'); print(m.predict(np.zeros((1,9))).shape)"
```

---

## HPC pipeline (background)

Typical order on the cluster (`$WORKDIR/absorption_model`):

```
Layer 1 — data generation
  1. Single-phase sweeps      slurm/submit_sweep_*.slurm  →  outputs/single_phase/parquet/batch_*.parquet
  2. (Optional) Low-exp sweep submit_sweep_low.slurm     →  batch_low_*.parquet

Layer 1 → 2 — brain training
  3. Repartition              submit_repartition.slurm    →  repart_parquet/part.*.parquet
  4. Train brain              submit_train_multi.slurm    →  outputs/single_phase/brains/*.pkl
  5. Verify                   submit_verify.slurm
  6. Copy .pkl to local       brains/

Path 3 — simulation optimization (local / future HPC)
  7. Constant-policy search   optimize_constant_policy.py  →  results/*.json

Layer 4 — RL (optional)
  8. RL training              slurm/submit_train_rl_parallelized_*.slurm
```

**Repartition:** `tools/do_repartition.py` globs all `parquet/*.parquet` and writes ~50 parts. Training samples 10% of high-exp rows and keeps 100% of `exp_ratio ≤ 0.10` plus all `batch_low_*` files (see trainer constants).

**Submit many Slurm jobs:**

```bash
cd slurm
for f in submit_train_rl_parallelized_*.slurm; do sbatch "$f"; done
```

HPC sessions are limited to **4 hours**; sweep scripts use `timeout` and checkpoint batch files for resume.

---

## RL training (Layer 4)

| Location | Description |
|----------|-------------|
| `rl_trainers/train_rl_parallelized_*.py` | Multi-action PPO; naming: `{curr,opt,ideal,prag}_{keys,qty,read}_{book,real}` |
| `rl_trainers/single_action/` | One lever per step (`SingleActionManningEnv`) |
| `train_rl_agent.py` | Simpler entry script |
| `evaluate_manning_agent.py` | Roll out a saved policy to CSV |

**Run modes:** `ideal`, `optimistic`, `pragmatic`, `current` — control which levers the agent can move and caps (UTE, retention, PAA).

**Reward modes:** `quantity_first`, `readiness_first`, `key_staff_first`.

**Saved models:** `saved_models/` (paths vary by script; `rl_app.py` loads from sidebar selection).

---

## Domain concepts (short glossary)

| Term | Meaning |
|------|---------|
| **PAA** | Primary assigned aircraft per squadron |
| **UTE** | Utilization rate (sorties per aircraft per month) |
| **RAP** | Ready Aircrew Program — monthly sortie/sim targets by qual (WG 9/mo sorties, FL/IP 8/mo, sim 3/mo) |
| **Phase** | Typical phase length (default 120 days via `SimulationConfig`); 3 phases per year |
| **MQT / FLUG / IPUG** | Upgrade syllabi (wingman → FL → IP) |
| **Exp ratio** | (IPs + FLs) / line pilots |
| **Staff funnel** | Over-manned line IPs/FLs moved to staff billets |

---

## Conventions & gotchas

1. **Import paths:** Run apps and scripts from **repo root** so `src.*` and `brains/` resolve correctly.
2. **Brain / code alignment:** Mismatched 12 vs 16 outputs silently breaks syllabus charts or sim deferrals. Always verify output count after pulling a new `.pkl`.
3. **Layer 1 vs Layer 2 (brain):** Single-phase sweeps train the brain on **capacity-constrained** outcomes; the brain manning path applies predicted rates **without** re-enforcing `PAA × UTE` fleet totals. Use `use_physics_allocator=True` when conclusions must match Layer 1 rules.
4. **Physics end-of-phase:** `process_end_of_phase` calls `graduate_completed_upgrades` (allocator-native), decrements ADSC, then staff funnel and retention — matching brain end-of-phase except aging came from the allocator.
5. **`total_pilots` in training data** = line pilot count in manning prediction, not including staff.
6. **`SimulationConfig`:** Fleet-wide phase length and (Layer 1) allocation noise live in `src/simulation_config.py`, not on `SquadronConfig`. Pass `sim_config=` to `run_phase_simulation()` or `CAFSimulation(...)`.
7. **`archive/`:** Historical scripts; not part of the active pipeline unless you know you need them.

---

## Current BIG PROBLEMS
1. **RL as end state:** Unsure the best way to explore and communicate the intra-variable interactions since most are non-linear.
2. **Reward hacking:** RL agents find their way to edge cases (primarily VERY low experience ratio -- < .10) where the ML brain performs poorly due to different physics (FLs and IPs hit monthly sortie maximum, resulting in different sortie allocation logic). 
3. **Inconsistent rules throughout pipeline:** Brain-based long-term sim and RL do not re-run Layer 1 allocation; errors can be large. **Mitigation:** Path 3 (`use_physics_allocator=True` + `manning_objective.py`) — slower but rule-consistent. Brain path remains useful for fast what-if and training-data generation.
4. **Runtime:** Full CAF × 20 years × one `run_phase_simulation` per squadron per phase is expensive; profile before large optimization runs.

---

## Suggested reading order for new collaborators

**Layer 1 — physics & brain training**

1. `src/models.py` — data structures and RAP constants  
2. `src/engine.py` — how sorties are actually allocated (single phase)  
3. `src/rap_state.py` — how shortfall is scored in sweepers  
4. `hpc_train_brain_multi_output.py` — what the brain learns  
5. `app.py` — interactive single-phase exploration  

**Layer 2 — long-term manning**

6. `src/manning_engine.py` — brain vs physics plant, `run_phase` flow  
7. `manning_app.py` — full CAF run + charts (brain path)  

**Path 3 — simulation optimization**

8. `src/manning_objective.py` — explicit cost `J` and breakdown  
9. `optimize_constant_policy.py` — rollout driver and search  

**Layer 4 — RL**

10. `src/manning_gym.py` — action space, observations, rewards  
11. `rl_trainers/train_rl_parallelized_*.py` — PPO training loop  
12. `rl_app.py` — policy evaluation UI  

---

## Contact / ownership

Maj Claire "Buzzer" Randolph - 15 AF CAG/Special Projects |
claire.randolph@us.af.mil |
DSN: 965-4147 (Comm prefix 803-895-XXXX)
