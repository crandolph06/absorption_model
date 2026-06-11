# PRD: Surrogate-Assisted Feasible-Envelope / Viability Analysis Prototype

**Repository:** `crandolph06/absorption_model`  
**Intended consumer:** Codex agent working in a local clone of the repository  
**Primary deliverable:** Prototype code path that answers whether the model can produce a ready force with the required number of pilots under allowable policy levers.

---

## 1. Executive Summary

Build a prototype for **feasible-envelope / viability analysis** on top of the existing absorption model.

The goal is not to train a new reinforcement learning agent and not initially to find a single mathematically optimal policy. The goal is to answer this decision question:

> Given the model, the initial force state, and allowed policy/resource levers, does there exist a feasible policy trajectory that produces both a ready force and the required number of pilots?

The prototype should:

1. Wrap the existing long-horizon manning model in a repeatable evaluator.
2. Define feasibility as explicit constraints instead of RL rewards.
3. Generate a small design of experiments over low-dimensional policy parameters.
4. Run direct model evaluations, preferably in parallel.
5. Fit first-pass surrogate models to constraint margins and aggregate violation.
6. Search the surrogate for feasible regions and boundary points.
7. Verify candidate feasible policies with the original model.
8. Produce simple tabular and graphical outputs showing feasible / infeasible / marginal regions.

This should be implemented as a prototype, not a full production framework.

---

## 2. Existing Repository Context

The repo already contains most of the needed ingredients.

### 2.1 Existing model layers

Relevant files to inspect before implementation:

- `src/manning_engine.py`
  - Contains `CAFSimulation`.
  - Handles long-horizon phase advancement, B-course graduates, upgrades, surrogate rate prediction, aging, retention, staff movement, and history logging.
  - Important methods: `run_phase`, `run_simulation`, `predict_rates_fast`, `process_end_of_phase`.

- `src/manning_gym.py`
  - Contains the RL environment `ManningEnv`.
  - Defines current RL levers: B-course intake, FLUG quota, IPUG quota, max manning, UTE, retention, PAA.
  - This file is useful for bounds and lever definitions, but the viability prototype should not depend on the RL `step()` logic.

- `hpc_train_brain_multi_output.py`
  - Current training path for the sortie/sim/deferral surrogate, referred to as the “brain.”
  - Uses engineered features and multi-output MLP regression.

- `hpc_sweepers/single_phase/hpc_single_phase_sweeper.py`
  - Existing single-phase design sweep.
  - Useful as a pattern for valid/invalid configuration handling, parallel evaluation, and output schema.

- `src/models.py`
  - Contains force entities and thresholds such as sortie RAP targets, sim RAP target, pilot state, squadron state, and stored statistics.

- `src/rap_state.py`
  - Contains RAP assessment helpers and RAP state encoding.

- `src/manning_config.py`
  - Contains initial squadron configuration helpers such as `get_initial_squadrons`.

### 2.2 Important integration point

For this prototype, do **not** replace the existing sortie/sim “brain.” The viability prototype should use the current model stack as-is, then build a higher-level surrogate over the **end-to-end policy-to-outcome response**.

There are therefore two surrogate concepts:

1. **Existing internal surrogate:** predicts sortie/sim rates inside the manning model.
2. **New viability surrogate:** predicts aggregate constraint violations and feasibility outcomes for policy designs.

The prototype should implement the second one while reusing the first one.

---

## 3. Problem Formulation

### 3.1 Decision question

Answer:

> Is there at least one allowable policy trajectory that yields enough pilots and sufficient readiness by the target year, and ideally maintains those requirements through the end of the analysis horizon?

### 3.2 Design variables

Start with a deliberately low-dimensional policy vector.

Initial constant-policy design vector:

```text
x = [
    annual_intake,
    retention_rate,
    ute,
    paa,
    max_manning_pct,
    flug_quota_per_phase,
    ipug_quota_per_phase
]
```

Approximate meanings:

| Variable | Meaning | Initial treatment |
|---|---|---|
| `annual_intake` | Annual B-course / new pilot intake | Integer or rounded continuous |
| `retention_rate` | Retention probability/rate | Continuous |
| `ute` | Utilization / sortie generation lever | Continuous or integer-like |
| `paa` | Aircraft/resource availability proxy | Integer or rounded continuous; apply to all squadrons initially |
| `max_manning_pct` | Maximum manning percent passed into `CAFSimulation` constructor | Continuous percent, e.g. 100–200 |
| `flug_quota_per_phase` | FLUG upgrade quota per squadron/phase when quotas are enabled | Integer or rounded continuous |
| `ipug_quota_per_phase` | IPUG upgrade quota per squadron/phase when quotas are enabled | Integer or rounded continuous |

The prototype should read bounds from config rather than hard-coding them. Reasonable defaults can be based on current RL logic and sweeper ranges.

### 3.3 Temporal parameterization

Do **not** optimize every lever independently for every phase at first. That would turn a small problem into a high-dimensional dynamic optimization problem.

Phase 1 should implement only a **constant policy** over the horizon.

Future policy parameterizations:

```text
constant policy:
    one value per lever for the full horizon

piecewise policy:
    one value per lever per multi-year block

ramp policy:
    initial value, final value, and ramp duration
```

The initial implementation should be structured so piecewise policies can be added later without rewriting the evaluator.

### 3.4 Constraints and aggregate violation

Define explicit constraints. A design is feasible if all configured constraints are satisfied.

Use this sign convention:

```text
g_j(x) <= 0 means constraint j is satisfied.
g_j(x) > 0 means constraint j is violated.
```

Define an aggregate normalized violation:

```text
phi(x) = max_j(g_j(x) / scale_j)
```

Then:

```text
phi(x) <= 0    feasible
phi(x) > 0     infeasible
```

This scalar `phi` is the main target for feasibility search and surrogate modeling.

### 3.5 Candidate constraints

The exact thresholds should be configurable. The initial prototype should support at least the following constraints.

#### Inventory constraints

```text
g_total_pilots_final = target_total_pilots - final_total_pilots

g_total_pilots_window = target_total_pilots - min_total_pilots_after_assessment_start
```

Use one or both depending on config.

#### Readiness constraints

The model already stores RAP shortfalls by WG/FL/IP. Use those directly.

```text
g_wg_rap = max_wg_rap_shortfall_after_assessment_start - allowed_wg_rap_shortfall

g_fl_rap = max_fl_rap_shortfall_after_assessment_start - allowed_fl_rap_shortfall

g_ip_rap = max_ip_rap_shortfall_after_assessment_start - allowed_ip_rap_shortfall
```

If threshold-based sortie rates are easier to reason about, convert shortfalls to margins, but preserve the sign convention above.

#### Line / staff / experience constraints

Add as configurable constraints where data is available:

```text
g_line_pilots = target_line_pilots - min_line_pilots_after_assessment_start

g_staff_ips = target_staff_ips - min_staff_ips_after_assessment_start

g_staff_fls = target_staff_fls - min_staff_fls_after_assessment_start

g_experience_ratio = min_experience_ratio - min_experience_ratio_after_assessment_start
```

If a metric is not reliably available from `history`, skip it or set `enabled: false` by default.

#### Training bottleneck / deferral constraints

Long-horizon `CAFSimulation.history` may not currently expose all deferral or remaining syllabus fields. If unavailable, add this as a later enhancement.

Potential future constraints:

```text
g_mqt_deferral = max_mqt_deferral - allowed_mqt_deferral

g_flug_deferral = max_flug_deferral - allowed_flug_deferral

g_ipug_deferral = max_ipug_deferral - allowed_ipug_deferral
```

Do not block the initial prototype on deferral constraints if they require deeper instrumentation.

---

## 4. Product Requirements

### 4.1 Primary user story

As an analyst, I want to run a repeatable feasibility study so that I can determine whether the modeled force can reach and maintain a ready state with the required number of pilots under specified policy bounds.

### 4.2 Secondary user stories

As an analyst, I want to:

1. Generate candidate policy designs over a configured design space.
2. Evaluate those designs with the existing model in parallel.
3. Compute pass/fail feasibility and constraint violations.
4. Fit simple surrogate models to the feasibility response.
5. Search the surrogate cheaply for likely feasible policies.
6. Verify surrogate-proposed policies in the original model.
7. Produce plots showing feasible, infeasible, and marginal regions.
8. Identify which constraints are binding for candidate policies.

### 4.3 Non-goals for prototype

Do not attempt the following in the first prototype:

- Do not replace or rewrite the existing manning engine.
- Do not rewrite the existing sortie/sim brain training pipeline.
- Do not train a new RL agent.
- Do not implement a full OpenMDAO or multidisciplinary optimization framework.
- Do not optimize hundreds of per-phase decision variables.
- Do not make a final decision claim based only on a surrogate. Always verify candidate feasible policies in the direct model.

---

## 5. Proposed Package Structure

Create a new package:

```text
src/viability/
    __init__.py
    config.py
    design_space.py
    policy.py
    evaluator.py
    metrics.py
    doe.py
    surrogate.py
    search.py
    active_learning.py
    plots.py
    report.py
    cli.py
```

Add example config:

```text
configs/viability.example.yaml
```

Add tests:

```text
tests/test_viability_metrics.py
tests/test_viability_design_space.py
tests/test_viability_evaluator_smoke.py
```

Output directory:

```text
outputs/viability/
    runs/<timestamp_or_name>/
        config_resolved.yaml
        doe.csv
        evaluations.parquet
        surrogate_metrics.json
        candidate_policies.csv
        verified_candidates.parquet
        envelope_*.png
        report.md
```

If Parquet dependencies are not available in a local environment, fall back to CSV with a warning.

---

## 6. Configuration Schema

Implement a YAML-driven config. Suggested first-pass schema:

```yaml
run:
  name: viability_smoke
  random_seed: 42
  output_dir: outputs/viability
  workers: 4

model:
  years_to_run: 20
  start_year: 2026
  assessment_start_year: 2040
  target_year: 2040
  brain_path: brains/hpc_sortie_brain_multi_output_mlp.pkl
  expected_brain_outputs: 16
  round_robin: true
  use_upgrade_quotas: true
  staff_priority_mode: random
  n_replications: 1

requirements:
  target_total_pilots: 3500
  target_line_pilots: null
  min_experience_ratio: null
  allowed_wg_rap_shortfall: 0.0
  allowed_fl_rap_shortfall: 0.0
  allowed_ip_rap_shortfall: 0.0
  target_staff_ips: null
  target_staff_fls: null

constraint_scales:
  total_pilots: 100.0
  line_pilots: 100.0
  wg_rap: 1.0
  fl_rap: 1.0
  ip_rap: 1.0
  staff_ips: 10.0
  staff_fls: 10.0
  experience_ratio: 0.05

policy:
  parameterization: constant
  variables:
    annual_intake:
      type: int
      low: 10
      high: 350
    retention_rate:
      type: float
      low: 0.10
      high: 0.65
    ute:
      type: float
      low: 6
      high: 20
    paa:
      type: int
      low: 18
      high: 30
    max_manning_pct:
      type: float
      low: 100
      high: 200
    flug_quota_per_phase:
      type: int
      low: 0
      high: 10
    ipug_quota_per_phase:
      type: int
      low: 0
      high: 10

doe:
  method: sobol
  n_initial: 128
  start_index: 0
  scramble: true
  include_corners: true
  include_baselines: true
  include_corners_on_resume: false
  include_baselines_on_resume: false

surrogate:
  enabled: true
  models:
    - ridge
    - gpr
  primary_target: phi
  test_size: 0.2
  cv_folds: 5

search:
  method: differential_evolution
  surrogate_screen_n: 50000
  n_candidates_to_verify: 25
  conservative_sigma: 1.0

plots:
  enabled: true
  slices:
    - x: annual_intake
      y: retention_rate
      fixed:
        ute: 12
        paa: 24
        max_manning_pct: 150
        flug_quota_per_phase: 3
        ipug_quota_per_phase: 2
    - x: paa
      y: ute
      fixed:
        annual_intake: 250
        retention_rate: 0.5
        max_manning_pct: 150
        flug_quota_per_phase: 3
        ipug_quota_per_phase: 2
```

The config parser should validate required fields and provide helpful errors.

For resumable Sobol campaigns, `start_index` and `n_initial` should define the
contiguous Sobol block for the current run. For example, a first run can use
`start_index: 0, n_initial: 128`; a continuation run can use
`start_index: 128, n_initial: 128` with the same `random_seed` and `scramble`
settings. Sobol sample `design_id` values should be stable strings such as
`sobol_000128`. Corner and baseline designs should have separate IDs so they do
not collide with resumed Sobol rows. By default, corners and baselines are only
included for `start_index: 0`; set `include_corners_on_resume` or
`include_baselines_on_resume` to repeat them in continuation batches.

### 6.1 Local brain artifact note

The tracked local artifact at `brains/hpc_sortie_brain_multi_output_mlp.pkl`
may be a stale 12-output model. Do not delete it as part of this prototype and
do not stage a deletion in the PR. Instead, keep `brain_path` configurable and
point viability runs at a current 16-output internal surrogate artifact.

The long-horizon `CAFSimulation` currently expects the 16-output layout:

```text
0-2    WG/FL/IP monthly sortie rates
3-5    WG/FL/IP blue monthly sortie rates
6-9    MQT/WG/FL/IP monthly sim rates
10-15  remaining syllabus outputs
```

If a configured brain produces any other output count, the viability evaluator
should fail clearly before running the long-horizon model. A silent fallback to a
legacy 12-output layout would make the long-horizon results difficult to trust.

For local review without a full HPC sweep, generate a small ignored single-phase
training batch with:

```bash
python tools/generate_local_brain_training_data.py --n 256 --workers 4
python hpc_train_brain_multi_output.py
```

That writes the configured local artifact at
`outputs/single_phase/brains/hpc_sortie_brain_multi_output_mlp.pkl`. Treat this
as a smoke/review brain unless it is trained on a production-sized sweep.

### 6.2 Prototype simplification

The first usable feasibility prototype should prioritize direct long-horizon
evaluations over adding a second surrogate layer. Generate input combinations
over the configured policy bounds, run the long-horizon model directly when a
compatible internal surrogate is available, and produce feasibility tables. Add
the policy-to-feasibility surrogate only if direct evaluations are too slow or a
dense boundary search becomes necessary.

---

## 7. Implementation Phases

## Phase 0 — Repo Inspection and Integration Plan

### Objective

Before coding heavily, inspect the repo locally and confirm how to construct a `CAFSimulation` instance with a specified brain, initial squadrons, and policy settings.

### Tasks

1. Inspect:
   - `src/manning_engine.py`
   - `src/manning_config.py`
   - `src/models.py`
   - `src/manning_gym.py`
   - existing scripts that instantiate `CAFSimulation`
2. Identify the cleanest call path for long-horizon simulation.
3. Confirm where the current brain artifact is expected to live.
4. Avoid hard-coded ambiguity between:
   - `brains/hpc_sortie_brain_multi_output_mlp.pkl`
   - `outputs/single_phase/brains/hpc_sortie_brain_multi_output_mlp.pkl`
5. Make `brain_path` configurable and load with `joblib.load`; pass the loaded brain into `CAFSimulation` if supported.

### Acceptance criteria

- Developer can run a one-off script that instantiates the model and completes a short simulation.
- The implementation notes are captured in comments or `docs/viability_notes.md` if needed.

---

## Phase 1 — Direct Viability Evaluator

### Objective

Build a deterministic wrapper around the existing model that evaluates one policy design and returns metrics, constraint violations, aggregate violation, and feasibility.

### Files

Implement:

```text
src/viability/policy.py
src/viability/evaluator.py
src/viability/metrics.py
src/viability/config.py
```

### Required data classes

Suggested shape:

```python
from dataclasses import dataclass
from typing import Dict, Any

@dataclass(frozen=True)
class PolicyDesign:
    annual_intake: int
    retention_rate: float
    ute: float
    paa: int
    max_manning_pct: float
    flug_quota_per_phase: int
    ipug_quota_per_phase: int

@dataclass
class EvaluationResult:
    design: Dict[str, Any]
    raw_metrics: Dict[str, float]
    constraints: Dict[str, float]
    phi: float
    feasible: bool
    status: str
    error: str | None = None
```

### Evaluator behavior

Function signature:

```python
def evaluate_design(design: PolicyDesign, config: ViabilityConfig, seed: int | None = None) -> EvaluationResult:
    ...
```

Expected behavior:

1. Set `random.seed(seed)` and `np.random.seed(seed)` for repeatability.
2. Load or reuse the configured brain.
3. Instantiate `CAFSimulation` with:
   - `annual_intake`
   - `retention_rate`
   - `round_robin`
   - loaded `brain`
   - `max_manning_pct`
   - `use_upgrade_quotas`
   - staff priority mode
4. Set FLUG/IPUG quotas from the design.
5. Get initial squadrons from `src.manning_config.get_initial_squadrons`.
6. Apply PAA to squadrons if requested by the design.
7. Run `sim.run_simulation(years_to_run, squadron_configs, ute=design.ute)` or the equivalent confirmed call path.
8. Aggregate the returned history by `(year, phase)`.
9. Compute raw metrics.
10. Compute constraints with the configured sign convention.
11. Compute `phi = max(normalized constraints)`.
12. Return `feasible = phi <= 0`.
13. If the model fails for a design, return `status='failed'`, `phi=+inf`, and capture the error string.

### Raw metrics to compute initially

Compute at least:

```text
final_total_pilots
final_line_pilots
final_staff_ips
final_staff_fls
min_total_pilots_after_assessment_start
min_line_pilots_after_assessment_start
min_experience_ratio_after_assessment_start
max_wg_rap_shortfall_after_assessment_start
max_fl_rap_shortfall_after_assessment_start
max_ip_rap_shortfall_after_assessment_start
mean_wg_rap_shortfall_after_assessment_start
mean_fl_rap_shortfall_after_assessment_start
mean_ip_rap_shortfall_after_assessment_start
```

If some fields are not present, skip them cleanly and log a warning. Do not crash unless the field is required by an enabled constraint.

### Tests

Add unit tests for `metrics.py` using synthetic history DataFrames. These tests should not require the full simulator.

Add one smoke test for `evaluate_design` with very short horizon, e.g. 1 year, and skip if the configured brain file is not available.

### Acceptance criteria

- A single design can be evaluated from CLI or a small script.
- Output contains design values, raw metrics, constraints, `phi`, and `feasible`.
- Constraint sign convention is documented and tested.

---

## Phase 2 — Design Space and DoE Generation

### Objective

Generate low-dimensional candidate policy designs for direct model evaluation.

### Files

Implement:

```text
src/viability/design_space.py
src/viability/doe.py
```

### Design-space requirements

1. Read variable names, types, lower bounds, and upper bounds from config.
2. Normalize variables to `[0, 1]` for surrogate training and sampling.
3. Denormalize back to physical units.
4. Round integer variables consistently.
5. Validate bounds.
6. Validate designs before model evaluation.

### DoE methods

Implement at least:

```text
random uniform
Sobol sequence if scipy.stats.qmc is available
Latin hypercube if scipy.stats.qmc is available
manual corner cases
configured baseline points
```

Use `scipy.stats.qmc` if available. If not, fall back to random uniform.

### Parallel evaluation

Add a function:

```python
def evaluate_designs_parallel(designs, config, workers: int) -> pandas.DataFrame:
    ...
```

Implementation notes:

- Use `concurrent.futures.ProcessPoolExecutor`.
- Pass only serializable objects to workers.
- Consider loading the brain per worker and caching it globally to avoid repeated load overhead.
- Use deterministic seeds per design and replication.
- Write partial results periodically so a long batch can be resumed or at least recovered.

### CLI command

Add:

```bash
python -m src.viability.cli run-doe --config configs/viability.example.yaml --n 128 --workers 4
```

Expected outputs:

```text
outputs/viability/runs/<run_name>/doe.csv
outputs/viability/runs/<run_name>/evaluations.parquet or evaluations.csv
outputs/viability/runs/<run_name>/config_resolved.yaml
```

### Acceptance criteria

- Can generate a DoE with at least 10 designs without evaluating the model.
- Can evaluate a small DoE with the model.
- Results file includes design columns, raw metrics, constraint columns, `phi`, `feasible`, `status`, and `error`.

---

## Phase 3 — Baseline Surrogate Models

### Objective

Fit first-pass surrogate models that predict aggregate violation and individual constraint margins from policy variables.

### Files

Implement:

```text
src/viability/surrogate.py
```

### Models to implement

Start with:

1. `Ridge` regression baseline.
2. Quadratic response surface if easy.
3. Gaussian Process Regression using scikit-learn if available.

Suggested GPR kernel:

```python
ConstantKernel() * Matern(nu=2.5, length_scale=[...]) + WhiteKernel()
```

Use ARD length scales when possible. Normalize inputs and standardize outputs.

### Targets

Train surrogates for:

```text
phi
all enabled individual constraints g_j
selected raw metrics if useful
```

The primary search target should be `phi`.

### Metrics

Report more than R². Include:

```text
MAE_phi
RMSE_phi
R2_phi
constraint_sign_accuracy
feasible_class_accuracy
false_feasible_rate
false_infeasible_rate
boundary_MAE_phi where |phi| <= boundary_threshold
```

False-feasible predictions are especially important:

```text
false feasible = surrogate predicts feasible, direct model says infeasible
```

### CLI command

Add:

```bash
python -m src.viability.cli fit-surrogate \
  --config configs/viability.example.yaml \
  --evaluations outputs/viability/runs/<run_name>/evaluations.parquet
```

Expected outputs:

```text
surrogate_phi_gpr.joblib
surrogate_phi_ridge.joblib
surrogate_constraints_*.joblib
surrogate_metrics.json
```

### Acceptance criteria

- Surrogate can be fit from an evaluations file.
- Metrics are written to JSON.
- Model artifacts can be reloaded and used for prediction.
- The surrogate code handles cases with no feasible points without crashing.

---

## Phase 4 — Surrogate Search and Candidate Verification

### Objective

Use the surrogate to find likely feasible policies, likely boundary policies, and minimum-violation policies. Then verify selected candidates with the original model.

### Files

Implement:

```text
src/viability/search.py
```

### Search methods

Implement at least two methods:

1. Large random/Sobol surrogate screening.
2. `scipy.optimize.differential_evolution` on the surrogate prediction of `phi`.

Optional later methods:

```text
CMA-ES
NSGA-II
Bayesian optimization
multi-start local optimization
```

### Candidate categories

Generate candidate policies in several categories:

```text
best predicted phi
predicted feasible with largest margin
near-boundary: abs(predicted_phi) small
uncertain boundary if GPR uncertainty is available
high-diversity candidates across the design space
```

For GPR, support conservative feasibility:

```text
mu_phi + k * sigma_phi <= 0
```

where `k` is `search.conservative_sigma` from config.

### Verification

Evaluate the top `n_candidates_to_verify` candidates with the original model.

Write:

```text
candidate_policies.csv
verified_candidates.parquet or verified_candidates.csv
```

### CLI commands

```bash
python -m src.viability.cli search \
  --config configs/viability.example.yaml \
  --surrogate outputs/viability/runs/<run_name>/surrogate_phi_gpr.joblib

python -m src.viability.cli verify-candidates \
  --config configs/viability.example.yaml \
  --candidates outputs/viability/runs/<run_name>/candidate_policies.csv \
  --workers 4
```

### Acceptance criteria

- Search returns candidate policies even when no feasible policy is predicted.
- Verification marks each candidate as direct-model feasible or infeasible.
- Report identifies best verified policy by `phi`.
- Report identifies binding constraints for best candidates.

---

## Phase 5 — Active Learning Loop

### Objective

Improve surrogate fidelity near the feasible/infeasible boundary without brute-force sampling the entire space.

### Files

Implement:

```text
src/viability/active_learning.py
```

### Loop

One active-learning iteration:

1. Fit surrogate to all evaluated points.
2. Generate many candidate points cheaply.
3. Score candidates using an acquisition rule.
4. Select a batch of new direct-model evaluations.
5. Run those evaluations in parallel.
6. Append to evaluations dataset.
7. Refit surrogate and update metrics.

### Acquisition rules

Implement a simple rule first:

```text
score = -abs(mu_phi) + lambda_uncertainty * sigma_phi
```

If uncertainty is unavailable:

```text
select points with smallest abs(predicted_phi)
plus best predicted feasible points
plus random exploration points
```

Candidate batch should include:

```text
boundary points
predicted feasible points
minimum-violation points
random/exploration points
```

### CLI command

```bash
python -m src.viability.cli active-learn \
  --config configs/viability.example.yaml \
  --evaluations outputs/viability/runs/<run_name>/evaluations.parquet \
  --iterations 3 \
  --batch-size 32 \
  --workers 4
```

### Acceptance criteria

- Each iteration appends new evaluated points.
- Surrogate metrics are updated after each iteration.
- The loop does not select duplicate designs unless replications are explicitly requested.
- Boundary error and false-feasible rate are reported over time.

---

## Phase 6 — Feasible-Envelope Visualization and Reporting

### Objective

Generate stakeholder-friendly plots and a Markdown report that answer the viability question.

### Files

Implement:

```text
src/viability/plots.py
src/viability/report.py
```

### Required plots

Implement 2-D slice plots first.

Examples:

```text
annual_intake vs retention_rate
paa vs ute
annual_intake vs ute
retention_rate vs paa
```

Each plot should show at least:

```text
predicted feasible region
predicted infeasible region
direct evaluated points
verified feasible candidates
verified infeasible candidates
approximate boundary phi = 0
```

Use simple matplotlib. Do not require seaborn.

### Optional projected envelope

For later:

For each point in a 2-D slice, optimize remaining variables on the surrogate:

```text
psi(a, b) = min_z phi(a, b, z)
```

Then mark `(a, b)` feasible if `psi(a, b) <= 0`.

This is more powerful but more complex. Do not block the initial prototype on projected envelopes.

### Markdown report

Generate:

```text
outputs/viability/runs/<run_name>/report.md
```

Report should include:

1. Study config summary.
2. Requirement thresholds.
3. Design variable bounds.
4. Number of direct model evaluations.
5. Number of feasible / infeasible points found.
6. Best verified policy by `phi`.
7. Whether any verified feasible policy was found.
8. Binding constraints for top candidates.
9. Surrogate metrics, especially false-feasible rate.
10. Plots.
11. Caveats and next steps.

### Acceptance criteria

- Running the report command produces a readable Markdown report.
- At least one 2-D envelope plot is generated when surrogate and config are available.
- Report clearly distinguishes predicted feasibility from verified direct-model feasibility.

---

## Phase 7 — Robustness and Validation Enhancements

### Objective

Make the viability answer more credible under stochasticity, surrogate error, and threshold uncertainty.

This phase is not required for the first prototype but should be anticipated in the design.

### Enhancements

1. **Replicated model evaluations**
   - Run each design over multiple seeds.
   - Aggregate with mean, worst case, and percentile violation.

2. **Robust feasibility**
   - Define feasibility using worst-case or high-percentile `phi`.
   - Example: feasible if `p95_phi <= 0`.

3. **Surrogate uncertainty margins**
   - Use `mu_phi + k * sigma_phi <= 0` for conservative feasibility.

4. **Sensitivity analysis**
   - Estimate which levers most affect feasibility.
   - Use GPR length scales, permutation importance, or local finite differences.

5. **Scenario comparison**
   - Current policy.
   - Pragmatic policy.
   - Optimistic policy.
   - High-intake policy.
   - High-retention policy.
   - Combined policy.

6. **Piecewise policy parameterization**
   - Add multi-year blocks after constant-policy prototype works.

---

## 8. CLI Design

Implement one CLI entry point:

```bash
python -m src.viability.cli <command> [args]
```

Commands:

```text
validate-config
run-one
generate-doe
run-doe
fit-surrogate
search
verify-candidates
active-learn
plot-envelope
make-report
run-pipeline
```

### Example commands

Evaluate one design:

```bash
python -m src.viability.cli run-one \
  --config configs/viability.example.yaml \
  --annual-intake 250 \
  --retention-rate 0.50 \
  --ute 12 \
  --paa 24 \
  --max-manning-pct 150 \
  --flug-quota-per-phase 3 \
  --ipug-quota-per-phase 2
```

Run initial DoE:

```bash
python -m src.viability.cli run-doe \
  --config configs/viability.example.yaml \
  --n 128 \
  --workers 4
```

Fit surrogate:

```bash
python -m src.viability.cli fit-surrogate \
  --config configs/viability.example.yaml \
  --evaluations outputs/viability/runs/viability_smoke/evaluations.parquet
```

Search and verify:

```bash
python -m src.viability.cli search \
  --config configs/viability.example.yaml \
  --run-dir outputs/viability/runs/viability_smoke

python -m src.viability.cli verify-candidates \
  --config configs/viability.example.yaml \
  --run-dir outputs/viability/runs/viability_smoke \
  --workers 4
```

Full pipeline:

```bash
python -m src.viability.cli run-pipeline \
  --config configs/viability.example.yaml \
  --n-initial 128 \
  --active-iterations 2 \
  --active-batch-size 32 \
  --workers 4
```

---

## 9. Data Schemas

### 9.1 Evaluations table

Each row is one evaluated design, or one design-replication if replications are enabled.

Required columns:

```text
run_id
design_id
replication_id
seed
status
error

annual_intake
retention_rate
ute
paa
max_manning_pct
flug_quota_per_phase
ipug_quota_per_phase

final_total_pilots
final_line_pilots
final_staff_ips
final_staff_fls
min_total_pilots_after_assessment_start
min_line_pilots_after_assessment_start
min_experience_ratio_after_assessment_start
max_wg_rap_shortfall_after_assessment_start
max_fl_rap_shortfall_after_assessment_start
max_ip_rap_shortfall_after_assessment_start
mean_wg_rap_shortfall_after_assessment_start
mean_fl_rap_shortfall_after_assessment_start
mean_ip_rap_shortfall_after_assessment_start

g_total_pilots_final
g_total_pilots_window
g_line_pilots
g_wg_rap
g_fl_rap
g_ip_rap
g_experience_ratio

phi
feasible
binding_constraint
```

Additional fields are allowed.

### 9.2 Candidate policies table

Required columns:

```text
candidate_id
source
predicted_phi
predicted_sigma_phi
predicted_feasible
conservative_predicted_feasible

annual_intake
retention_rate
ute
paa
max_manning_pct
flug_quota_per_phase
ipug_quota_per_phase
```

After verification, append direct-model result columns:

```text
verified_phi
verified_feasible
verified_binding_constraint
verified_status
verified_error
```

---

## 10. Implementation Notes and Pitfalls

### 10.1 Do not optimize through the RL environment

The RL environment is useful for understanding existing levers and bounds, but the viability prototype should not call `ManningEnv.step()` as the primary evaluator.

Reason: the RL environment encodes discrete increment/hold/decrement actions and reward shaping. The viability analysis needs direct policy designs and explicit constraints.

### 10.2 Keep direct model verification separate from surrogate prediction

Always label outputs clearly:

```text
predicted_feasible
verified_feasible
```

Do not present surrogate-predicted feasibility as a final answer without direct model verification.

### 10.3 Handle stochastic retention

Retention currently includes randomness. At minimum:

- Set seeds for repeatability.
- Add `n_replications` support in config.
- Aggregate replications later if needed.

Prototype can use `n_replications: 1`, but the code structure should support more.

### 10.4 Watch unit conventions

`CAFSimulation` constructor uses `max_manning_pct`, then divides by 100 internally. Elsewhere, `sim.max_manning` may be a ratio. Be explicit in the viability design vector: use `max_manning_pct` as a percent when constructing the simulation.

### 10.5 Apply PAA carefully

Initial squadron configs likely carry their own `paa`. The prototype should apply design-level `paa` consistently across all squadrons unless a more detailed PAA allocation is later added.

Initial rule:

```python
for sq in squadrons:
    sq.paa = int(round(design.paa))
```

Document this assumption in the report.

### 10.6 Integer variables

For GPR and continuous search, integer-like variables can be treated as continuous internally, then rounded before direct model evaluation.

Variables to round:

```text
annual_intake
paa
flug_quota_per_phase
ipug_quota_per_phase
```

Surrogate search may produce fractional values. Store both raw and rounded values if helpful, but direct evaluation should use valid rounded values.

### 10.7 GPR scaling

Normalize design variables to `[0, 1]`. Standardize outputs before fitting GPR. GPR can become slow for large datasets due to cubic scaling. If direct evaluations grow beyond roughly 1,000–2,000 points, consider:

```text
subsampled GP
random forest / gradient boosting
kernel ridge regression
neural network ensemble
BoTorch/GPyTorch later
```

Do not prematurely add heavy dependencies.

### 10.8 Missing brain artifact

If the brain file is missing, the evaluator should fail with a clear message:

```text
Could not find configured brain_path: <path>.
Train or provide the sortie brain before running viability analysis.
```

Do not silently train the brain inside the viability pipeline.

---

## 11. Tests

### Unit tests

Add tests for:

1. Design normalization/denormalization.
2. Integer rounding.
3. Constraint sign convention.
4. Aggregate violation calculation.
5. Binding constraint selection.
6. Handling disabled constraints.
7. Surrogate fit/predict on synthetic data.

### Integration / smoke tests

Add smoke tests that run only if required model artifacts exist:

1. Evaluate one design for 1 year.
2. Generate 4-point DoE and evaluate with 1 worker.
3. Fit a ridge surrogate to synthetic or tiny evaluated data.
4. Run surrogate search on synthetic data.

Use `pytest.mark.skipif` for tests requiring the brain artifact.

---

## 12. Acceptance Criteria for First Prototype

The prototype is acceptable when all of the following are true:

1. A user can run a single design evaluation from CLI.
2. A user can run a small DoE from CLI.
3. The DoE output includes feasibility, `phi`, and binding constraints.
4. A user can fit at least one surrogate from the evaluations table.
5. A user can search the surrogate and produce candidate policies.
6. Candidate policies can be verified in the original model.
7. The report clearly states whether any verified feasible policy was found.
8. The report identifies the best verified policy and its binding constraints.
9. At least one 2-D feasible-envelope plot can be generated.
10. The code does not alter existing RL training or existing manning simulation behavior.

---

## 13. Suggested Development Order for Codex

Implement in this order:

1. `config.py` — load and validate YAML.
2. `policy.py` — `PolicyDesign` dataclass and rounding helpers.
3. `design_space.py` — normalization, denormalization, sampling helpers.
4. `metrics.py` — aggregate history and compute constraints/phi using synthetic tests.
5. `evaluator.py` — wrap `CAFSimulation` and return `EvaluationResult`.
6. `doe.py` — generate DoE and evaluate in serial first.
7. Add parallel evaluation.
8. `cli.py` — expose `run-one`, `generate-doe`, and `run-doe`.
9. `surrogate.py` — fit ridge and GPR, write metrics.
10. `search.py` — surrogate screening and differential evolution.
11. `plots.py` — 2-D slice plotting.
12. `report.py` — Markdown report.
13. Add `run-pipeline` orchestration.

Keep commits small if using version control.

---

## 14. Minimal Pseudocode

### 14.1 Constraint and violation calculation

```python
def compute_constraints(metrics, cfg):
    req = cfg.requirements
    g = {}

    if req.target_total_pilots is not None:
        g["g_total_pilots_final"] = (
            req.target_total_pilots - metrics["final_total_pilots"]
        )
        g["g_total_pilots_window"] = (
            req.target_total_pilots - metrics["min_total_pilots_after_assessment_start"]
        )

    if req.allowed_wg_rap_shortfall is not None:
        g["g_wg_rap"] = (
            metrics["max_wg_rap_shortfall_after_assessment_start"]
            - req.allowed_wg_rap_shortfall
        )

    if req.allowed_fl_rap_shortfall is not None:
        g["g_fl_rap"] = (
            metrics["max_fl_rap_shortfall_after_assessment_start"]
            - req.allowed_fl_rap_shortfall
        )

    if req.allowed_ip_rap_shortfall is not None:
        g["g_ip_rap"] = (
            metrics["max_ip_rap_shortfall_after_assessment_start"]
            - req.allowed_ip_rap_shortfall
        )

    return g


def aggregate_violation(constraints, scales):
    normalized = {}
    for name, value in constraints.items():
        scale = scales.get(name.replace("g_", ""), 1.0)
        normalized[name] = value / scale
    phi = max(normalized.values()) if normalized else float("nan")
    binding = max(normalized, key=normalized.get) if normalized else None
    feasible = phi <= 0.0
    return phi, feasible, binding, normalized
```

### 14.2 Evaluator skeleton

```python
def evaluate_design(design, cfg, seed=None):
    try:
        set_random_seeds(seed)
        design = design.rounded()

        brain = load_brain(cfg.model.brain_path)
        sim = CAFSimulation(
            annual_intake=design.annual_intake,
            retention_rate=design.retention_rate,
            round_robin=cfg.model.round_robin,
            brain=brain,
            max_manning_pct=design.max_manning_pct,
            staff_priority_mode=parse_priority(cfg.model.staff_priority_mode),
            use_upgrade_quotas=cfg.model.use_upgrade_quotas,
        )

        sim.sq_phase_flug_intake = design.flug_quota_per_phase
        sim.sq_phase_ipug_intake = design.ipug_quota_per_phase

        squadrons = get_initial_squadrons(sim.current_year)
        for sq in squadrons:
            sq.paa = design.paa

        history = sim.run_simulation(
            years_to_run=cfg.model.years_to_run,
            squadron_configs=squadrons,
            ute=design.ute,
        )

        metrics = compute_raw_metrics(history, cfg)
        constraints = compute_constraints(metrics, cfg)
        phi, feasible, binding, normalized = aggregate_violation(
            constraints, cfg.constraint_scales
        )

        return EvaluationResult(
            design=asdict(design),
            raw_metrics=metrics,
            constraints=constraints,
            phi=phi,
            feasible=feasible,
            status="ok",
            error=None,
        )

    except Exception as exc:
        return EvaluationResult(
            design=asdict(design),
            raw_metrics={},
            constraints={},
            phi=float("inf"),
            feasible=False,
            status="failed",
            error=str(exc),
        )
```

### 14.3 Surrogate search logic

```python
def search_surrogate(model, design_space, cfg):
    # 1. Generate many cheap candidate points.
    X = design_space.sample(cfg.search.surrogate_screen_n, method="sobol")

    # 2. Predict phi.
    mu, sigma = model.predict_phi(X, return_std=True)

    # 3. Rank by categories.
    best_phi_idx = np.argsort(mu)[:cfg.search.n_candidates_to_verify]
    boundary_idx = np.argsort(np.abs(mu))[:cfg.search.n_candidates_to_verify]

    conservative = mu + cfg.search.conservative_sigma * sigma
    feasible_idx = np.where(conservative <= 0)[0]
    feasible_idx = feasible_idx[np.argsort(conservative[feasible_idx])]

    # 4. Merge, dedupe, denormalize, round.
    selected = merge_and_dedupe([best_phi_idx, boundary_idx, feasible_idx])
    return design_space.to_designs(X[selected])
```

---

## 15. Final Output Expectations

A successful prototype should let the analyst say something like:

```text
Under the configured policy bounds and model assumptions, we found verified feasible policies / did not find verified feasible policies.

The best verified policy has phi = <value>.

The binding constraint is <constraint>.

The feasible envelope suggests that feasibility requires approximately:
    annual_intake >= <value>
    retention_rate >= <value>
    UTE/PAA combination within <region>

Surrogate accuracy near the feasibility boundary is <summary>, and candidate feasible policies were verified with the direct model.
```

If no feasible policy is found, the report should still be useful:

```text
No verified feasible policy was found under the tested bounds.
The closest policy misses by:
    <pilot shortfall>
    <readiness shortfall>
    <staffing shortfall>
The binding bottleneck appears to be <constraint>.
Recommended next analyses are:
    extend policy bounds,
    relax timeline,
    add resource lever,
    improve training throughput,
    run active learning near the best candidate region.
```

---

## 16. Summary Guidance to Codex Agent

Build the prototype as a new viability-analysis layer. Wrap the existing model. Do not rewrite it. Define feasibility using explicit constraints. Use `phi = max(normalized constraint violations)` as the central scalar. Generate DoE samples, evaluate them in parallel, train simple surrogates, search the surrogate, verify candidate policies in the direct model, and produce a report and plots. Start simple: constant policy, small DoE, ridge/GPR surrogate, surrogate screening, and direct verification.

The code should make it easy to answer:

> Is there any region of allowed policy space where the model produces both a ready force and the required number of pilots?
