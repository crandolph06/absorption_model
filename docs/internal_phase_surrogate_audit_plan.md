# Internal Phase Backend Audit And Surrogate Plan

## Summary

Before relying on the viability surrogate and search workflow, choose and verify
the phase backend used inside the long-horizon manning model. There are now two
different "direct" concepts in the repo:

1. **Direct long-horizon viability evaluation:** bypasses the outer signed-RAP
   viability surrogate and runs `CAFSimulation` for the selected policy.
2. **Direct single-phase physics:** bypasses the internal sortie brain and calls
   `run_phase_simulation()` inside each simulated phase through
   `CAFSimulation(use_physics_allocator=True, brain=None)`.

The Phase 7 dashboard's "Run direct verification" button currently means the
first concept only. It still uses the configured internal phase backend. The
next development slice should wire the second concept into the viability
evaluator/config so near-boundary policies can be checked without relying on the
pretrained MLP.

The current viability workflow still uses the configured 16-output MLP at:

```text
outputs/single_phase/brains/hpc_sortie_brain_multi_output_mlp.pkl
```

That artifact matches the current long-horizon model interface:

```text
9 engineered inputs -> 16 sortie/sim/deferral outputs
```

The repository also contains an older legacy artifact at:

```text
brains/hpc_sortie_brain_multi_output_mlp.pkl
```

That artifact returns only 12 outputs and is not compatible with the current
long-horizon engine, which indexes sim outputs at columns 6-9 and deferral
outputs at columns 10-15. The audit should make this distinction explicit and
prevent accidental use of the stale 12-output artifact.

The repository also now contains a direct-physics policy-search path:

```text
optimize_constant_policy.py
src/manning_objective.py
src/manning_engine.py::CAFSimulation(use_physics_allocator=True)
src/simulation_config.py::SimulationConfig
```

The revised goal of this plan is to answer:

> Can direct single-phase physics be used as the authoritative viability backend,
> and, if so, how different are the MLP-backed feasibility conclusions?

If the direct-physics path is too slow for broad search, keep the MLP for fast
screening, but require physics-backed checks for final candidate verification and
near-boundary claims. Retrain the MLP only if we still need a fast brain-backed
backend after measuring the physics-backed response.

## Current Repo Status After Phase 7 And Main Merge

Implemented:

- `viability_dashboard.py` provides live slider exploration using the outer
  signed-RAP surrogate and a button for long-horizon verification.
- `src/viability/dashboard.py` has reusable slider interval, candidate loading,
  surrogate scoring, and trajectory aggregation helpers.
- `src/viability/evaluator.py` still loads `config.model.brain_path` for every
  viability evaluation and validates `expected_brain_outputs`.
- `src/viability/config.py` currently requires
  `model.expected_brain_outputs >= 16`; there is no phase-backend switch yet.
- `configs/viability.example.yaml` points at the 16-output artifact under
  `outputs/single_phase/brains/`.
- `CAFSimulation` can run either the brain path or the direct physics allocator.
- `optimize_constant_policy.py` already demonstrates the direct-physics path for
  constant-policy optimization outside the viability workflow.
- `SimulationConfig` carries phase length, allocation noise, and
  `upgrade_sortie_fraction`, which should be exposed deliberately when the
  viability evaluator uses the physics backend.

Not implemented yet:

- A viability config field such as `model.phase_backend: brain | physics`.
- A physics-backed branch in `src/viability/evaluator.py`.
- Dashboard copy/state that distinguishes brain-backed direct verification from
  physics-backed direct verification.
- Side-by-side comparison outputs for brain-backed and physics-backed
  long-horizon runs.

## What The Internal Phase Surrogate Does

The long-horizon manning simulation calls the internal phase surrogate inside
each simulated phase. It does not directly predict feasibility. It predicts local
squadron phase response quantities from engineered squadron features.

Current feature columns:

```text
paa
ute
exp_ratio
ip_ratio
fl_congestion
wg_crowding
sorties_avail
pilot_to_sortie
ip_to_stud_ratio
```

Current expected target columns:

```text
wg_monthly
fl_monthly
ip_monthly
wg_blue_monthly
fl_blue_monthly
ip_blue_monthly
mqt_sim_monthly
wg_sim_monthly
fl_sim_monthly
ip_sim_monthly
remaining_mqt_syllabi_mean
remaining_flug_syllabi_mean
remaining_ipug_syllabi_mean
remaining_mqt_syllabi_sorties_only_mean
remaining_flug_syllabi_sorties_only_mean
remaining_ipug_syllabi_sorties_only_mean
```

The outer viability surrogate is downstream of this model. If the internal phase
surrogate is biased, the long-horizon evaluator and the viability surrogate can
look internally consistent while still being wrong relative to the direct
single-phase model.

## Phase 1: Interface And Provenance Check

Purpose: make sure we know exactly which artifact is being used and what data
trained it.

Tasks:

1. Add or run a read-only inspection command that reports, for each local
   internal surrogate artifact:
   - path
   - model type
   - input feature count
   - output count
   - expected feature names, if available
   - modification time and file size
   - pass/fail against the 16-output validator
2. Confirm `configs/viability.example.yaml` points to the 16-output artifact.
3. Search for any direct `CAFSimulation(...)` construction that might rely on
   the legacy default path under `brains/`.
4. Record available training-data inputs:
   - `outputs/single_phase/repart_parquet/part.*.parquet`
   - `outputs/single_phase/parquet/batch_low_*.parquet`
   - any fixed holdout or validation parquet files
5. Write a short provenance note in the audit output directory explaining which
   artifact was evaluated.

Deliverable:

```text
outputs/single_phase/audit/interface_report.json
outputs/single_phase/audit/artifact_report.csv
```

Acceptance criteria:

- The configured artifact passes the 16-output interface check.
- Any 12-output artifact is clearly reported as incompatible.
- The audit never silently falls back to `brains/`.

## Phase 2: Wire Direct Single-Phase Physics Into Viability

Purpose: make the viability evaluator able to run the long-horizon model without
the internal MLP, using the already-merged direct physics allocator.

Target contract:

```yaml
model:
  phase_backend: brain   # current default, uses brain_path
  # or
  phase_backend: physics # uses run_phase_simulation inside each phase
  simulation:
    phase_length_days: 120
    allocation_noise: 0.0
    upgrade_sortie_fraction: null
```

Tasks:

1. Add an explicit `model.phase_backend` config field with accepted values
   `brain` and `physics`.
2. Preserve the current `brain` behavior as the default for existing configs.
3. Add a `physics` branch in `src/viability/evaluator.py`:
   - do not load `brain_path`
   - instantiate `CAFSimulation(..., brain=None, use_physics_allocator=True)`
   - pass an explicit `SimulationConfig`
   - keep the same policy levers, requirements, constraint scales, and output
     row schema used by the current evaluator
4. Record the backend in every evaluation row and dashboard direct result.
5. Update validation so `expected_brain_outputs >= 16` is required only for
   `phase_backend: brain`.
6. Add focused tests:
   - physics backend does not require a brain artifact
   - brain backend still rejects stale 12-output or missing artifacts
   - both backends produce the required trajectory and constraint columns
   - `SimulationConfig` values are propagated to `CAFSimulation`
7. Update dashboard labels:
   - "Direct long-horizon verification (brain backend)" when using the MLP
   - "Direct long-horizon verification (physics backend)" when using direct
     single-phase physics
   - never imply that a brain-backed direct run is direct single-phase truth

Deliverable:

```text
src/viability/evaluator.py
src/viability/config.py
configs/viability.example.yaml
tests/test_viability_evaluator_smoke.py
tests/test_viability_config.py
viability_dashboard.py
```

Acceptance criteria:

- A small physics-backed viability evaluation runs without any MLP artifact.
- Existing brain-backed viability configs keep working.
- Evaluation artifacts identify the phase backend used.
- The dashboard cannot confuse outer-surrogate bypass with internal-MLP bypass.

## Phase 3: Brain-vs-Physics Held-Out Accuracy Audit

Purpose: score the configured 16-output MLP against direct single-phase truth so
we know whether it is suitable for fast screening or only for historical
comparison.

Preferred data source:

Use an existing held-out single-phase parquet dataset if one exists and has all
16 target columns.

Fallback:

Generate a new fixed Sobol holdout with the direct single-phase sweeper. This
holdout should not be used for retraining.

Recommended holdout:

```text
n = 2048
method = sobol
scramble = true
fixed seed
explicit start index outside existing training blocks
```

Metrics:

Report per-output and grouped metrics:

```text
MAE
RMSE
R2
bias = mean(predicted - truth)
max absolute error
```

Groups:

```text
sortie rates
blue sortie rates
sim rates
deferral / remaining syllabus outputs
```

Important slices:

```text
low exp_ratio
high wg_crowding
high fl_congestion
low ip_to_stud_ratio
near-zero or high deferral quantities
boundary-like cases that are likely to affect RAP feasibility
```

Plots:

```text
prediction_vs_truth_by_output.png
residuals_by_output.png
error_by_exp_ratio.png
error_by_wg_crowding.png
error_by_fl_congestion.png
```

Deliverable:

```text
outputs/single_phase/audit/mlp_holdout_metrics.csv
outputs/single_phase/audit/mlp_slice_metrics.csv
outputs/single_phase/audit/*.png
```

Acceptance criteria:

- The model is scored output-by-output, not only with one averaged R2.
- Outputs that drive RAP and training progression are inspected directly.
- The audit explicitly identifies whether the current MLP is acceptable,
  questionable, or unacceptable for viability work.

## Phase 4: Long-Horizon Backend Sensitivity Check

Purpose: determine whether MLP-vs-physics backend differences matter for the
outer feasibility conclusions.

Tasks:

1. Select a small set of long-horizon policies:
   - known infeasible policies
   - near-boundary policies
   - best surrogate-screened candidate policies
   - any verified feasible candidates, if available
2. Run each policy with the brain backend and the physics backend.
3. Compare raw metrics, normalized constraints, `phi`, active constraint, and
   feasible/infeasible status.
4. For brain-backed runs, inspect the phase-level feature rows generated during
   the long-horizon rollout.
5. Compare those feature rows to the single-phase training/holdout feature
   distribution.
6. Flag extrapolation or low-density regions.
7. Summarize which internal outputs dominate downstream feasibility outcomes.

Deliverable:

```text
outputs/single_phase/audit/long_horizon_feature_coverage.csv
outputs/single_phase/audit/long_horizon_sensitivity_summary.md
outputs/viability/internal_backend_comparison/comparison_metrics.csv
outputs/viability/internal_backend_comparison/summary.md
```

Acceptance criteria:

- We know whether long-horizon feasibility search is operating inside the
  single-phase surrogate's training domain when the brain backend is used.
- We know whether near-boundary feasibility conclusions are stable when rerun
  with the physics backend.
- Dashboard and report caveats label the relevant backend risk.

## Phase 5: Retrain If Needed

Trigger this phase only if the MLP remains necessary for fast screening and the
audit shows unacceptable errors, unstable boundary behavior, or poor coverage in
the regions used by long-horizon feasibility search.

Training data plan:

1. Preserve the fixed holdout from Phase 3.
2. Build a new Sobol training set from direct single-phase evaluations.
3. Oversample or actively sample regimes that matter downstream:
   - low `exp_ratio`
   - high `wg_crowding`
   - high `fl_congestion`
   - low `ip_to_stud_ratio`
   - large deferral / remaining syllabus cases
   - long-horizon feature rows outside the current training distribution
4. Train a replacement 16-output model.
5. Compare old and new models on the exact same fixed holdout.

Candidate model families:

Primary replacement path:

```text
Gaussian process regression with ARD length scales
matched Sobol train/holdout splits
kernel sweep over Matern nu = 0.5, 1.5, 2.5 and RBF
shared, grouped, and optional per-target output groupings
active-learning batches selected from a larger Sobol candidate pool
```

Fallback comparators if exact GPR becomes too slow or unstable:

```text
per-output gradient boosted trees or random forests
current multi-output MLP architecture as a baseline only
```

Do not assume the MLP is the best production choice. The replacement should be
chosen based on holdout performance, stability, and interpretability.

Deliverable:

```text
outputs/single_phase/retrain/surrogate.joblib
outputs/single_phase/retrain/model_comparison_metrics.csv
outputs/single_phase/retrain/model_comparison_plots/
```

Acceptance criteria:

- Replacement model returns exactly 16 outputs in the current schema.
- Replacement model is evaluated against the same fixed holdout as the current
  MLP.
- Replacement improves the outputs and slices that matter for viability.
- The viability config can point to the replacement artifact without changing
  long-horizon model code.

Current execution status on 2026-06-14:

- `scripts/audit_single_phase_surrogate.py` showed the configured 16-output MLP
  is not acceptable for policy claims: all-target R2 was about 0.595 and the
  sortie-rate group was about 0.244 on the Sobol audit.
- A 1024 train / 1024 holdout kernel sweep in
  `outputs/single_phase/kernel_sweep/n1024_20260614/` compared shared and
  grouped ARD GPRs with Matern `nu = 0.5, 1.5, 2.5` and RBF kernels.
- The best sweep result by threshold-gap ranking was
  `shared_ard + matern_nu2p5_ard`; it did not pass quality gates at 1024
  training rows, but it was the best candidate for larger active learning.
- `scripts/active_learn_single_phase_surrogate.py` then ran a 2048-point fixed
  holdout, 2048 initial Sobol training points, and four 512-point active
  batches using the selected kernel with fixed optimized hyperparameters.
- The final 4096-row candidate passes the current group gates:

```text
artifact:
  outputs/single_phase/active_learning/shared_matern_nu2p5_fixed_n4096_20260614/final/single_phase_gpr_bundle.joblib
holdout:
  outputs/single_phase/active_learning/shared_matern_nu2p5_fixed_n4096_20260614/holdout_truth.csv
metrics:
  all_targets_r2       = 0.9327
  sortie_rates_r2      = 0.8809
  blue_sortie_rates_r2 = 0.9067
  sim_rates_r2         = 0.9491
  deferrals_r2         = 0.8440
```

Recommendation:

- Treat the 4096-row shared ARD Matern `nu=2.5` GPR as the current fast
  single-phase surrogate candidate.
- Keep direct single-phase physics as the source of truth for final
  long-horizon policy claims.
- Do not claim every individual sparse output is solved. Some near-zero MQT
  sim/deferral targets still have weak individual R2, so downstream validation
  should focus on whether those sparse-output errors change policy feasibility.

Current branch closeout status on 2026-06-15:

- The viability workflow has been reframed as a finite-horizon nonlinear
  optimal-control search for dynamic policies:
  - state `x_k`: compressed force/training state represented by simulator
    history at phase `k`
  - control `u_k`: the seven policy levers applied over each open-loop epoch
  - dynamics: `x_{k+1} = f_k(x_k, u_k)` through the physics-backed simulator
  - objective: minimize maximum normalized constraint violation `phi`
- The implemented dynamic-policy class is intentionally conservative:
  three piecewise-constant epochs over the horizon, not a fully free 70-knob
  per-phase schedule.
- A direct-physics 3-epoch search has been run from fresh artifacts under
  ignored `outputs/viability/dynamic_policy_search/`:

```text
run: run_3epoch_512_32768_096
direct evaluations: 615
feasible evaluations: 0
best phi: 6.001732739632043
best schedule id: heuristic_0002
best active constraint: wg_rap
positive constraints at best:
  total_pilots_window = 456.0
  wg_rap = 6.001732739632043
  fl_rap = 2.8692818292818294
```

- The local finite-difference diagnostic around the best schedule has also
  been run. It indicates retention has the strongest local authority over the
  binding WG RAP violation, while total-pilot-window and FL RAP improvement are
  in tension near the best current trajectory.
- No feasible dynamic policy has been found yet. That is evidence against the
  current requirement set and current 3-epoch policy class, not a proof that no
  feasible policy exists in every possible dynamic-policy class.

## Branch Closeout Checklist

Use this checklist to keep the current PR-prep work honest. Update each item as
work is completed.

1. [x] Reframe dynamic-policy work as finite-horizon nonlinear optimal control
   in code/report language.
2. [x] Run a direct-physics 3-epoch dynamic search and local
   finite-difference diagnostic from fresh generated outputs.
3. [x] Add a real evaluator regression proving a constant 3-epoch schedule
   matches the legacy constant-policy evaluator when all epoch controls are the
   same.
4. [x] Clean the dynamic-search heuristic seed implementation so named
   schedules are readable and intentionally experimental.
5. [x] Update the dashboard so it can open a dynamic-search result bundle and
   explain nearest misses when no feasible verified static candidates exist.
6. [x] Keep the old constant-policy slider dashboard clearly labeled as static
   surrogate guidance, not direct feasibility truth.
7. [x] Update this plan with the final status, validation commands, and any
   remaining work before pushing.
8. [x] Commit the branch locally after validation. Do not push from this
   checklist unless explicitly requested.

Closeout implementation notes:

- `tests/test_viability_dynamic_policy.py` now includes a real direct-physics
  regression: an identical 3-epoch schedule must match the constant-policy
  evaluator for `phi`, active constraint, constraints, and key raw metrics.
- Dynamic heuristic seeds are now named templates in
  `src/viability/dynamic_search.py`. New artifact rows can preserve
  `template_name`; old generated runs without that column remain readable.
- `viability_dashboard.py` now defaults to a dynamic-search result mode. It
  loads `dynamic_search_summary.json`, `all_evaluations.parquet`, and optional
  local sensitivity/report artifacts; surfaces the nearest miss and positive
  constraint relaxations first; and keeps the static signed-surrogate slider
  dashboard as a secondary mode.
- This dashboard cleanup is a practical PR-ready improvement, not a final
  front-end redesign. If the branch scope includes a polished dashboard, do a
  separate design pass with a concrete visual target after the result workflow
  is accepted.

Remaining PR-prep recommendations:

1. Run the final focused test suite and compile checks before committing.
2. Commit local changes only; do not push until the collaborator merge strategy
   and PR scope are decided.
3. Do not claim feasibility. Current direct-physics evidence says no feasible
   3-epoch dynamic policy was found in the executed search.
4. Do not claim infeasibility in the mathematical sense. The current evidence
   covers the sampled/search-optimized 3-epoch open-loop policy class, not every
   possible per-phase or feedback policy.
5. Next analysis after this branch should be either targeted dynamic refinement
   near the best miss or a requirement-relaxation study for WG RAP, FL RAP, and
   total-pilot-window constraints.

Closeout validation on 2026-06-15:

```bash
venv/bin/python -m unittest \
  tests.test_viability_dynamic_policy \
  tests.test_viability_dynamic_search \
  tests.test_viability_dashboard \
  tests.test_viability_cli \
  tests.test_viability_config \
  tests.test_viability_doe \
  tests.test_viability_evaluator_smoke \
  tests.test_viability_search \
  tests.test_viability_surrogate \
  tests.test_viability_plots_report

venv/bin/python -m compileall src/viability viability_dashboard.py
git diff --check
```

Result:

- 69 viability tests passed.
- Compile check passed.
- Whitespace check passed.
- Streamlit smoke test loaded the dynamic dashboard on
  `http://localhost:8502` against the current direct-physics artifact bundle
  and reported no browser console errors. The local server was stopped after the
  check.

## Near-Term Execution Checklist

1. Wire `model.phase_backend: brain | physics` into the viability evaluator and
   dashboard so "direct verification" can mean true direct single-phase physics,
   not just bypassing the outer viability surrogate. Completed locally on
   2026-06-14.
2. Find feasible policies under the selected authoritative backend so the next
   comparison uses policy points that matter operationally, not just inherited
   static-policy candidates. A 615-evaluation direct-physics 3-epoch dynamic
   search has found no feasible point so far; continue with targeted dynamic
   refinement or requirement-relaxation analysis rather than returning to stale
   MLP-backed outputs.
3. Add a small brain-vs-physics long-horizon comparison on known feasible,
   infeasible, and near-boundary policies. Use this to determine whether the
   MLP or GPR backend changes feasibility labels.
4. Add a loader/adapter for the new GPR bundle if it should replace the current
   `.predict()` MLP brain for fast screening.
5. Relax the long-term policy representation so the single-phase backend remains
   a static policy over one phase, but long-horizon viability can represent
   dynamic policies across phases or years. Completed locally for 3-epoch
   open-loop schedules; fully free per-phase controls remain intentionally
   deferred.
6. Clean up the dashboard UX after backend status and source-of-truth labels are
   first-class data in the app. Current closeout scope is dynamic-result support
   and clearer information hierarchy, not a full design-system rewrite.
7. Do a final review pass and only then push/open the PR.

## Phase 6: Viability Recheck With The Selected Phase Backend

Purpose: quantify whether changing the phase backend changes the outer
feasibility story.

Tasks:

1. Run a small fixed set of long-horizon policy evaluations with:
   - brain backend with the current configured MLP
   - direct physics backend
   - replacement internal surrogate, if trained and still needed
2. Compare raw metrics, constraint margins, `phi`, feasibility labels, and active
   constraints.
3. Refit or reload the outer signed-RAP viability surrogate only after selecting
   the backend used for broad search.
4. Re-run candidate search and direct verification with the selected internal
   backend.

Deliverable:

```text
outputs/viability/internal_backend_comparison/comparison_metrics.csv
outputs/viability/internal_backend_comparison/summary.md
```

Acceptance criteria:

- We can state whether the feasibility envelope is robust to the phase backend
  choice.
- If the brain and physics backends disagree near the feasibility boundary,
  physics-backed validation becomes a required caveat for any claimed feasible
  region.

## Guardrails

- Do not use the 12-output legacy artifact for viability evaluation.
- Do not call a brain-backed long-horizon run "direct single-phase" or "physics
  verified"; it is only direct with respect to the outer viability surrogate.
- Do not make the dashboard's live signed-RAP surrogate status look equivalent
  to direct long-horizon verification.
- Do not overwrite existing surrogate artifacts. Write replacements to a new
  short, explicit directory.
- Keep the fixed holdout read-only once created.
- Keep generated audit outputs out of git unless explicitly requested.
- Avoid hidden defaults in any new audit/retraining commands.
- Keep `phase_backend`, `brain_path`, and `SimulationConfig` values explicit in
  resolved configs and evaluation metadata.
- Prefer short artifact directories:

```text
outputs/single_phase/audit
outputs/single_phase/retrain
outputs/viability/internal_backend_comparison
```

## Suggested Implementation Order

1. Add `model.phase_backend` and `model.simulation` config support.
2. Wire `phase_backend: physics` into `src/viability/evaluator.py`.
3. Update dashboard labels and direct-result metadata so backend state is
   visible.
4. Add tests proving physics-backed viability evaluation runs without a brain
   artifact and brain-backed evaluation still validates the 16-output contract.
5. Run a tiny fixed policy set through both backends and compare outputs.
6. Add a read-only audit command or script for MLP artifact
   interface/provenance.
7. Add a holdout scoring command for the configured 16-output MLP.
8. Generate or locate a fixed single-phase holdout.
9. Produce metrics and plots for the current MLP.
10. Decide whether the MLP is useful for fast screening or should be retired from
    policy claims.
11. If needed, train replacement candidates and compare on the fixed holdout.
12. Point viability config at the selected backend/artifact.
13. Re-run the outer viability surrogate/search verification path.
14. Perform a cleanup pass:
   - simplify names
   - delete duplicate helper paths
   - remove unused knobs
   - make stale-artifact risk explicit
   - keep the final docs and code readable

## Decision Point

At the end of Phase 2, make a deliberate call:

```text
Physics backend is usable:
    Treat physics-backed long-horizon evaluation as the authoritative direct
    verification path. Keep the MLP only for fast screening if later audits
    justify it.

Physics backend is not yet usable:
    Keep the 16-output MLP-backed path temporarily, but do not claim direct
    single-phase validation until the physics blocker is resolved.
```

At the end of Phase 3, make a second deliberate call:

```text
Current 16-output MLP is good enough for screening:
    Use it only as a fast guide, with physics-backed direct checks for final
    candidates and near-boundary claims.

Current 16-output MLP is not good enough:
    Do not retrain by default. First decide whether the physics backend can
    simply replace the brain for policy claims. Retrain only if runtime makes a
    fast surrogate necessary.
```
