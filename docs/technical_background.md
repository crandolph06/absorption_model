# Technical Background: Feasible-Envelope / Viability Analysis for the Absorption Model

**Repository:** `crandolph06/absorption_model`  
**Companion document:** `viability_analysis_prd_for_codex.md`  
**Intended consumer:** Codex agent or developer implementing a prototype in a local clone of the repository  
**Purpose:** Preserve the key technical framing, modeling assumptions, and search/surrogate rationale discussed in chat.

---

## 1. Executive Summary

The end-state deliverable is not primarily “train an optimizer” or “replace RL with a fancier algorithm.” The end-state deliverable is an existence and viability question:

> Given this model, the initial force state, and allowable policy/resource levers, is it possible to produce a ready force with the required number of pilots?

The recommended technical framing is:

```text
surrogate-assisted feasible-envelope / viability analysis
```

This means:

1. Define readiness, inventory, staffing, training, and policy requirements as explicit constraints.
2. Evaluate policy designs using the existing model.
3. Compute a scalar aggregate constraint violation, `phi(x)`.
4. Learn a surrogate of `phi(x)` and important individual constraint margins.
5. Use search/optimization to map where `phi(x) <= 0`.
6. Verify candidate feasible policies with the original model.
7. Report the feasible envelope, active constraints, margins, and minimum required policy changes.

The key distinction is:

```text
Traditional optimization:
    Find the best point.

Viability / feasible-envelope analysis:
    Find whether feasible points exist, where they exist, what boundary separates
    feasible from infeasible regions, and which constraints bind.
```

Search/optimization is still required. The difference is that optimization is a tool for boundary discovery and feasibility verification, not the headline deliverable.

---

## 2. Non-Pilot Mental Model

For implementation and communication, treat the model as an inventory-and-throughput system rather than as a pilot-specific system.

Generic view:

```text
new people enter -> they train -> they become qualified -> they fill operational jobs
               -> some move to staff roles -> some retain or separate
```

Decision-makers can change policy/resource levers such as intake, retention, training capacity, resource availability, and manning limits.

The model reports whether the system has enough people, whether they are sufficiently ready, whether the training pipeline is bottlenecked, and whether staffing requirements can be met.

Use this translation table when writing code comments, documentation, or visualization labels:

| Technical / Optimization Term | Model-Specific Version | Generic Meaning |
|---|---|---|
| Design variable | intake, UTE, PAA, retention, max manning, upgrade quotas | Knob a decision-maker can change |
| State variable | total pilots, line pilots, staff pilots, WG/FL/IP counts, experience ratio | Current inventory / system condition |
| Output | RAP shortfall, sim rates, deferrals, pilot counts | Performance measure |
| Constraint | minimum pilot count, readiness threshold, max backlog, resource bounds | Requirement |
| Objective | policy burden, readiness margin, violation minimization | Criterion for preferring one solution over another |
| Feasible point | all constraints satisfied | Acceptable policy design |
| Infeasible point | at least one constraint violated | Unacceptable policy design |
| Feasible envelope | set of all feasible points | Region of acceptable policies |
| Active constraint | constraint close to violation at solution | Bottleneck / limiting requirement |

The Codex implementation should preserve this generic structure. Pilot-specific names can remain in code where they correspond to existing repository variables, but the viability layer should expose generic feasibility concepts.

---

## 3. Existing Repository Concepts

The repository already contains most of the technical ingredients needed for a viability prototype.

### 3.1 Model layers

The repo can be viewed as three layers:

```text
Layer 1: High-fidelity / direct phase-level model
    Generates detailed sortie, training, RAP, sim, and deferral behavior.

Layer 2: Long-horizon manning simulation
    Rolls the force forward over phases/years, using an internal surrogate brain
    to estimate sortie/sim/training rates.

Layer 3: RL search layer
    Uses a Gym environment and reward shaping to search policy actions.
```

The proposed prototype should not remove Layer 1 or Layer 2. It should replace or bypass Layer 3 with feasibility analysis and surrogate-assisted search.

Relevant files to inspect in the local clone:

```text
src/manning_engine.py
    CAFSimulation and long-horizon model logic.

src/manning_gym.py
    RL environment, current policy levers, reward shaping, and useful bounds.

hpc_train_brain_multi_output.py
    Existing internal surrogate training path.

hpc_sweepers/single_phase/hpc_single_phase_sweeper.py
    Existing single-phase sweep / DoE-like pattern.

src/models.py
    Entities, thresholds, RAP target values, pilot/squadron state.

src/rap_state.py
    RAP assessment helpers.

src/manning_config.py
    Initial squadron configuration helpers.
```

### 3.2 Two different surrogate concepts

There are two surrogate layers that must not be confused.

#### Existing internal surrogate

The current model already uses a surrogate, often referred to as the “brain.” It predicts lower-level response quantities such as sortie rates, sim rates, and syllabus/deferral values from local squadron features.

This surrogate is inside the existing model stack.

#### Proposed viability surrogate

The new prototype should add a higher-level surrogate:

```text
policy vector x -> end-to-end feasibility metrics
```

This surrogate predicts quantities such as:

```text
aggregate violation phi(x)
individual constraint margins g_j(x)
probability or classification of feasibility
candidate margins / bottleneck indicators
```

The new viability surrogate should reuse the existing model stack as its evaluator. Do not initially replace the internal sortie/sim brain.

---

## 4. Core Mathematical Formulation

Let `x` be a low-dimensional policy design vector.

Example constant-policy vector:

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

Let the existing model evaluate that policy:

```text
result = M(x)
```

where `result` contains time histories of pilot inventory, readiness shortfalls, training pipeline quantities, staff quantities, and any other outputs required to determine feasibility.

Define constraint functions:

```text
g_j(x) <= 0 means constraint j is satisfied.
g_j(x) > 0 means constraint j is violated.
```

Examples:

```text
g_total_pilots = target_total_pilots - observed_total_pilots

g_wg_readiness = observed_wg_shortfall - allowed_wg_shortfall

g_line_pilots = target_line_pilots - observed_line_pilots

g_training_backlog = observed_training_backlog - allowed_training_backlog

g_experience_ratio = minimum_experience_ratio - observed_experience_ratio
```

Normalize and aggregate:

```text
phi(x) = max_j(g_j(x) / scale_j)
```

Then:

```text
phi(x) <= 0    feasible
phi(x) > 0     infeasible
```

The feasible envelope is:

```text
F = { x : phi(x) <= 0 }
```

The central deliverable is to learn, search, visualize, and verify this set `F`.

---

## 5. Why Feasibility Comes Before Optimization

The primary decision question is:

```text
Does any allowable policy produce both enough pilots and enough readiness?
```

This is not first an optimization question. It is a feasibility question:

```text
find x such that all requirements are satisfied
```

Optimization becomes useful in three ways:

### 5.1 Find any feasible point

Solve:

```text
minimize phi(x)
```

If the best verified `phi(x) <= 0`, at least one feasible policy exists.

### 5.2 Find the feasible boundary

Search for points where:

```text
phi(x) ~= 0
```

These points show where the model transitions from feasible to infeasible.

### 5.3 Find minimum required policy changes

If no feasible point is found under current assumptions, solve variations of:

```text
minimize policy burden
subject to phi(x) <= 0
```

or:

```text
minimize remaining violation
```

This answers:

```text
How much more intake, retention, UTE, PAA, or upgrade capacity would be needed?
```

---

## 6. Objective vs Constraint

Avoid collapsing the problem into an RL-style scalar reward.

RL-style framing:

```text
reward = weighted blend of pilot count, readiness, staff fill, etc.
```

Viability framing:

```text
pilot count requirement        -> constraint
readiness requirement          -> constraint
staffing requirement           -> constraint
training backlog requirement   -> constraint
policy bounds                  -> constraint
policy burden / margin         -> optional objective
```

Thresholds usually belong in constraints, not in the objective.

For example, instead of:

```text
maximize total pilots
```

use:

```text
total pilots >= required threshold
```

Then use an objective such as:

```text
minimize policy burden
maximize feasibility margin
minimize time to feasibility
minimize maximum normalized violation
```

This produces more decision-relevant results than “maximize readiness at any cost.”

---

## 7. Feasible Envelope Outputs

The end product should not be a single optimum. It should be a package of evidence.

A good final report should answer:

### 7.1 Is it possible?

```text
Yes / no / only under specified assumptions.
```

### 7.2 Which policies make it possible?

Examples:

```text
minimum annual intake required
minimum retention level required
required UTE/PAA range
required upgrade throughput
required manning cap
```

### 7.3 What constraints bind?

Examples:

```text
pilot inventory binds first
readiness binds later
training throughput limits recovery speed
staffing trades against line readiness
```

### 7.4 How much margin exists?

Examples:

```text
feasible with 8% readiness margin
feasible only with <1% margin
infeasible by 120 pilots
infeasible by 0.7 monthly sorties
infeasible by X training deferrals
```

### 7.5 What makes an infeasible case feasible?

Examples:

```text
+X annual intake
+Y retention
+Z UTE
+K PAA
relax readiness threshold by M
extend timeline by N years
```

This is more useful than simply returning “no feasible solution found.”

---

## 8. Why Surrogate-Assisted Search Is Appropriate

The direct model evaluation is reportedly slow. The decision space is expected to be low-dimensional if the policy is parameterized carefully. This combination is a strong fit for surrogate-assisted search.

The intended loop is:

```text
initial DoE
    -> parallel direct model evaluations
        -> compute constraints and phi
            -> fit surrogate
                -> search surrogate cheaply
                    -> select informative new direct evaluations
                        -> update surrogate
                            -> repeat
```

The surrogate is not the final authority. It is an accelerator and guide.

Final claims should be based on verified direct-model runs.

---

## 9. Search Is Still Required

Feasible-envelope analysis does not eliminate optimization/search. It changes the search target.

Instead of searching for:

```text
best reward
```

search for:

```text
phi(x) <= 0 regions
phi(x) ~= 0 boundary points
minimum phi(x) points if infeasible
policy settings with robust feasibility margin
```

The search space does not need to be exhaustively enumerated. It needs to be explored enough to understand the feasible/infeasible boundary and active constraints.

---

## 10. Recommended Surrogate Targets

Train surrogates for both aggregate and diagnostic quantities.

### 10.1 Aggregate violation surrogate

Target:

```text
phi(x)
```

Use this to search for feasibility and boundary regions.

### 10.2 Individual constraint surrogate(s)

Targets:

```text
g_total_pilots(x)
g_wg_readiness(x)
g_fl_readiness(x)
g_ip_readiness(x)
g_line_pilots(x)
g_staff_fill(x)
g_training_backlog(x)
g_experience_ratio(x)
```

Use these to diagnose which constraint binds and to avoid opaque results.

### 10.3 Optional feasibility classifier

Target:

```text
feasible = 1 if phi(x) <= 0 else 0
```

This can support visualization and acquisition logic, but regression on `phi` and `g_j` is usually more informative.

---

## 11. Surrogate Model Choices

Start simple and increase complexity only when needed.

### 11.1 Linear / ridge regression

Useful for:

```text
sanity checks
quick baseline
sign/directionality checks
feature importance intuition
```

Limitations:

```text
may miss nonlinear thresholds and interactions
may poorly represent boundary curvature
not ideal as final feasibility-boundary model
```

### 11.2 Quadratic response surface

Useful next step if linear is too simple but data are sparse.

Captures simple curvature and low-order interactions.

### 11.3 Gaussian process regression / kriging

Strong candidate for the main prototype if dimensionality stays low.

Advantages:

```text
smooth interpolation
uncertainty estimates
sample efficiency
active-learning acquisition functions
length-scale information
natural fit for expensive low-dimensional models
```

Recommended default:

```text
Matérn 5/2 kernel
ARD length scales
normalized inputs
standardized targets
small nugget/noise term
```

If simulator outputs are noisy, either run replications or use a noise-aware GP.

### 11.4 Tree-based models

Useful if the response has discontinuities or sharp threshold behavior:

```text
random forest
gradient-boosted trees
extra-trees
```

Limitations:

```text
uncertainty is less principled unless using ensembles/quantile methods
less smooth for gradient-based optimization
```

### 11.5 Neural networks

Probably not the first choice for the viability surrogate unless the training set becomes large.

Useful later if:

```text
dimension grows
data volume grows
response is strongly nonlinear
GP scaling becomes problematic
```

---

## 12. Gaussian Process / Kriging Notes

For this problem, GPR is attractive because the relevant question is boundary confidence:

```text
Is this point actually feasible, or is the surrogate merely optimistic?
```

For a GP prediction:

```text
phi(x) ~ Normal(mu_phi(x), sigma_phi(x)^2)
```

Useful quantities:

```text
predicted feasibility:
    mu_phi(x) <= 0

conservative feasibility:
    mu_phi(x) + k * sigma_phi(x) <= 0

boundary target:
    abs(mu_phi(x)) small

uncertain region:
    sigma_phi(x) large

classification confidence:
    P(phi(x) <= 0)
```

For candidate final feasible policies, prefer conservative feasibility:

```text
mu_phi(x) + 1.96 * sigma_phi(x) <= 0
```

Then verify in the direct model.

---

## 13. Adaptive Sampling / Active Learning

Do not keep sampling randomly after the initial DoE.

Add points that improve boundary knowledge or validate feasible candidates.

Important point types:

```text
1. Predicted feasible points with good margin
2. Predicted boundary points where phi ~= 0
3. High-uncertainty points
4. Points where the surrogate predicts feasibility but uncertainty is high
5. Points near known operationally meaningful scenarios
6. Points near active constraints or bottlenecks
```

Simple acquisition examples:

### 13.1 Boundary uncertainty acquisition

```text
A(x) = -abs(mu_phi(x)) + lambda * sigma_phi(x)
```

This favors points near the predicted boundary with high uncertainty.

### 13.2 Conservative feasibility acquisition

Find points minimizing:

```text
mu_phi(x) + k * sigma_phi(x)
```

This searches for robustly feasible candidates.

### 13.3 Expected feasibility improvement

Use Bayesian optimization-style acquisition to find points likely to reduce violation below zero.

A simple practical substitute is to search for low lower-confidence-bound values:

```text
LCB(x) = mu_phi(x) - k * sigma_phi(x)
```

Then verify candidates in the direct model.

---

## 14. DoE Guidance

The initial design should be space-filling and include expert scenarios.

Recommended initial design components:

```text
Sobol or Latin hypercube samples
corner / bound points
current policy baseline
pragmatic policy baseline
optimistic policy baseline
known bad / stress-test scenarios
handpicked points near suspected bottlenecks
```

For low-dimensional problems, a rough initial budget is:

```text
N_initial ~= 10d to 30d
```

where `d` is the number of policy variables.

For example, if `d = 7`, start around:

```text
70 to 210 direct model runs
```

If direct model runs are very slow, start smaller but include corners and known scenarios.

Parallelism should be used aggressively because each policy evaluation is independent.

---

## 15. Feasible, Operationally Infeasible, and Domain Infeasible Points

Distinguish three categories.

### 15.1 Feasible

The input is valid and all operational requirements are met:

```text
phi(x) <= 0
```

### 15.2 Operationally infeasible

The input is valid, the model can run, but at least one requirement fails:

```text
phi(x) > 0
```

These points are valuable. They define the boundary.

### 15.3 Domain infeasible

The input combination is mathematically or structurally impossible, such as impossible counts or incompatible state quantities.

These should not be passed blindly to the physics/model. Handle them with:

```text
explicit algebraic constraints
validity checks
feasibility classifier if needed
clear status labels in output data
```

Do not confuse domain-infeasible points with operationally infeasible but model-valid points.

---

## 16. Cross-Validation and Trust Metrics

Average prediction error is not enough.

The dangerous error is a false-feasible prediction:

```text
surrogate says feasible
real model says infeasible
```

Track at least:

```text
RMSE / MAE for phi
RMSE / MAE for individual g_j
constraint sign accuracy
false-feasible rate
false-infeasible rate
boundary error near phi = 0
calibration of GP uncertainty if using GPR
performance on low-experience / bottleneck scenarios
```

Suggested definitions:

```text
false_feasible_rate = P(real_phi > 0 | predicted_phi <= 0)

constraint_sign_accuracy = P(sign(real_g_j) == sign(predicted_g_j))

boundary_subset = points where abs(real_phi) <= boundary_tolerance
boundary_mae = MAE(predicted_phi, real_phi on boundary_subset)
```

For decision support, false-feasible rate matters more than aggregate R².

---

## 17. Direct Model Evaluation and Stochasticity

The existing simulator may contain stochastic elements, especially retention behavior. If stochasticity remains active, a single evaluation of `M(x)` may not represent the expected outcome.

Implementation options:

```text
1. Fix random seeds for deterministic reproducibility.
2. Run multiple replications per design and aggregate outputs.
3. Replace stochastic pieces with expected-value approximations if appropriate.
4. Treat stochastic variability as observation noise in the surrogate.
```

For the prototype, the simplest path is:

```text
fixed seed per design, plus optional replication support
```

But the technical background should recognize that feasibility under stochasticity is stronger than nominal feasibility.

A robust version of the problem is:

```text
P(g_j(x, omega) <= 0 for all j) >= required_confidence
```

where `omega` represents random outcomes or uncertain assumptions.

---

## 18. Policy Parameterization

The problem remains low-dimensional only if the policy is parameterized carefully.

Avoid this first:

```text
7 levers * 60 phases = 420 design variables
```

Start with one of these:

### 18.1 Constant policy

One value per lever for the full horizon.

```text
x = [intake, retention, UTE, PAA, max_manning, FLUG quota, IPUG quota]
```

### 18.2 Piecewise policy

One value per lever per multi-year block.

Example:

```text
intake_2026_2030
intake_2031_2035
retention_2026_2030
retention_2031_2035
...
```

### 18.3 Ramp policy

Parameterize each lever by:

```text
initial value
final value
ramp duration or ramp slope
```

This preserves low dimensionality while allowing realistic time variation.

### 18.4 Recommended implementation approach

Write the evaluator against an abstract policy object:

```text
policy.value(lever_name, year, phase)
```

Then constant, piecewise, and ramp policies can share the same evaluation code.

---

## 19. Handling Integer / Discrete Variables

Some variables are naturally integer-like:

```text
annual intake
PAA
FLUG quota
IPUG quota
pilot counts
student counts
```

For the prototype, use continuous optimization/search internally and round before model evaluation:

```text
x_continuous -> sanitize / round / clip -> model input
```

Record both:

```text
raw design vector
applied design vector
```

This is important because the optimizer/surrogate may operate on continuous variables, but the model may evaluate rounded values.

Later options:

```text
mixed-integer optimization
integer-aware evolutionary search
enumeration over small discrete dimensions
separate surrogate over applied integer designs
```

---

## 20. Role of GA / EA / Evolutionary Search

GA/EA methods are valid tools, especially for non-smooth, mixed discrete/continuous, or awkward response surfaces.

However, do not start by running a large GA directly on the slow model unless there is no alternative.

Better use cases:

```text
run GA/EA on the cheap surrogate
use differential evolution or CMA-ES to minimize predicted phi
use NSGA-II for trade studies after feasibility is understood
use evolutionary search to find diverse feasible policies
use parallel direct evaluations for selected candidate batches
```

GA/EA is sample-hungry. Surrogate-assisted search is likely more efficient.

Suggested default:

```text
initial DoE + GPR surrogate + adaptive sampling + surrogate search
```

Fallback / supplement:

```text
differential evolution or CMA-ES over the surrogate
parallel evaluation of selected candidates in the direct model
```

---

## 21. Gradient-Based Optimization

Gradient-based optimization is feasible only on smooth surrogates or differentiable reformulations.

It is not the best first move through the existing simulator as written because the simulator likely includes:

```text
discrete counts
thresholds
rounding
quota logic
retention randomness
staff movement queues
if/else branching
```

These break or weaken gradient assumptions.

Good use of gradients:

```text
optimize a differentiable GPR mean or neural-net surrogate
optimize a smooth aggregate of constraints
use local gradient refinement after global surrogate search
```

Less good use:

```text
direct gradient optimization through the current simulator
```

For a Martins-style engineering optimization formulation, the eventual smooth problem can be:

```text
minimize policy burden or violation
subject to smooth surrogate-predicted constraints
```

But all final candidate policies should be verified in the original model.

---

## 22. Feasible Envelope Visualization

Two visualization modes are especially useful.

### 22.1 Fixed-slice envelope

Choose two variables to plot and hold the rest fixed.

Examples:

```text
annual intake vs retention
PAA vs UTE
retention vs UTE
intake vs PAA
```

For each grid point, classify:

```text
feasible
marginal
infeasible
unknown / high uncertainty
```

This is easy to explain but depends on the fixed values of the other variables.

### 22.2 Projected feasible envelope

For two plotted variables, optimize over the remaining variables:

```text
psi(a, b) = min_z phi(a, b, z)
```

Then:

```text
psi(a, b) <= 0 means there exists some setting z that makes (a, b) feasible.
```

This is more powerful and more expensive. It is a true projection of the feasible region.

Use surrogate search for the inner minimization, then verify representative points with the direct model.

---

## 23. Minimal Implementation Flow

A practical first prototype can be:

```text
1. Define a constant policy vector with 5-8 variables.
2. Define feasibility constraints and phi.
3. Generate 70-200 initial DoE points, depending on dimension and runtime.
4. Run direct model evaluations in parallel.
5. Store raw results, constraints, phi, and feasibility status.
6. Fit ridge regression and GPR baselines.
7. Search the surrogate using dense sampling and differential evolution.
8. Select top predicted feasible, boundary, and uncertain points.
9. Evaluate selected points in the direct model.
10. Update surrogate.
11. Generate 2-D envelope plots.
12. Report feasible candidates, active constraints, and margins.
```

The prototype should be modular enough that later phases can add:

```text
piecewise policies
robust/stochastic feasibility
multi-objective trade studies
better surrogate ensembles
active-learning acquisition functions
projected envelopes
```

---

## 24. Example Pseudocode

```python
# 1. Define design space
space = DesignSpace.from_config("configs/viability/default.yaml")

# 2. Generate initial DoE
X = sobol_or_lhs(space, n_samples=150)
X = add_baseline_and_corner_cases(X, space)

# 3. Evaluate direct model in parallel
records = parallel_map(evaluate_policy, X)

# 4. Compute constraints and phi
for r in records:
    r.constraints = compute_constraints(r.model_outputs, config.constraints)
    r.phi = max_normalized_violation(r.constraints, config.scales)
    r.feasible = r.phi <= 0.0

# 5. Fit surrogates
phi_model = fit_gpr(X, [r.phi for r in records])
g_models = fit_constraint_models(X, [r.constraints for r in records])

# 6. Search surrogate
candidate_pool = sample_many(space, n=100_000)
mu, sigma = phi_model.predict(candidate_pool, return_std=True)

boundary_candidates = select_near_boundary(candidate_pool, mu, sigma)
feasible_candidates = select_conservative_feasible(candidate_pool, mu, sigma)
uncertain_candidates = select_high_uncertainty(candidate_pool, mu, sigma)

X_new = batch_select(boundary_candidates, feasible_candidates, uncertain_candidates)

# 7. Verify selected candidates in direct model
new_records = parallel_map(evaluate_policy, X_new)

# 8. Repeat until boundary stabilizes
```

---

## 25. Recommended Data Schema

Each evaluated design should produce one row in a tabular artifact.

Suggested fields:

```text
design_id
raw_annual_intake
raw_retention_rate
raw_ute
raw_paa
raw_max_manning_pct
raw_flug_quota_per_phase
raw_ipug_quota_per_phase

applied_annual_intake
applied_retention_rate
applied_ute
applied_paa
applied_max_manning_pct
applied_flug_quota_per_phase
applied_ipug_quota_per_phase

status
run_seed
n_replications

final_total_pilots
min_total_pilots_assessment_window
final_line_pilots
min_line_pilots_assessment_window
final_staff_pilots
min_experience_ratio_assessment_window
max_wg_rap_shortfall
max_fl_rap_shortfall
max_ip_rap_shortfall

constraint_total_pilots
constraint_line_pilots
constraint_wg_readiness
constraint_fl_readiness
constraint_ip_readiness
constraint_experience_ratio
constraint_training_backlog

phi
feasible
active_constraint
active_constraint_value
notes
```

Use Parquet or CSV for results. Parquet is preferable for larger runs.

---

## 26. Active Constraint Identification

For each evaluated design, identify the active or most violated constraint:

```python
active_constraint = max(constraints.items(), key=lambda item: item[1] / scale[item[0]])
```

This supports bottleneck reporting.

Useful report fields:

```text
best feasible designs ranked by policy burden
best infeasible designs ranked by smallest phi
most common active constraint among near-feasible designs
minimum relaxation needed for infeasible best case
```

---

## 27. Robust Feasibility

Nominal feasibility means one deterministic or seeded run satisfies constraints.

Robust feasibility means the policy works under uncertainty.

Potential uncertainty sources:

```text
retention variation
initial inventory uncertainty
surrogate error
training throughput variation
resource availability variation
future intake variation
```

Prototype robust approach:

```text
for selected candidate policies:
    run K replications or scenarios
    compute phi for each
    report mean phi, worst phi, and probability feasible
```

Robust feasibility criterion example:

```text
P(phi(x, omega) <= 0) >= 0.90
```

or conservative:

```text
max_k phi(x, omega_k) <= 0
```

Do not make robust feasibility mandatory for the first prototype, but design the evaluation schema to support replications.

---

## 28. Key Technical Risks

### 28.1 False-feasible surrogate predictions

Most dangerous failure mode:

```text
surrogate predicts phi <= 0, direct model returns phi > 0
```

Mitigation:

```text
conservative GP margin
verification in direct model
boundary-focused CV
active learning around false-feasible cases
```

### 28.2 Extrapolation

The optimizer may push to regions poorly covered by DoE.

Mitigation:

```text
input bounds
distance-to-training-data checks
GP uncertainty
explicit domain feasibility checks
adaptive sampling
```

### 28.3 Overly high-dimensional policy

If every lever varies every phase, GPR becomes much less effective.

Mitigation:

```text
constant / piecewise / ramp policies
small number of policy parameters
```

### 28.4 Stochastic simulator outputs

A single run may misclassify feasibility.

Mitigation:

```text
fixed seeds
replications
noise-aware surrogate
robust feasibility reporting
```

### 28.5 Confusing internal surrogate with viability surrogate

The existing brain predicts local sortie/sim rates. The new surrogate predicts end-to-end feasibility margins.

Mitigation:

```text
clear naming, e.g. `sortie_brain` vs `viability_surrogate`
```

### 28.6 Treating the surrogate as ground truth

The direct model remains the verification authority.

Mitigation:

```text
all candidate feasible policies must be verified by direct model evaluation
```

---

## 29. How to Discuss This With Stakeholders

Recommended language:

```text
We are not trying to find a single magic optimum first.
We are trying to map the combinations of policy levers that make the modeled force viable.

A viable policy is one that satisfies explicit requirements for inventory, readiness,
staffing, and training throughput.

Because direct model runs are slow, we use a surrogate to cheaply explore the policy space,
but all final candidate policies are checked with the original model.

The deliverable is the feasible envelope, the binding bottlenecks, and the minimum policy
changes needed to make the force viable under the model assumptions.
```

Avoid leading with:

```text
We are building a genetic algorithm.
We are doing AI optimization.
We are replacing the model with a surrogate.
```

Better:

```text
We are doing model-based feasibility analysis with surrogate acceleration.
```

---

## 30. Relationship to the PRD

The PRD specifies what Codex should build:

```text
modules
CLI commands
schemas
acceptance criteria
phased implementation
```

This technical background explains why those pieces exist:

```text
why phi is the central scalar
why feasibility is the top-level framing
why GPR is attractive
why direct model verification is required
why GA/EA is a tool but not the main conceptual frame
why policy parameterization matters
why false-feasible predictions are the main risk
```

Use the PRD as the implementation plan. Use this document as the modeling rationale.

---

## 31. Short Version for Codex Context

Build a prototype that maps:

```text
F = { x : phi(x) <= 0 }
```

where:

```text
x = low-dimensional policy vector
phi(x) = maximum normalized constraint violation
```

Use the existing model as the evaluator. Use a surrogate only to accelerate search and boundary mapping. Start with constant policies, direct parallel evaluation, ridge/GPR surrogates, and 2-D envelope plots. Verify all candidate feasible policies in the direct model. Report feasible designs, infeasible near-misses, active constraints, and margins.

