# Tyler Optimization Closeout Plan

## Purpose

Close this branch with a paper-quality, evidence-backed analysis of the pilot-production viability search. The next work should make the process understandable, show the policy trade space visually, and answer what would need to change to obtain a feasible policy: relaxed requirements, relaxed input bounds, or both.

Status: execution in progress on local branch `Tyler-optimization`. Raw simulation outputs remain under ignored `outputs/viability/...`; curated paper figures are tracked under `docs/figures/viability`.

## Immediate Facts To Carry Forward

### Current policy input bounds

The current authoritative physics config is `outputs/viability/dynamic_policy_search/config_physics.yaml`.

| Input | Type | Current bounds |
| --- | --- | --- |
| `annual_intake` | int | `[10, 350]` |
| `retention_rate` | float | `[0.10, 0.65]` |
| `ute` | float | `[6, 20]` |
| `paa` | int | `[18, 30]` |
| `max_manning_pct` | float | `[100, 200]` |
| `flug_quota_per_phase` | int | `[0, 10]` |
| `ipug_quota_per_phase` | int | `[0, 10]` |

The input-bound question is not fully answered yet. We know the current bounds, and we have observed near misses under those bounds. We do not yet have a systematic study showing the smallest input-bound changes that create feasibility.

### Current requirements

| Requirement | Current value |
| --- | --- |
| `target_total_pilots` | `3500` |
| `allowed_wg_rap_shortfall` | `0.0` |
| `allowed_fl_rap_shortfall` | `0.0` |
| `allowed_ip_rap_shortfall` | `0.0` |
| `target_line_pilots` | disabled |
| `min_experience_ratio` | disabled |
| `target_staff_ips` | disabled |
| `target_staff_fls` | disabled |

The constraint convention is `g_i <= 0` satisfied. The scalar search objective is `phi = max_i(g_i / scale_i)`, so `phi <= 0` means feasible and `phi = 0` is exactly on the active feasibility boundary.

### Meaning of `total_pilots_window`

`total_pilots_window = target_total_pilots - min_total_pilots_after_assessment_start`.

So a positive value means the policy dips below the 3500-pilot target at least once during the assessment window. It is not necessarily the final-year shortfall. For example, if `total_pilots_window = 533`, the minimum post-assessment total pilot count is `3500 - 533 = 2967`.

### Current best observed direct-physics result

The current best refined 3-epoch direct-physics policy is `refine_0008` from `outputs/viability/dynamic_policy_search/run_3epoch_refine_2048_131072_256`.

| Quantity | Value |
| --- | ---: |
| `phi` | `5.830747666120982` |
| active constraint | `wg_rap` |
| positive `total_pilots_window` relaxation | `533.0` pilots |
| positive `wg_rap` relaxation | `5.830747666120982` |
| positive `fl_rap` relaxation | `2.948444061385238` |
| positive `ip_rap` relaxation | `0.0` |
| feasible direct-physics policies observed | `0` |

This is observed-search evidence, not a mathematical infeasibility proof.

## Branch And Worktree Hygiene

- [x] Preserve the current uncommitted edit to `docs/viability_dynamic_policy_technical_note.tex` by resolving the TODO intent into a rewritten note.
- [x] Rename the local branch from `codex-viability-prototype` to `Tyler-optimization`.
- [x] Do not push.
- [x] Keep generated raw outputs under ignored `outputs/viability/...`.
- [x] Track only curated publication figures under `docs/figures/viability`.

## Paper Revision Plan

### TODO intent captured from the TeX

The current TeX TODOs are not asking for cosmetic edits. They are asking for a structural rewrite that makes the note read like a technical method and result, not a branch diary.

- [x] Rewrite the abstract and introduction so they introduce the problem first: find a policy that can produce 3500 pilots while satisfying WG/FL/IP RAP requirements.
- [x] Remove "this branch..." framing from the opening. Branch and implementation details belong later or in a footnote, not in the first argument of the paper.
- [x] Explain the workflow in plain order before using jargon:
  1. define policy levers and requirements;
  2. use Sobol sampling to cover the policy space;
  3. fit/use surrogates to cheaply search for promising feasible or near-feasible regions;
  4. expand/refine around the best regions;
  5. verify selected policies with the direct physics solver.
- [x] Briefly explain Sobol sampling as a low-discrepancy, space-filling sequence used to cover high-dimensional bounded policy spaces more evenly than naive random sampling.
- [x] Roll the physics/surrogate hierarchy into the introduction: surrogate for exploration, direct physics for claims.
- [x] Collapse the MLP history into a footnote or short aside. It should justify why the MLP is not authoritative without becoming a full section.
- [x] Explain finite-horizon control before naming it. The reader needs to understand that a policy is a sequence of decisions applied over a 20-year simulation horizon before seeing "nonlinear optimal control."
- [x] Define epochs concretely: for a 20-year, 60-phase simulation, an `E`-epoch policy divides the horizon into `E` contiguous blocks; each block holds the seven controls fixed.
- [x] Remove the temporary TODO markers from the TeX by resolving them into prose, figures, and tables.

### High-level method section

- [x] Add a clearer early-roadmap paragraph that says what the branch actually does:
  1. define the finite-horizon policy-search problem;
  2. sample and seed candidate schedules;
  3. screen or refine candidates with surrogate guidance where useful;
  4. verify selected candidates with the physics simulator;
  5. diagnose nearest misses and quantify requirement/input relaxations.
- [x] Add a TikZ workflow figure modeled after `/Users/tylerkb/Desktop/code/phd-meta/papers/tis/main.tex`.
- [x] Make the source-of-truth hierarchy explicit: direct physics is authoritative; surrogates are screening and refinement tools only.
- [x] Put the process flowchart early:
  - requirements and bounds;
  - Sobol/heuristic policy generation;
  - surrogate screening/refinement;
  - direct-physics verification;
  - relaxation and trade-space analysis.

### Optimization problem statement

- [x] Add a formal subsection defining the finite-horizon nonlinear optimal-control/search problem:
  - state `x_k`: compressed simulator state at phase `k`;
  - control `u_e`: epoch policy levers for the epoch containing phase `k`;
  - dynamics `x_{k+1} = f_k(x_k, u_e)` from the physics-backed simulator;
  - decision vector `U = [u_1, ..., u_E]`, with `E = 3` or `5` in current results;
  - constraints `g_i(U) <= 0`;
  - objective `min_U phi(U) = min_U max_i(g_i(U) / s_i)`;
  - feasibility condition `phi(U) <= 0`.
- [x] Include a requirements table and a policy-bound table in the paper.
- [x] Explain that the current results are for structured open-loop epoch policies, not fully free per-phase 70-knob schedules.
- [x] Define `total_pilots_window` in the methodology before using it in results.
- [x] Explicitly state that `phi = 0` is the feasibility boundary and positive `phi` is normalized violation.

### MLP / surrogate explanation

- [x] Collapse and sharpen the MLP explanation into an introduction footnote.
- [x] State that the earlier configured 16-output MLP had all-target R2 about `0.595` and sortie-rate-group R2 about `0.244`, making it unacceptable for policy claims.
- [x] State that the later shared ARD Matern GPR improved screening quality, but final claims still use direct physics.

### IPUG / instructor-pilot clarity

- [x] Add a short explanation of what `ipug_quota_per_phase = 0` means in these policies.
- [x] Audit whether the best policies produce or retain instructor pilots through other simulator mechanisms.
- [x] If `IPUG = 0` suppresses new instructor-pilot upgrades, explain why the optimizer selected it and what trade it made.
- [x] Add staff/IP trajectory evidence and an IPUG counterfactual table discussion.
- [x] Resolve the `0,0,0` IPUG reader-facing issue in the paper.

### Results narrative

- [x] Separate "no feasible policy observed" from "infeasible problem proven."
- [x] Report the best 3-epoch refinement, best 5-epoch search, Pareto nearest misses, and relaxation study results.
- [x] Explain the binding tradeoff: WG RAP, FL RAP, and total-pilot-window violations appear together, rather than one isolated constraint blocking feasibility.
- [x] Lead results with figures and trajectories, not only scalar tables.
- [x] Show snapshots over time for the best policies: total pilots, WG/FL/IP RAP, and relevant staff/IP quantities.
- [x] Add a trade-space figure showing how near-miss policies move between total-pilot-window, WG RAP, and FL RAP violations.
- [x] Move active-constraint counts into a short result paragraph with interpretation.

### Recommendations and close

- [x] Make recommendations decision-oriented:
  - if requirements are fixed, what input-bound relaxations should be tested next;
  - if input bounds are fixed, what requirement relaxations are suggested by the observed nearest miss;
  - if neither can change, what larger policy class would be the next justified search.
- [x] Rewrite the conclusion after figures and trade studies are added. It states the decision-relevant result, not just the workflow.

## Additional Analysis To Run

### 1. Input-bound relaxation study

Goal: answer what policy-input bounds would need to change to create a feasible point.

- [x] Add a small analysis helper and CLI that can temporarily widen selected policy bounds without changing the baseline config on disk.
- [x] Start with one-at-a-time bound extensions around the best schedules:
  - retention upper bound: `0.65 -> 0.75 -> 0.85 -> 0.95`;
  - annual intake upper bound: `350 -> 400 -> 500 -> 650`;
  - UTE upper bound: `20 -> 22 -> 25 -> 30`;
  - PAA upper bound: `30 -> 35 -> 40`;
  - max manning upper bound: `200 -> 225 -> 250`;
  - FLUG and IPUG quota upper bounds: `10 -> 15 -> 20`.
- [x] For each extension, run fixed-shape schedule sweeps first so the response is cheap and interpretable.
- [x] Run a targeted widened-bound dynamic search for the most promising combination. The completed joint-bound check widened annual intake, retention, UTE, and PAA together, then ran both a fresh three-epoch search and an anchored refinement. It improved the nearest miss but still found no feasible policy.
- [x] Produce a table saying: bound changed, best observed `phi`, active constraint, total-window relaxation, WG/FL/IP RAP relaxations, and whether feasibility was found.
- [x] Clearly label this as observed evidence, not a proof over all possible policies.

### 2. Requirement-relaxation study expansion

Goal: present what requirements must be relaxed if input bounds remain fixed.

- [x] Use the existing `relaxation_study_v1` outputs as the baseline.
- [x] Add a publication table for minimum observed relaxations:
  - L-infinity normalized relaxation;
  - total-pilot-window-only relaxation;
  - WG-only, FL-only, IP-only relaxations;
  - paired and triple relaxations for total window, WG RAP, and FL RAP.
- [x] Add a trade-space plot for `total_pilots_window`, `wg_rap`, and `fl_rap`.
- [x] Explain that the best current observed policy needs simultaneous relaxation of about `533` pilots in the assessment window, `5.83` WG RAP shortfall, and `2.95` FL RAP shortfall.

### 3. Policy trajectory extraction and figures

Goal: make the paper show actual schedules and outcomes, not just scalar `phi`.

- [x] Generate trajectories for:
  - best refined 3-epoch policy `refine_0008`;
  - best 5-epoch policy;
  - best static or earlier near-miss policy if useful as a baseline;
  - selected Pareto policies with different tradeoffs.
- [x] Add figures:
  - total pilots vs time with 3500 target and assessment-window marker;
  - WG, FL, and IP RAP shortfalls vs time with zero requirement line;
  - line pilots, staff IPs, and staff FLs if these help explain the trade;
  - policy controls by epoch as a table or step plot;
  - Pareto/trade-space scatter colored by `phi` or active constraint.
- [x] Add captions that state whether each trajectory is direct-physics verified.

### 4. IPUG-specific diagnostic

Goal: answer whether the optimized schedules are creating instructor pilots.

- [x] For the best schedules, extract `ipug_quota_per_phase`, instructor-pilot counts, IP RAP shortfall, and staff IP history.
- [x] Run a small counterfactual set around the best schedule with IPUG increased while other controls are fixed.
- [x] Report whether increasing IPUG improves IP constraints, harms WG/FL RAP, harms total pilot count, or has little authority under current simulator rules.

### 5. Dashboard alignment

Goal: make the dashboard support the paper and review, not compete with it.

- [x] Make the dashboard default to dynamic direct-physics results.
- [x] Surface the same best policies, relaxation tables, trajectory plots, bound-relaxation table, and IPUG diagnostic used in the paper.
- [x] Keep static sliders as legacy/supporting workflow only.
- [x] Add clear labels for direct-verified, surrogate-screened, and requirement-relaxed policies.

## Code Changes Likely Needed

- [x] Add a viability-local helper for input-bound relaxation studies.
- [x] Add a viability-local helper for trajectory extraction/plot generation from selected dynamic policies.
- [x] Add a small IPUG/counterfactual diagnostic helper if existing diagnostics do not cover it cleanly.
- [x] Keep edits inside `src/viability`, `tests`, `docs`, and `viability_dashboard.py` unless the current code structure forces otherwise.
- [x] Add focused tests for:
  - widened-bound config cloning;
  - relaxation-table generation on synthetic data;
  - trajectory figure generation with missing columns handled clearly;
  - IPUG counterfactual schedule construction;
  - dashboard loading without feasible policies.

## Validation And Compile Steps

- [x] Run focused unit tests for viability dynamic search, relaxation, dashboard, CLI, and plots.
- [x] Run `venv/bin/python -m compileall src/viability viability_dashboard.py`.
- [x] Run `git diff --check`.
- [x] Compile `docs/viability_dynamic_policy_technical_note.tex`.
- [x] Put the compiled PDF next to the TeX.
- [x] Confirm no raw generated outputs are staged.
- [x] Commit locally only after the plan items above are coherently complete. Do not push.

## PR Readiness Remediation

- [x] Do not track the generated GPR bundle in normal git. The useful shared ARD Matern GPR artifact is about 129 MB; preserve it through Git LFS or external artifact storage only if the project needs a canonical portable model.
- [x] Update the paper author to `Tyler Korenyi-Both`.
- [x] Add paper language explaining that raw outputs stay ignored, curated figures are tracked, and dashboard users must regenerate or point to local artifacts.
- [x] Add exact closeout artifact regeneration commands to the paper appendix.
- [x] Clarify the Table 7 total-window sign convention so negative values are not mistaken for shortfall.
- [x] Distinguish the completed targeted joint widened-bound check from future exhaustive widened-bound or richer policy-class optimization.
- [x] Split the oversized `src/viability/dynamic_analysis.py` implementation into focused viability-local modules while preserving import compatibility.

## Acceptance Criteria

This branch is ready for PR when:

- [x] The branch is named `Tyler-optimization`.
- [x] The paper clearly states the optimization problem, constraints, policy bounds, point-selection process, and physics authority.
- [x] The paper includes a methodology flowchart.
- [x] The paper includes direct-physics trajectory plots for the best policies.
- [x] The paper answers what requirement relaxations are needed under current input bounds.
- [x] The paper includes a first-pass answer to what input bounds may need to relax, with enough direct-physics evidence to be useful.
- [x] The IPUG/instructor-pilot behavior is explained with data.
- [x] The dashboard reflects the dynamic-policy result bundle and does not center the legacy static-slider workflow.
- [x] Tests and LaTeX compilation pass.
- [x] The final report distinguishes observed-search evidence from proof of infeasibility.
