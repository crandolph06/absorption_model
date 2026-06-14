# Internal Phase Surrogate Audit And Retraining Plan

## Summary

Before relying on the viability surrogate and search workflow, verify that the
existing internal phase surrogate is accurate enough. The viability workflow now
uses the configured 16-output MLP at:

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

The goal of this phase is to answer:

> Is the configured 16-output internal phase surrogate good enough to support
> long-horizon feasibility analysis?

If not, train a replacement single-phase surrogate ourselves and use that in the
viability workflow.

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

## Phase 2: Held-Out Accuracy Audit

Purpose: score the configured 16-output MLP against direct single-phase truth.

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

## Phase 3: Long-Horizon Sensitivity Check

Purpose: determine whether plausible internal-surrogate errors matter for the
outer feasibility conclusions.

Tasks:

1. Select a small set of long-horizon policies:
   - known infeasible policies
   - near-boundary policies
   - best surrogate-screened candidate policies
   - any verified feasible candidates, if available
2. For each policy, inspect the phase-level feature rows generated during the
   long-horizon rollout.
3. Compare those feature rows to the single-phase training/holdout feature
   distribution.
4. Flag extrapolation or low-density regions.
5. Summarize which internal outputs dominate downstream feasibility outcomes.

Deliverable:

```text
outputs/single_phase/audit/long_horizon_feature_coverage.csv
outputs/single_phase/audit/long_horizon_sensitivity_summary.md
```

Acceptance criteria:

- We know whether long-horizon feasibility search is operating inside the
  single-phase surrogate's training domain.
- Near-boundary viability conclusions are labeled with the relevant internal
  surrogate risk.

## Phase 4: Retrain If Needed

Trigger this phase if the MLP audit shows unacceptable errors, unstable boundary
behavior, or poor coverage in the regions used by long-horizon feasibility
search.

Training data plan:

1. Preserve the fixed holdout from Phase 2.
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

Start simple:

```text
per-output gradient boosted trees or random forests
```

Keep MLP as a baseline:

```text
current multi-output MLP architecture
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

## Phase 5: Viability Recheck With The Selected Internal Surrogate

Purpose: quantify whether changing the internal phase surrogate changes the
outer feasibility story.

Tasks:

1. Run a small fixed set of long-horizon policy evaluations with:
   - current configured MLP
   - replacement internal surrogate, if trained
2. Compare raw metrics, constraint margins, `phi`, feasibility labels, and active
   constraints.
3. Refit or reload the outer signed-RAP viability surrogate only after selecting
   the internal surrogate.
4. Re-run candidate search and direct verification with the selected internal
   surrogate.

Deliverable:

```text
outputs/viability/internal_surrogate_comparison/comparison_metrics.csv
outputs/viability/internal_surrogate_comparison/summary.md
```

Acceptance criteria:

- We can state whether the feasibility envelope is robust to the internal
  surrogate choice.
- If the old and new internal surrogates disagree near the feasibility boundary,
  direct single-phase validation becomes a required caveat for any claimed
  feasible region.

## Guardrails

- Do not use the 12-output legacy artifact for viability evaluation.
- Do not overwrite existing surrogate artifacts. Write replacements to a new
  short, explicit directory.
- Keep the fixed holdout read-only once created.
- Keep generated audit outputs out of git unless explicitly requested.
- Avoid hidden defaults in any new audit/retraining commands.
- Prefer short artifact directories:

```text
outputs/single_phase/audit
outputs/single_phase/retrain
outputs/viability/internal_surrogate_comparison
```

## Suggested Implementation Order

1. Add a read-only audit command or script for artifact interface/provenance.
2. Add a holdout scoring command for the configured 16-output MLP.
3. Generate or locate a fixed single-phase holdout.
4. Produce metrics and plots for the current MLP.
5. Decide whether retraining is needed.
6. If needed, train replacement candidates and compare on the fixed holdout.
7. Point viability config at the selected internal surrogate.
8. Re-run the outer viability surrogate/search verification path.
9. Perform a cleanup pass:
   - simplify names
   - delete duplicate helper paths
   - remove unused knobs
   - make stale-artifact risk explicit
   - keep the final docs and code readable

## Decision Point

At the end of Phase 2, make a deliberate call:

```text
Current 16-output MLP is good enough:
    Continue viability search with the configured artifact.

Current 16-output MLP is not good enough:
    Train and select a replacement single-phase surrogate before trusting
    outer feasibility results.
```

