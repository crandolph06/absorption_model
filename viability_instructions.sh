python -m src.viability.cli run-doe \
  --config configs/viability/a_current_unit.yaml \
  --output-dir outputs/viability/rap_a_current_unit/doe_128 \
  --workers 8

python -m src.viability.cli select-holdout \
  --config configs/viability/a_current_unit.yaml \
  --evaluations outputs/viability/rap_a_current_unit/doe_128/evaluations.parquet \
  --output outputs/viability/rap_a_current_unit/doe_128/holdout.parquet

python -m src.viability.cli active-learn \
  --config configs/viability/a_current_unit.yaml \
  --evaluations outputs/viability/rap_a_current_unit/doe_128/evaluations.parquet \
  --holdout-evaluations outputs/viability/rap_a_current_unit/doe_128/holdout.parquet \
  --output-dir outputs/viability/rap_a_current_unit/active_learn

# Surrogate from active learning
SURR=outputs/viability/rap_a_current_unit/active_learn/surrogate_constraints_gpr.joblib

python -m src.viability.cli search \
  --config configs/viability/a_current_unit.yaml \
  --surrogate "$SURR" \
  --output-dir outputs/viability/rap_a_current_unit/search

python -m src.viability.cli verify-candidates \
  --config configs/viability/a_current_unit.yaml \
  --candidates outputs/viability/rap_a_current_unit/search/candidate_policies.csv \
  --output-dir outputs/viability/rap_a_current_unit/verify \
  --workers 8

python -m src.viability.cli plot-envelope \
  --config configs/viability/a_current_unit.yaml \
  --surrogate "$SURR" \
  --evaluations outputs/viability/rap_a_current_unit/doe_128/evaluations.parquet \
  --verified-candidates outputs/viability/rap_a_current_unit/verify/verified_candidates.parquet \
  --output-dir outputs/viability/rap_a_current_unit/envelope

python -m src.viability.cli make-report \
  --config configs/viability/a_current_unit.yaml \
  --evaluations outputs/viability/rap_a_current_unit/doe_128/evaluations.parquet \
  --verified-candidates outputs/viability/rap_a_current_unit/verify/verified_candidates.parquet \
  --search-summary outputs/viability/rap_a_current_unit/search/search_summary.json \
  --verification-summary outputs/viability/rap_a_current_unit/verify/verification_summary.json \
  --envelope-summary outputs/viability/rap_a_current_unit/envelope/envelope_summary.json \
  --output outputs/viability/rap_a_current_unit/report.md

streamlit run viability_dashboard.py