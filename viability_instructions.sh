# 0. Delete old outputs if needed
mv outputs/viability/rap_a_current_enterprise outputs/viability/rap_a_current_enterprise_broken_physics_$(date +%Y%m%d)
mkdir -p outputs/viability/rap_a_current_enterprise

# 1. DOE (128 designs)
python -m src.viability.cli run-doe \
  --config configs/viability/a_current_enterprise.yaml \
  --output-dir outputs/viability/rap_a_current_enterprise/doe_128 \
  --workers 8

# 2. Holdout
python -m src.viability.cli select-holdout \
  --config configs/viability/a_current_enterprise.yaml \
  --evaluations outputs/viability/rap_a_current_enterprise/doe_128/evaluations.parquet \
  --output outputs/viability/rap_a_current_enterprise/doe_128/holdout.parquet

# 3. Active learning (3 × 64 = 192 new evals)
python -m src.viability.cli active-learn \
  --config configs/viability/a_current_enterprise.yaml \
  --evaluations outputs/viability/rap_a_current_enterprise/doe_128/evaluations.parquet \
  --holdout-evaluations outputs/viability/rap_a_current_enterprise/doe_128/holdout.parquet \
  --output-dir outputs/viability/rap_a_current_enterprise/active_learn

# 4. Surrogate (from active-learn state)
SURR=$(python -c "import json; print(json.load(open('outputs/viability/rap_a_current_enterprise/active_learn/state.json'))['latest_model_path'])")

python -m src.viability.cli search \
  --config configs/viability/a_current_enterprise.yaml \
  --surrogate "$SURR" \
  --output-dir outputs/viability/rap_a_current_enterprise/search

python -m src.viability.cli verify-candidates \
  --config configs/viability/a_current_enterprise.yaml \
  --candidates outputs/viability/rap_a_current_enterprise/search/candidate_policies.csv \
  --output-dir outputs/viability/rap_a_current_enterprise/verify \
  --workers 8

python -m src.viability.cli plot-envelope \
  --config configs/viability/a_current_enterprise.yaml \
  --surrogate "$SURR" \
  --evaluations outputs/viability/rap_a_current_enterprise/doe_128/evaluations.parquet \
  --verified-candidates outputs/viability/rap_a_current_enterprise/verify/verified_candidates.parquet \
  --output-dir outputs/viability/rap_a_current_enterprise/envelope

python -m src.viability.cli make-report \
  --config configs/viability/a_current_enterprise.yaml \
  --evaluations outputs/viability/rap_a_current_enterprise/doe_128/evaluations.parquet \
  --verified-candidates outputs/viability/rap_a_current_enterprise/verify/verified_candidates.parquet \
  --search-summary outputs/viability/rap_a_current_enterprise/search/search_summary.json \
  --verification-summary outputs/viability/rap_a_current_enterprise/verify/verification_summary.json \
  --envelope-summary outputs/viability/rap_a_current_enterprise/envelope/envelope_summary.json \
  --output outputs/viability/rap_a_current_enterprise/report.md