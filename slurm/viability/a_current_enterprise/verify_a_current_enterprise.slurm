#!/bin/bash
#
## Required Slurm Directives --------------------------------------
#SBATCH --account=OUSAF40080700
#SBATCH --partition=general
#SBATCH --qos=background
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#
## Optional Slurm Directives --------------------------------------
#SBATCH --job-name=verify_a_current_enterprise
#SBATCH --export=ALL
#SBATCH --output=logs/viability/%x_%j.out
#SBATCH --error=logs/viability/%x_%j.err
#
## Execution Block ----------------------------------------------
JOBID=`echo $SLURM_JOB_ID`
#
PROJECT_DIR="$WORKDIR/absorption_model"
cd "$PROJECT_DIR" || exit 1
#
module purge
module load python/3.11/3.11
#
source /p/work1/clrand16/absorption_model/venv/bin/activate
#
echo "Checking Environment..."
which python
python --version
#
export PYTHONUNBUFFERED=1
#
mkdir -p "$PROJECT_DIR/logs/viability"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
#
timeout 14100s python -u -m src.viability.cli verify-candidates \
  --config configs/viability/a_current_enterprise.yaml \
  --candidates outputs/viability/rap_a_current_enterprise/search/candidate_policies.csv \
  --output-dir outputs/viability/rap_a_current_enterprise/verify \
  --workers 8 || true
#