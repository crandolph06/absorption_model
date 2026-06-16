#!/bin/bash
#
## Required Slurm Directives --------------------------------------
#SBATCH --account=OUSAF40080700
#SBATCH --partition=general
#SBATCH --qos=background
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#
## Optional Slurm Directives --------------------------------------
#SBATCH --job-name=single_phase_sweep_5
#SBATCH --export=ALL
#SBATCH --output=logs/sweep_5/%x_%j.out
#SBATCH --error=logs/sweep_5/%x_%j.err
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
mkdir -p "$PROJECT_DIR/logs/sweep_5"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
timeout 14100s python -u single_phase_sweepers/hpc_single_phase_sweeper_5.py || true
#

if [ -f SWEEP_5_COMPLETE.txt ]; then
   echo "✅ Sweep finished. Cleaning up and exiting."
   rm SWEEP_5_COMPLETE.txt
else
   echo "Sweep did not finish (SWEEP_5_COMPLETE.txt missing). Resubmit manually if needed."
fi
