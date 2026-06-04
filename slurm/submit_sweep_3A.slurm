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
#SBATCH --job-name=single_phase_sweep_3A
#SBATCH --export=ALL
#SBATCH --output=$WORKDIR/absorption_model/logs/sweep_3A_%j.out
#SBATCH --error=$WORKDIR/absorption_model/logs/sweep_3A_%j.err
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
mkdir -p $WORKDIR/absorption_model/logs
timeout 14100s python -u single_phase_sweepers/hpc_single_phase_sweeper_3A.py || true
#

if [ -f SWEEP_3A_COMPLETE.txt ]; then
   echo "✅ Sweep finished. Cleaning up and exiting."
   rm SWEEP_3A_COMPLETE.txt
else
   echo "Sweep did not finish (SWEEP_3A_COMPLETE.txt missing). Resubmit manually if needed."
fi
