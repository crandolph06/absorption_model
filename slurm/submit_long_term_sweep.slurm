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
#SBATCH --job-name=long_term_sweep_full
#SBATCH --export=ALL
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
#SBATCH --mem=256G
#
## Execution Block ----------------------------------------------
# Environment Setup
JOBID=`echo $SLURM_JOB_ID`
#
PROJECT_DIR="$WORKDIR/absorption_model"
cd $PROJECT_DIR
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
python -u hpc_long_term_sweeper.py
#

if [ -f SWEEP_COMPLETE.txt ]; then
   echo "✅ Sweep finished. Cleaning up and exiting."
else
   echo "🔄 Sweep not finished. Resubmitting..."
   sbatch /p/work1/clrand16/absorption_model/submit_long_term_sweep.slurm
fi
