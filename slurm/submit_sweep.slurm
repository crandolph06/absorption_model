#!/bin/bash
#
## Required Slurm Directives --------------------------------------
#SBATCH --account=OUSAF40080700
#SBATCH --partition=general
#SBATCH --qos=background
#SBATCH --nodes=1
# One Python process; parallelize inside it (ProcessPoolExecutor reads SLURM_CPUS_PER_TASK).
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --exclusive
#SBATCH --time=04:00:00
#
## Optional Slurm Directives --------------------------------------
#SBATCH --job-name=single_phase_sweep_full
#SBATCH --export=ALL
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
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
mkdir -p logs
# No srun: batch jobs here do not expose srun in PATH. Python writes SWEEP_COMPLETE.txt when done (rank 0/1).
timeout 14100s python -u hpc_single_phase_sweeper.py || true
#

if [ -f SWEEP_COMPLETE.txt ]; then
   echo "✅ Sweep finished. Cleaning up and exiting."
   rm SWEEP_COMPLETE.txt
else
   echo "🔄 Sweep not finished. Resubmitting..."
   if command -v sbatch >/dev/null 2>&1; then
      sbatch "$PROJECT_DIR/slurm/submit_sweep.slurm"
   elif [ -x /usr/bin/sbatch ]; then
      /usr/bin/sbatch "$PROJECT_DIR/slurm/submit_sweep.slurm"
   else
      echo "⚠️ sbatch not found on this node; resubmit manually from a login node:" >&2
      echo "   sbatch $PROJECT_DIR/slurm/submit_sweep.slurm" >&2
   fi
fi
