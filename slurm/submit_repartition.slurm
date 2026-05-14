#!/bin/bash
#
## Required Slurm Directives --------------------------------------
#SBATCH --account=OUSAF40080700
#SBATCH --partition=general
#SBATCH --qos=background
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#
## Optional Slurm Directives --------------------------------------
#SBATCH --job-name=repartition_dataset
#SBATCH --export=ALL
#SBATCH --output=logs/repart_%j.out
#SBATCH --error=logs/repart_%j.err
#
## Execution Block ----------------------------------------------
PROJECT_DIR="$WORKDIR/absorption_model"
cd "$PROJECT_DIR" || exit 1 

module purge
module load python/3.11/3.11 

source /p/work1/clrand16/absorption_model/venv/bin/activate 

echo "Checking Environment..."
which python
python --version

export PYTHONUNBUFFERED=1 

echo "Starting Repartitioning: 996 batches -> 50 parts..."

# Run the repartitioning script
mkdir -p logs
python -u do_repartition.py

if [ $? -eq 0 ]; then
   echo "✅ Repartitioning complete."
else
   echo "❌ Repartitioning failed. Check logs/repart_$SLURM_JOB_ID.err"
fi