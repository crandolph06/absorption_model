#!/bin/bash
#
## Required Slurm Directives --------------------------------------
#SBATCH --account=OUSAF40080700
#SBATCH --partition=general
#SBATCH --qos=background
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#
## Optional Slurm Directives --------------------------------------
#SBATCH --job-name=train_rl_parallelized_opt_read_book
#SBATCH --export=ALL
#SBATCH --output=logs/train_rl_%j.out
#SBATCH --error=logs/train_rl_%j.err
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
mkdir -p logs
#
python -u train_rl_parallelized_opt_read_book.py
#
