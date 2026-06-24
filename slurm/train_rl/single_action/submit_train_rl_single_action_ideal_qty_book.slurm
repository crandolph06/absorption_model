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
#SBATCH --job-name=train_rl_single_action_ideal_qty_book
#SBATCH --export=ALL
#SBATCH --output=$WORKDIR/absorption_model/logs/train_rl_%j.out
#SBATCH --error=$WORKDIR/absorption_model/logs/train_rl_%j.err
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
mkdir -p $WORKDIR/absorption_model/logs
#
python -u rl_trainers/single_action/train_rl_single_action_ideal_qty_book.py
#
