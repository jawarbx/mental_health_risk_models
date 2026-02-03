#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=24:00:00

source .slurm

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Start Time: $(date)"
echo "=========================================="

mkdir -p $LOGS

module load $CONDA_MODULE
conda activate $CONDA_ENV

export TOKENIZERS_PARALLELISM=false
mkdir -p $HF_HOME

python preprocess.py
