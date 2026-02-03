#!/bin/bash

# Function to display help
show_help() {
	cat << EOF
Usage: ./create_dataset.sh [OPTIONS]
   or: sbatch create_dataset.sh [OPTIONS]

Create MCI dataset with specified parameters.

Options:
    --month_deltas DELTAS      Month deltas for prediction windows (space-separated integers)
                               Default: 6 8 12
                               Example: --month_deltas 6 12 18
    
    --matching_method METHOD   Matching method for samples
                               Choices: PSM, none
                               Default: none
                               Example: --matching_method PSM
    
    -h, --help                 Show this help message and exit

Examples:
    # Direct execution
    ./create_dataset.sh --month_deltas 6 8 12 --matching_method PSM
    ./create_dataset.sh --month_deltas 12 24
    
    # SLURM submission
    sbatch create_dataset.sh --help (not recommended to run this)
    sbatch create_dataset.sh --month_deltas 6 12

EOF
    exit 0
}

# Parse command line arguments
PYTHON_ARGS=""

while [[ $# -gt 0 ]]; do
	case $1 in
		-h|--help)
			show_help
			;;
		--month_deltas)
			shift
			DELTAS=""
			while [[ $# -gt 0 ]] && [[ ! $1 =~ ^-- ]]; do
				DELTAS="$DELTAS $1"
				shift
			done
			PYTHON_ARGS="$PYTHON_ARGS --month_deltas$DELTAS"
			;;
		--matching_method)
			PYTHON_ARGS="$PYTHON_ARGS --matching_method $2"
			shift 2
			;;
		*)
			echo "Unknown option: $1"
			echo "Use --help for usage information"
			exit 1
			;;
	esac
done

# SLURM directives (only used when submitted via sbatch)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=24:00:00

source .slurm

echo "=========================================="
echo "Job Information"
echo "=========================================="
if [ -n "$SLURM_JOB_ID" ]; then
	echo "Execution Mode: SLURM"
	echo "Job ID: $SLURM_JOB_ID"
	echo "Job Name: $SLURM_JOB_NAME"
	echo "Node: $SLURM_NODELIST"
	echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
else
	echo "Execution Mode: Direct"
	echo "Hostname: $(hostname)"
	echo "Number of CPUs: $(nproc)"
fi
echo "Start Time: $(date)"
echo "Python Arguments: $PYTHON_ARGS"
echo "=========================================="

mkdir -p $LOGS
module load $CONDA_MODULE
conda activate $CONDA_ENV
export TOKENIZERS_PARALLELISM=false
mkdir -p $HF_HOME

python src/preprocess.py $PYTHON_ARGS

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
