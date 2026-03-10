#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=1:00:00


# Function to display help
show_help() {
	cat << EOF
Usage: ./relabel_dataset.sh [OPTIONS]
   or: sbatch relabel_dataset.sh [OPTIONS]

Relabel tokenized dataset with specified parameters.

Options:
    --month_deltas DELTAS      Month deltas for prediction windows (space-separated integers)
                               Default: 6 8 12
                               Example: --month_deltas 6 12 18
    
   
    -h, --help                 Show this help message and exit

Examples:
    # Direct execution
    ./relabel_dataset.sh --month_deltas 6 8 12 --matching_method PSM
    ./relabel_dataset.sh --month_deltas 12 24
    
    # SLURM submission
    sbatch relabel_dataset.sh --help (not recommended to run this)
    sbatch relabel_dataset.sh --month_deltas 6 12

EOF
    exit 0
}

# Parse command line arguments
PYTHON_ARGS=""
EXECUTION_MODE=0
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
		--gap)
			PYTHON_ARGS="$PYTHON_ARGS $1 $2"
			shift 2
			;;
		-l|--local)
			EXECUTION_MODE=1
			shift
			;;
		*)
			echo "Unknown option: $1"
			echo "Use --help for usage information"
			exit 1
			;;
	esac
done

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

python src/relabel_dataset.py $PYTHON_ARGS

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
