#!/bin/bash

source .slurm
# Check arguments

# Default configuration
DEFAULT_MONTH_DELTAS="6 8 12"
DEFAULT_MATCHING_METHOD=""

# Initialize with defaults
MONTH_DELTAS="${MONTH_DELTAS:-$DEFAULT_MONTH_DELTAS}"
MATCHING_METHOD="${MATCHING_METHOD:-$DEFAULT_MATCHING_METHOD}"

# Parse command-line arguments to override defaults
while [[ $# -gt 0 ]]; do
    case $1 in
        --month_deltas)
            shift
            MONTH_DELTAS=""
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                MONTH_DELTAS="$MONTH_DELTAS $1"
                shift
            done
            MONTH_DELTAS=$(echo $MONTH_DELTAS | xargs)  # Trim spaces
            ;;
        --matching_method)
            MATCHING_METHOD="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--month_deltas <deltas...>] [--matching_method <PSM|None>]"
            exit 1
            ;;
    esac
done

# Create logs directory if it doesn't exist
mkdir -p $LOGS

echo "========================================="
echo "Submitting Training Pipeline"
echo "========================================="

# Submit dataset creation job
echo "Submitting dataset creation job..."

# Build arguments string for passing to dataset creation
DATASET_ARGS="--month_deltas $MONTH_DELTAS"
if [ -n "$MATCHING_METHOD" ]; then
    DATASET_ARGS="$DATASET_ARGS --matching_method $MATCHING_METHOD"
fi

export MONTH_DELTAS
export MATCHING_METHOD

DATASET_JOB=$(sbatch --parsable \
	--output=${LOGS}/${EXPERIMENT_NAME}_dataset_%j.out \
	--error=${LOGS}/${EXPERIMENT_NAME}_dataset_%j.err create_dataset.sh \
	--job-name=$EXPERIMENT_NAME_dataset \
	--mail-type=BEGIN,END,FAIL \
	--mail-user=${USER_EMAIL} \
	--export=ALL, MONTH_DELTAS="$MONTH_DELTAS", MATCHING_METHOD="$MATCHING_METHOD" \
	create_dataset.sh
)

if [ -z "$DATASET_JOB" ]; then
    echo "ERROR: Failed to submit dataset creation job"
    exit 1
fi

echo "Dataset job submitted: Job ID $DATASET_JOB"

echo "Submitting training job with dependency..."
TRAIN_JOB=$(sbatch --parsable \
	--dependency=afterok:$DATASET_JOB \
	--output=${LOGS}/${EXPERIMENT_NAME}_training_%j.out \
	--error=${LOGS}/${EXPERIMENT_NAME}_training_%j.err \
	--job-name=$EXPERIMENT_NAME_training \
	--mail-type=BEGIN,END,FAIL \
	--mail-user=${USER_EMAIL} \
	run_training.sh
)

if [ -z "$TRAIN_JOB" ]; then
    echo "ERROR: Failed to submit training job"
    exit 1
fi

echo "Training job submitted: Job ID $TRAIN_JOB"
echo ""
echo "========================================="
echo "Pipeline Summary:"
echo "========================================="
echo "1. Dataset Creation: Job $DATASET_JOB"
echo "2. Training:         Job $TRAIN_JOB (runs after $DATASET_JOB completes)"
echo ""
echo "View logs:"
echo "  tail -f ${LOGS}/${EXPERIMENT_NAME}_dataset_${DATASET_JOB}.out"
echo "  tail -f ${LOGS}/${EXPERIMENT_NAME}_training_${TRAIN_JOB}.out"
echo "View errors:"
echo "  tail -f ${LOGS}/${EXPERIMENT_NAME}_dataset_${DATASET_JOB}.err"
echo "  tail -f ${LOGS}/${EXPERIMENT_NAME}_training_${TRAIN_JOB}.err"
echo ""
echo "========================================="
