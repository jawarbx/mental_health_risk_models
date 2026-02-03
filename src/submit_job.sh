#!/bin/bash

source .slurm
# Create logs directory if it doesn't exist
mkdir -p $LOGS

echo "========================================="
echo "Submitting Training Pipeline"
echo "========================================="

# Submit dataset creation job
echo "Submitting dataset creation job..."

DATASET_JOB=$(sbatch --parsable \
	--output=${LOGS}/${EXPERIMENT_NAME}_dataset_%j.out \
	--error=${LOGS}/${EXPERIMENT_NAME}_dataset_%j.err create_dataset.sh \
	--job-name=$EXPERIMENT_NAME_dataset \
	--mail-type=BEGIN,END,FAIL \
	--mail-user=${USER_EMAIL} \
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

echo "✓ Training job submitted: Job ID $TRAIN_JOB"
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
echo ""
echo "========================================="
