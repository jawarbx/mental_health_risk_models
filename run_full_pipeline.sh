#!/bin/bash

# Function to display help
show_help() {
    cat << EOF
Usage: ./run_full_pipeline.sh [OPTIONS]

Run the full MCI prediction pipeline (dataset creation + training).

Execution Modes:
    --local                    Run pipeline locally (non-SLURM execution)
                               Default: Submit to SLURM

Pipeline Options:
    -h, --help                 Show this help message and exit

========================================
Dataset Creation Options:
========================================
EOF
    # Call create_dataset.sh help (suppress exit)
    bash create_dataset.sh --help 2>/dev/null | sed -n '/Options:/,/Examples:/p' | head -n -1

    cat << EOF

========================================
Training Options:
========================================
EOF
    # Call run_training.sh help (suppress exit)
    bash run_training.sh --help 2>/dev/null | sed -n '/Options:/,/Examples:/p' | head -n -1

    cat << EOF

========================================
Examples:
========================================
# SLURM execution (default)
./run_full_pipeline.sh --month_deltas 6 12 --batch_size 32 --num_epochs 3

# Local execution
./run_full_pipeline.sh --local --month_deltas 6 12 18 --matching_method PSM

# Training parameters only
./run_full_pipeline.sh --learning_rate 1e-5 --batch_size 16

# Full custom pipeline
./run_full_pipeline.sh \\
    --month_deltas 6 8 12 \\
    --matching_method PSM \\
    --train_split 0.75 \\
    --test_split 0.20 \\
    --batch_size 32 \\
    --num_epochs 10 \\
    --learning_rate 3e-5

# Local execution with all parameters
./run_full_pipeline.sh --local \\
    --month_deltas 12 24 \\
    --train_split 0.8 \\
    --test_split 0.1 \\
    --val_split 0.1

EOF
    exit 0
}

# Parse command line arguments
LOCAL_MODE=false
DATASET_ARGS=""
TRAINING_ARGS=""

while [[ $# -gt 0 ]]; do
	case $1 in
		-h|--help)
			show_help
			;;
		--local)
			LOCAL_MODE=true
			shift
			;;
		# Dataset creation arguments
		--month_deltas)
			shift
			DELTAS=""
			while [[ $# -gt 0 ]] && [[ ! $1 =~ ^-- ]]; do
				DELTAS="$DELTAS $1"
				shift
			done
			DATASET_ARGS="$DATASET_ARGS --month_deltas$DELTAS"
			;;
		--matching_method)
			DATASET_ARGS="$DATASET_ARGS --matching_method $2"
			shift 2
			;;
		# Training arguments
		--train_split|--test_split|--val_split|--learning_rate)
			TRAINING_ARGS="$TRAINING_ARGS $1 $2"
			shift 2
			;;
		--batch_size|--num_epochs)
			TRAINING_ARGS="$TRAINING_ARGS $1 $2"
			shift 2
			;;
		*)
			echo "Unknown option: $1"
			echo "Use --help for usage information"
			exit 1
			;;
	esac
done

source .slurm

# Create logs directory if it doesn't exist
mkdir -p $LOGS

echo "========================================="
echo "Running Full Pipeline"
echo "========================================="
echo "Execution Mode: $([ "$LOCAL_MODE" = true ] && echo "LOCAL" || echo "SLURM")"
echo "Dataset Arguments: $DATASET_ARGS"
echo "Training Arguments: $TRAINING_ARGS"
echo "========================================="

if [ "$LOCAL_MODE" = true ]; then
    # LOCAL EXECUTION
    echo ""
    echo "Step 1: Creating Dataset..."
    echo "========================================="
    
    bash create_dataset.sh $DATASET_ARGS
    DATASET_EXIT=$?
    
    if [ $DATASET_EXIT -ne 0 ]; then
        echo "ERROR: Dataset creation failed with exit code $DATASET_EXIT"
        exit 1
    fi
    
    echo ""
    echo "========================================="
    echo "Step 2: Running Training..."
    echo "========================================="
    
    bash run_training.sh $TRAINING_ARGS
    TRAINING_EXIT=$?
    
    if [ $TRAINING_EXIT -ne 0 ]; then
        echo "ERROR: Training failed with exit code $TRAINING_EXIT"
        exit 1
    fi
    
    echo ""
    echo "========================================="
    echo "Pipeline Completed Successfully!"
    echo "========================================="
    
else
    # SLURM EXECUTION
    echo ""
    echo "Submitting dataset creation job..."
    
    DATASET_JOB=$(sbatch --parsable \
            --output=${LOGS}/${EXPERIMENT_NAME}_dataset_%j.out \
            --error=${LOGS}/${EXPERIMENT_NAME}_dataset_%j.err \
            --job-name=${EXPERIMENT_NAME}_dataset \
            --mail-type=BEGIN,END,FAIL \
            --mail-user=${USER_EMAIL} \
            create_dataset.sh $DATASET_ARGS
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
            --job-name=${EXPERIMENT_NAME}_training \
            --mail-type=BEGIN,END,FAIL \
            --mail-user=${USER_EMAIL} \
            run_training.sh $TRAINING_ARGS
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
    echo "Monitor jobs:"
    echo "  squeue -u $USER"
    echo ""
    echo "View logs:"
    echo "  tail -f ${LOGS}/${EXPERIMENT_NAME}_dataset_${DATASET_JOB}.out"
    echo "  tail -f ${LOGS}/${EXPERIMENT_NAME}_training_${TRAIN_JOB}.out"
    echo ""
    echo "View errors:"
    echo "  tail -f ${LOGS}/${EXPERIMENT_NAME}_dataset_${DATASET_JOB}.err"
    echo "  tail -f ${LOGS}/${EXPERIMENT_NAME}_training_${TRAIN_JOB}.err"
    echo ""
    echo "Cancel jobs:"
    echo "  scancel $DATASET_JOB $TRAIN_JOB"
    echo ""
    echo "========================================="
fi
