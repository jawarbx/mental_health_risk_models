#!/bin/bash
# Called by wandb agent with hyperparams as CLI args.
# Re-launches src/training.py via accelerate so DDP works correctly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.slurm"

if [[ -z "${VENV:-}" ]]; then
    echo "ERROR: VENV is not set in .slurm"
    exit 1
fi

source "$VENV/bin/activate"

export HF_HOME
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export ACCELERATE_USE_TORCH_COMPILE=0
export NCCL_SOCKET_IFNAME=ib0

MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
MASTER_PORT=29500
NUM_MACHINES=$SLURM_JOB_NUM_NODES
NUM_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')

echo "[sweep_launcher] args: $*"
echo "[sweep_launcher] MASTER_ADDR=$MASTER_ADDR NUM_MACHINES=$NUM_MACHINES NUM_GPUS=$NUM_GPUS"

# srun fans out across the job's existing allocation (both nodes)
srun bash -c "
    source ${VENV}/bin/activate
    export MASTER_ADDR=${MASTER_ADDR}
    export MASTER_PORT=${MASTER_PORT}
    export HF_HOME=${HF_HOME}
    export TOKENIZERS_PARALLELISM=false
    export TORCHINDUCTOR_DISABLE=1
    export TORCHDYNAMO_DISABLE=1
    export ACCELERATE_USE_TORCH_COMPILE=0
    export NCCL_SOCKET_IFNAME=ib0
    echo \"[\$(hostname)] machine_rank=\$SLURM_NODEID launching accelerate\"
    accelerate launch \
        --num_machines=${NUM_MACHINES} \
        --num_processes=$((NUM_MACHINES * NUM_GPUS)) \
        --machine_rank=\$SLURM_NODEID \
        --main_process_ip=${MASTER_ADDR} \
        --main_process_port=${MASTER_PORT} \
        --mixed_precision=bf16 \
        src/training.py $*
"
