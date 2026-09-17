#!/bin/bash
#SBATCH -A EUHPC_D34_049
#SBATCH -J scale_2node_4gas
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=2                   # <-- the only line you change to scale
#SBATCH --ntasks-per-node=1         # ONE task per node. accelerate spawns the GPUs.
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH -o /leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/logs/%x-%j.out
#SBATCH -e /leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/logs/%x-%j.err

source /leonardo/home/userexternal/lrosenbe/distill_env_cu126/bin/activate

export WORK_LUKAS=/leonardo_work/EUHPC_D34_049/WORK_LUKAS

# --- rendezvous: every node must agree on one address and port ---
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export GPUS_PER_NODE=4
export TOTAL_PROCS=$((SLURM_NNODES * GPUS_PER_NODE))

export DS_CONFIG=$WORK_LUKAS/distillkit/ds_config_zero2.json

export RUN_CONFIG=scaling_test.yaml

export HF_HOME=$WORK_LUKAS/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1

export WANDB_MODE=offline
export WANDB_DIR=$WORK_LUKAS/wandb

export OMP_NUM_THREADS=8 ## 32 cores ÷ 4 processes per node
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

export NCCL_DEBUG=INFO              # FIRST multi-node run only; remove afterwards

cd $WORK_LUKAS/distillkit
echo "master=$MASTER_ADDR nodes=$SLURM_NNODES world=$WORLD_SIZE"

# per-host GPU log -- otherwise the nodes overwrite each other
(
  while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader 2>/dev/null || echo "$(date) NVML_ERROR"
    sleep 5
  done
) > $WORK_LUKAS/distillkit/logs/gpu_${SLURM_JOB_ID}_$(hostname).log &

srun bash -c '
  export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}_${SLURM_PROCID}
  export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${SLURM_JOB_ID}_${SLURM_PROCID}
  accelerate launch \
    --use_deepspeed \
    --deepspeed_multinode_launcher standard \
    --deepspeed_config_file $DS_CONFIG \
    --num_machines $SLURM_NNODES \
    --num_processes $TOTAL_PROCS \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    $(which distillkit) -v $RUN_CONFIG
'