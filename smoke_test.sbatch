#!/bin/bash
#SBATCH -A EUHPC_D34_049
#SBATCH -J smoke_afm_selfdistill
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH -o /leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/logs/%x-%j.out
#SBATCH -e /leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/logs/%x-%j.err

source /leonardo/home/userexternal/lrosenbe/distill_env_cu126/bin/activate

# ---- paths ----
export WORK_LUKAS=/leonardo_work/EUHPC_D34_049/WORK_LUKAS

# ---- HuggingFace: one cache, offline ----
export HF_HOME=$WORK_LUKAS/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1
# NOTE: TRANSFORMERS_CACHE deliberately NOT set -- deprecated in favour of HF_HOME

# ---- JIT compile caches: node-local, unique per job ----
# DeepSpeed JIT-compiles CUDA ops; Triton compiles kernels at runtime.
# Both default to a shared-filesystem path where 4 ranks race each other.
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${SLURM_JOB_ID}

# ---- Weights & Biases: no internet on compute nodes ----
export WANDB_MODE=offline
export WANDB_DIR=$WORK_LUKAS/wandb

# ---- performance ----
export TORCH_ALLOW_TF32=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export OMP_NUM_THREADS=8          # 32 cores / 4 processes
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false   # silences fork warnings; dataset_num_proc does the work

cd $WORK_LUKAS/distillkit

# ---- GPU monitoring in the background (this is where peak memory comes from) ----
(
  while true; do
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader 2>/dev/null || echo "$(date) NVML_ERROR"
    sleep 5
  done
) > $WORK_LUKAS/distillkit/logs/gpu_${SLURM_JOB_ID}.log &

srun accelerate launch \
  --use_deepspeed \
  --deepspeed_config_file $WORK_LUKAS/distillkit/ds_config_zero2.json \
  --num_processes 4 \
  --num_machines 1 \
  --main_process_port 29500 \
  $(which distillkit) -v smoke_test_hsd.yaml