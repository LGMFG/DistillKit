#!/bin/bash
#SBATCH --account=EUHPC_D34_049
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --qos=normal
#SBATCH --time=23:59:00
#SBATCH -p boost_usr_prod
#SBATCH -J tulu3_math_full_4B_instr_06B_base_sl4096_bs2_gacc2_lr1e-4_mgn3_ce0.3_kl0.7_t2_4gpu
#SBATCH -o /leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/logs/%x-%j.out
#SBATCH -e /leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/logs/%x-%j.err

# Activate your env
#source /leonardo_work/EUHPC_D34_049/WORK_LUKAS/distill_env/bin/activate
source /leonardo/home/userexternal/lrosenbe/distill_env_cu126/bin/activate

# CRITICAL: HuggingFace offline setup
export HF_HOME=$WORK/WORK_LUKAS/hf_cache

export HF_DATASETS_CACHE=$WORK/WORK_LUKAS/hf_cache/datasets
export TRANSFORMERS_CACHE=$WORK/WORK_LUKAS/hf_cache/transformers

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# performance optimizations
export TORCH_ALLOW_TF32=1 # Allow TF32 for faster training on Ampere+ GPUs. TF32 is a math mode that provides faster computation with some precision trade-offs, ideal for training large models. You can disable it if you need full precision.
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 
#export TRANSFORMERS_NO_FLASH_ATTN=1 # Disable flash attention for compatibility

# possibly faster preprocessing
export TOKENIZERS_PARALLELISM=true

export OMP_NUM_THREADS=8 # Set the number of threads for OpenMP to match the number of CPU cores allocated. This can speed up data loading and preprocessing.
export MKL_NUM_THREADS=8

# Go to distillkit repo
cd /leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit

# -------------------------------
# GPU monitoring (background)
# -------------------------------
(
  while true; do
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total \
        --format=csv,noheader 2>/dev/null || \
        echo "$(date) NVML_ERROR"
    else
      echo "$(date) nvidia-smi not available"
    fi
    sleep 60
  done
) > /leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/logs/gpu_${SLURM_JOB_ID}.log &

# Run DistillKit with accelerate launch
srun accelerate launch \
  --num_processes 4 \
  --num_machines 1 \
  --mixed_precision bf16 \
  $(which distillkit) test_config_4B_instruct.yaml