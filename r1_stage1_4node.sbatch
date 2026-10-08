#!/bin/bash
#SBATCH -A EUHPC_D34_049
#SBATCH -J r1_stage1_hybrid31
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH -o /leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/logs/%x-%j.out
#SBATCH -e /leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/logs/%x-%j.err

source /leonardo/home/userexternal/lrosenbe/distill_env_cu126/bin/activate

export WORK_LUKAS=/leonardo_work/EUHPC_D34_049/WORK_LUKAS

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export GPUS_PER_NODE=4
export TOTAL_PROCS=$((SLURM_NNODES * GPUS_PER_NODE))   # NOT "WORLD_SIZE"

export DS_CONFIG=$WORK_LUKAS/distillkit/ds_config_zero2.json
export RUN_CONFIG=r1_stage1_4node.yaml
export STUDENT=$WORK_LUKAS/students/afm_kda_hybrid31_init

export HF_HOME=$WORK_LUKAS/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR=$WORK_LUKAS/wandb
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

cd $WORK_LUKAS/distillkit

echo "=== provenance ==="
git -C $WORK_LUKAS/distillkit log -1 --oneline          # the TRAINER: repo hash is correct here
echo "--- model (the code that will RUN, not the repo) ---"
cat $STUDENT/build_info.json
sha1sum $STUDENT/*.py
echo "=================="

# versions + pre-warm the remote-code module cache, single process
python - <<'PY'
import os, torch, transformers
from importlib.metadata import version
from transformers import AutoModelForCausalLM
print("VERSIONS torch", torch.__version__,
      "| transformers", transformers.__version__,
      "| fla", version("flash-linear-attention"),
      "| deepspeed", version("deepspeed"))
m = AutoModelForCausalLM.from_pretrained(
    os.environ["STUDENT"], trust_remote_code=True, dtype=torch.bfloat16)
print("PREWARM class:", type(m).__name__)
print("PREWARM params:", sum(p.numel() for p in m.parameters()))
PY

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