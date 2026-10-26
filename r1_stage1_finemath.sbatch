#!/bin/bash
#SBATCH -A EUHPC_D34_049
#SBATCH -J r1_stage1_hybrid31_finemath
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH -o /leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/logs/%x-%j.out
#SBATCH -e /leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/logs/%x-%j.err

# FineMath STAGE 1 = R1 with only the training corpus changed (r1_stage1_finemath.yaml).
# Submit from $WORK_LUKAS/distillkit:   sbatch r1_stage1_finemath.sbatch
# Pre-registered checks (read them in the .out before trusting the run):
#   * Froze 147 tensors (x16), one train_runtime, 4580 steps, held-out curve monotone
#   * step-0 CE ~7.9 (the init student, as R1), no "roll back to CPU", no NaN
#   * tokenizing the 352 k-doc split must finish well inside the 600 s barrier
#     (estimate ~2 min; FineWeb's 500 k docs took 106 s)

source /leonardo/home/userexternal/lrosenbe/distill_env_cu126/bin/activate
export WORK_LUKAS=/leonardo_work/EUHPC_D34_049/WORK_LUKAS

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export GPUS_PER_NODE=4
export TOTAL_PROCS=$((SLURM_NNODES * GPUS_PER_NODE))

export DS_CONFIG=$WORK_LUKAS/distillkit/ds_config_zero2.json
export RUN_CONFIG=r1_stage1_finemath.yaml
export STUDENT=$WORK_LUKAS/students/afm_kda_hybrid31_init
export OUTDIR=$WORK_LUKAS/output/r1_stage1_hybrid31_finemath

export HF_HOME=$WORK_LUKAS/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR=$WORK_LUKAS/wandb
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

cd $WORK_LUKAS/distillkit

YAML_OUT=$(grep -m1 '^output_path:' $RUN_CONFIG | awk '{print $2}')
if [ "$YAML_OUT" != "$OUTDIR" ]; then
  echo "FATAL: yaml output_path='$YAML_OUT' != OUTDIR='$OUTDIR'"; exit 1
fi
if [ -e "$OUTDIR" ]; then
  echo "FATAL: $OUTDIR already exists -- refusing to overwrite"; exit 1
fi
YAML_MODEL=$(grep -m1 '^model:' $RUN_CONFIG | awk '{print $2}')
if [ "$YAML_MODEL" != "$STUDENT" ]; then
  echo "FATAL: yaml model='$YAML_MODEL' != STUDENT='$STUDENT'"; exit 1
fi
[ -f $WORK_LUKAS/datasets/finemath_4plus_stage1/dataset_info.json ] || { echo "FATAL: no finemath_4plus_stage1"; exit 1; }
echo "guards OK: $RUN_CONFIG -> $OUTDIR, student $STUDENT"

echo "=== provenance ==="
git -C $WORK_LUKAS/distillkit log -1 --oneline
cat $STUDENT/build_info.json
sha1sum $STUDENT/*.py
echo "==================="

AFM_SHA=$(git -C $WORK_LUKAS/afm_kda rev-parse HEAD)
AFM_DIRTY=$(git -C $WORK_LUKAS/afm_kda status --porcelain)

python - <<'PY'
import os, torch, transformers
from importlib.metadata import version
from transformers import AutoModelForCausalLM
print("VERSIONS torch", torch.__version__, "| transformers", transformers.__version__,
      "| fla", version("flash-linear-attention"), "| deepspeed", version("deepspeed"))
m = AutoModelForCausalLM.from_pretrained(os.environ["STUDENT"], trust_remote_code=True, dtype=torch.bfloat16)
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
SRUN_RC=$?

if [ $SRUN_RC -ne 0 ]; then
  echo "FAILED: srun exited $SRUN_RC -- not stamping $OUTDIR"
  exit $SRUN_RC
fi

echo "$AFM_SHA"          > $OUTDIR/afm_kda_commit.txt
printf '%s' "$AFM_DIRTY" > $OUTDIR/afm_kda_dirty.txt
echo "DONE: stamped $OUTDIR"