#!/bin/bash
#SBATCH -A EUHPC_D34_049
#SBATCH -J r1_stage2_kda_hybrid31_unfreeze
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH -o /leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/logs/%x-%j.out
#SBATCH -e /leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/logs/%x-%j.err

# Stage 2 on R1 = RADLADS step 2 (full-parameter KL distillation).  One script, two uses -- the config and its output dir come in
# through --export and are READ here, never assigned (an assignment would
# silently beat --export):
#
#   smoke (debug, 20 steps):
#     sbatch --qos=boost_qos_dbg --time=00:30:00 -J r1_stage2_smoke \
#       --export=ALL,RUN_CONFIG=r1_stage2_smoke.yaml,OUTDIR=$WORK_LUKAS/output/r1_stage2_smoke r1_stage2.sbatch
#   real (300 M tokens), only if the smoke exits 0:
#     sbatch --dependency=afterok:<SMOKE_JOBID> \
#       --export=ALL,RUN_CONFIG=r1_stage2_kda_hybrid31_unfreeze.yaml,OUTDIR=$WORK_LUKAS/output/r1_stage2_kda_hybrid31_unfreeze r1_stage2.sbatch
: "${RUN_CONFIG:?--export RUN_CONFIG=...yaml}"
: "${OUTDIR:?--export OUTDIR=...}"

source /leonardo/home/userexternal/lrosenbe/distill_env_cu126/bin/activate
export WORK_LUKAS=/leonardo_work/EUHPC_D34_049/WORK_LUKAS

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export GPUS_PER_NODE=4
export TOTAL_PROCS=$((SLURM_NNODES * GPUS_PER_NODE))

export DS_CONFIG=$WORK_LUKAS/distillkit/ds_config_zero2.json
export STUDENT=$WORK_LUKAS/output/r1_stage1_hybrid31/checkpoint-4580

export HF_HOME=$WORK_LUKAS/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR=$WORK_LUKAS/wandb
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

cd $WORK_LUKAS/distillkit

# ---- guards: yaml must write where we think; never over an existing run ----
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
if [ ! -f "$STUDENT/config.json" ]; then
  echo "FATAL: no config.json in $STUDENT"; exit 1
fi
echo "guards OK: $RUN_CONFIG -> $OUTDIR, starting from $STUDENT"

echo "=== provenance ==="
git -C $WORK_LUKAS/distillkit log -1 --oneline
echo "--- starting weights (R1 checkpoint-4580) and the code they carry ---"
sha1sum $STUDENT/*.py
python -c "import json;s=json.load(open('$STUDENT/trainer_state.json'));print('start global_step', s['global_step'])"
echo "=================="

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