#!/bin/bash
#SBATCH -A EUHPC_D34_049
#SBATCH -J r6_stage2_kda_headgate_hybrid31_unfreeze
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH -o /leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/logs/%x-%j.out
#SBATCH -e /leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/logs/%x-%j.err
 
# O2 / R6 stage 2 = claude/r1_stage2.sbatch (R1-S2 recipe), started from R6's stage-1
# checkpoint-4580.  Same guards, provenance and stamping; the final check confirms
# checkpoint-15260 exists, which the stage-2 eval job of the chain starts from.
source /leonardo/home/userexternal/lrosenbe/distill_env_cu126/bin/activate
export WORK_LUKAS=/leonardo_work/EUHPC_D34_049/WORK_LUKAS
 
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export GPUS_PER_NODE=4
export TOTAL_PROCS=$((SLURM_NNODES * GPUS_PER_NODE))
 
export DS_CONFIG=$WORK_LUKAS/distillkit/ds_config_zero2.json
export RUN_CONFIG=o2_stage2.yaml
export STUDENT=$WORK_LUKAS/output/r6_stage1_kda_headgate_hybrid31/checkpoint-4580
OUTDIR=$WORK_LUKAS/output/r6_stage2_kda_headgate_hybrid31_unfreeze
 
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
YAML_MODEL=$(grep -m1 '^model:' $RUN_CONFIG | awk '{print $2}')
[ "$YAML_OUT" = "$OUTDIR" ]   || { echo "FATAL: yaml output_path '$YAML_OUT' != $OUTDIR"; exit 1; }
[ "$YAML_MODEL" = "$STUDENT" ] || { echo "FATAL: yaml model '$YAML_MODEL' != $STUDENT"; exit 1; }
[ -e "$OUTDIR" ] && { echo "FATAL: $OUTDIR already exists -- refusing to overwrite"; exit 1; }
[ -f "$STUDENT/config.json" ] || { echo "FATAL: no config.json in $STUDENT"; exit 1; }
grep -q '"kda_forget_gate": "head"' $STUDENT/config.json || { echo "FATAL: $STUDENT is not a head-gate student"; exit 1; }
echo "guards OK: $RUN_CONFIG -> $OUTDIR, from $STUDENT"
 
echo "=== provenance ==="
git -C $WORK_LUKAS/distillkit log -1 --oneline
python -c "import json;print('start global_step', json.load(open('$STUDENT/trainer_state.json'))['global_step'])"
sha1sum $STUDENT/*.py
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
print("PREWARM class:", type(m).__name__, "| kda_forget_gate:", m.config.kda_forget_gate)
print("PREWARM params:", sum(p.numel() for p in m.parameters()), "(expected 4982645176)")
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
if [ $SRUN_RC -ne 0 ]; then echo "FAILED: srun exited $SRUN_RC -- not stamping"; exit $SRUN_RC; fi
CK=$OUTDIR/checkpoint-15260
[ -f "$CK/config.json" ] && [ -f "$CK/modeling_afm_kda.py" ] || { echo "FATAL: $CK incomplete -- the stage-2 evaluation cannot start"; exit 1; }
echo "$AFM_SHA"          > $OUTDIR/afm_kda_commit.txt
printf '%s' "$AFM_DIRTY" > $OUTDIR/afm_kda_dirty.txt
echo "DONE: stamped $OUTDIR; $CK ready for evaluation"
