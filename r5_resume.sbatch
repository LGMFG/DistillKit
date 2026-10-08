#!/bin/bash
#SBATCH -A EUHPC_D34_049
#SBATCH -J r5_resume_kda_hybrid31_1ep
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH -o /leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/logs/%x-%j.out
#SBATCH -e /leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/logs/%x-%j.err

# =====================================================================
# RESUME r5_stage1_kda_hybrid31_1ep from its newest checkpoint.
#
# Differences from r5_stage1_1ep_4node.sbatch, and why:
#
#  1. The `[ -e "$OUTDIR" ]` guard is INVERTED.  do_distill() calls
#     os.makedirs(output_path, exist_ok=True) on its first line, so the
#     directory always exists by the time anything useful has happened.
#     A resume therefore REQUIRES it, and requires a checkpoint inside it.
#
#  2. The `output_path == OUTDIR` guard is KEPT, unchanged.  That is the
#     one that would have prevented the 2026-09-29 incident where a
#     resume config still carried R1's output_path and overwrote it.
#     Never remove this.
#
#  3. resume_from_checkpoint is injected into a DERIVED yaml rather than
#     hand-edited into the original.  main.py reads it from
#       config.training_args.get("resume_from_checkpoint", None)
#     so it must sit UNDER training_args, with the right indentation.
#     Generating it with the yaml module removes that failure mode.
#
#  4. trainer_state.json is checked explicitly.  Its absence is exactly
#     what killed the resume test on 2026-09-29:
#       FileNotFoundError: .../checkpoint-10/trainer_state.json
#     A checkpoint directory can exist and still be unusable.
#
# OVERRIDE: to resume from a checkpoint other than the newest,
#   sbatch --export=ALL,RESUME_CKPT=/full/path/to/checkpoint-10482 r5_resume.sbatch
# =====================================================================

source /leonardo/home/userexternal/lrosenbe/distill_env_cu126/bin/activate

export WORK_LUKAS=/leonardo_work/EUHPC_D34_049/WORK_LUKAS

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export GPUS_PER_NODE=4
export TOTAL_PROCS=$((SLURM_NNODES * GPUS_PER_NODE))

export DS_CONFIG=$WORK_LUKAS/distillkit/ds_config_zero2.json
export RUN_CONFIG=r5_stage1_1ep_4node.yaml
export RESUME_CONFIG=r5_stage1_1ep_4node.resume.yaml
export STUDENT=$WORK_LUKAS/students/afm_kda_hybrid31_init
export OUTDIR=$WORK_LUKAS/output/r5_stage1_kda_hybrid31_1ep

export HF_HOME=$WORK_LUKAS/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR=$WORK_LUKAS/wandb
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

cd $WORK_LUKAS/distillkit

# ---- guard 1: the yaml must name the directory we think it does. -------
# ---- KEEP THIS.  It is what stops a resume writing over another run. ---
YAML_OUT=$(grep -m1 '^output_path:' $RUN_CONFIG | awk '{print $2}')
if [ "$YAML_OUT" != "$OUTDIR" ]; then
  echo "FATAL: yaml output_path='$YAML_OUT' != OUTDIR='$OUTDIR'"; exit 1
fi

# ---- guard 2: INVERTED.  A resume needs the run to already exist. ------
if [ ! -d "$OUTDIR" ]; then
  echo "FATAL: $OUTDIR does not exist -- there is nothing to resume."
  echo "       Launch r5_stage1_1ep_4node.sbatch instead."
  exit 1
fi

# ---- pick the checkpoint: highest step number, or RESUME_CKPT ----------
if [ -n "${RESUME_CKPT:-}" ]; then
  CKPT="$RESUME_CKPT"
  echo "using RESUME_CKPT override"
else
  LAST=$(ls -d $OUTDIR/checkpoint-* 2>/dev/null \
         | sed 's#.*/checkpoint-##' | sort -n | tail -1)
  if [ -z "$LAST" ]; then
    echo "FATAL: no checkpoint-* directory in $OUTDIR"
    echo "       Contents:"; ls -la "$OUTDIR"
    exit 1
  fi
  CKPT="$OUTDIR/checkpoint-$LAST"
fi
export CKPT

# ---- guard 3: the checkpoint must actually be loadable -----------------
# trainer_state.json is the file whose absence broke the 09-29 resume test.
for f in trainer_state.json config.json; do
  if [ ! -f "$CKPT/$f" ]; then
    echo "FATAL: $CKPT/$f missing -- this checkpoint cannot be resumed from."
    echo "       It was probably written only partially.  Try the one before"
    echo "       it with --export=ALL,RESUME_CKPT=..."
    ls -la "$CKPT"
    exit 1
  fi
done
# DeepSpeed ZeRO-2 keeps optimiser state in a per-step subdirectory.
# Without it the weights load but the optimiser restarts from zero, which
# silently changes the run.  Warn loudly rather than fail -- a bf16 weight
# reload is still better than starting over, but it must not pass unnoticed.
if ! ls -d "$CKPT"/global_step* >/dev/null 2>&1; then
  echo "WARNING: no global_step*/ in $CKPT -- DeepSpeed optimiser state is"
  echo "         ABSENT.  The resume will reload weights but reset Adam."
  echo "         Record this; it is not a clean continuation."
fi

RESUME_STEP=$(python -c "import json,sys; print(json.load(open('$CKPT/trainer_state.json'))['global_step'])")
MAX_STEPS=$(grep -m1 '^  max_steps:' $RUN_CONFIG | awk '{print $2}')
echo "=== resume plan ==="
echo "  checkpoint : $CKPT"
echo "  resuming at: step $RESUME_STEP"
echo "  target     : step $MAX_STEPS"
echo "  remaining  : $((MAX_STEPS - RESUME_STEP)) steps"
echo "==================="
if [ "$RESUME_STEP" -ge "$MAX_STEPS" ]; then
  echo "FATAL: already at or past max_steps -- nothing to do."; exit 1
fi

# ---- build the derived config -----------------------------------------
python - <<'PY'
import os, yaml
src, dst, ckpt = os.environ["RUN_CONFIG"], os.environ["RESUME_CONFIG"], os.environ["CKPT"]
cfg = yaml.safe_load(open(src))
assert cfg["output_path"] == os.environ["OUTDIR"], \
    f"output_path {cfg['output_path']!r} != OUTDIR {os.environ['OUTDIR']!r}"
cfg["training_args"]["resume_from_checkpoint"] = ckpt
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
print(f"wrote {dst}")
print(f"  output_path             = {cfg['output_path']}")
print(f"  max_steps               = {cfg['training_args']['max_steps']}")
print(f"  resume_from_checkpoint  = {cfg['training_args']['resume_from_checkpoint']}")
PY
if [ $? -ne 0 ]; then echo "FATAL: could not build $RESUME_CONFIG"; exit 1; fi

echo "--- the config that will run ---"
grep -nE 'output_path|max_steps|save_steps|eval_steps|resume_from_checkpoint|learning_rate|num_samples' $RESUME_CONFIG
echo "--------------------------------"

echo "=== provenance ==="
git -C $WORK_LUKAS/distillkit log -1 --oneline
echo "--- model (the code that will RUN, not the repo) ---"
cat $STUDENT/build_info.json
sha1sum $STUDENT/*.py
echo "=================="

AFM_SHA=$(git -C $WORK_LUKAS/afm_kda rev-parse HEAD)
AFM_DIRTY=$(git -C $WORK_LUKAS/afm_kda status --porcelain)

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
    $(which distillkit) -v $RESUME_CONFIG
'
SRUN_RC=$?

if [ $SRUN_RC -ne 0 ]; then
  echo "FAILED: srun exited $SRUN_RC -- not stamping $OUTDIR."
  echo "        Checkpoints already in $OUTDIR are untouched; resume again"
  echo "        with this same script once the cause is understood."
  exit $SRUN_RC
fi

echo "$AFM_SHA"           > $OUTDIR/afm_kda_commit.txt
printf '%s' "$AFM_DIRTY"  > $OUTDIR/afm_kda_dirty.txt
echo "resumed from $CKPT (step $RESUME_STEP)" >> $OUTDIR/resume_history.txt
echo "DONE: stamped $OUTDIR"