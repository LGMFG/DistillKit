#!/bin/bash
#SBATCH -A EUHPC_D34_049
#SBATCH -J warmup_finemath_stage2
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH -o /leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/logs/%x-%j.out
#SBATCH -e /leonardo_work/EUHPC_D34_049/WORK_LUKAS/distillkit/logs/%x-%j.err

# Single-process warm-up of the FineMath stage-2 split, as done for R5 and FineWeb stage 2:
#  (1) writes the tokenize/pack cache beside datasets/finemath_4plus_stage2, so the 4-node run
#      skips that pass (otherwise rank 0 tokenizes ~1.2 M long docs while 15 ranks wait in a
#      600 s NCCL barrier -> crash);
#  (2) prints the EXACT number of packed 2048-token sequences = effective tokens of the split.
# It runs the real YAML with only output_path / max_steps / epochs / save / eval changed (none
# of them is part of the dataset cache key; dataset_num_proc 16 and the tokenizer are unchanged).
# After "Num examples" the trainer starts; an OOM or the time limit after that point is EXPECTED.
# Submit from $WORK_LUKAS/distillkit:   sbatch warmup_finemath_stage2.sbatch
source /leonardo/home/userexternal/lrosenbe/distill_env_cu126/bin/activate
export WORK_LUKAS=/leonardo_work/EUHPC_D34_049/WORK_LUKAS
export HF_HOME=$WORK_LUKAS/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export DS_CONFIG=$WORK_LUKAS/distillkit/ds_config_zero2.json
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
cd $WORK_LUKAS/distillkit

SRC=r1_stage2_finemath.yaml
TMPOUT=$WORK_LUKAS/output/_warmup_finemath_stage2_${SLURM_JOB_ID}
WARM=warmup_finemath_stage2_${SLURM_JOB_ID}.yaml
python - "$SRC" "$WARM" "$TMPOUT" <<'PY'
import sys, yaml
src, dst, out = sys.argv[1:]
c = yaml.safe_load(open(src))
c["output_path"] = out
t = c["training_args"]
t.update(max_steps=-1, num_train_epochs=1, save_strategy="no", eval_strategy="no")
assert t["dataset_num_proc"] == 16
yaml.safe_dump(c, open(dst, "w"), sort_keys=False)
print("warm-up yaml:", dst, "| data:", c["dataset"]["train_dataset"]["disk_path"])
PY

LOG=$WORK_LUKAS/distillkit/logs/warmup_finemath_stage2_${SLURM_JOB_ID}.full.log
timeout 26m accelerate launch --use_deepspeed --deepspeed_config_file $DS_CONFIG \
    --num_machines 1 --num_processes 1 $(which distillkit) -v $WARM 2>&1 | tee $LOG

echo "=================== WARM-UP SUMMARY ==================="
grep -E "Adding EOS|Tokenizing|Packing" $LOG | tail -6
grep -E "Num examples|Total optimization steps" $LOG
N=$(grep -m1 "Num examples" $LOG | grep -oE "[0-9][0-9,]*" | tail -1 | tr -d ,)
if [ -n "$N" ]; then
  python -c "n=$N; print(f'packed sequences {n:,} = {n*2048/1e9:.4f} B effective tokens; need 488,320 for 15,260 steps x 32 ->', 'OK' if n>=488320 else 'TOO SMALL')"
else
  echo "no Num-examples line: preprocessing did not finish in time -- read the log before resubmitting"
fi
ls $WORK_LUKAS/datasets/finemath_4plus_stage2 | grep -c '^cache-' | xargs echo "cache files beside the split:"
rm -rf $TMPOUT $WARM