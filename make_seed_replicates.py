#!/usr/bin/env python
"""Write the seed-replicate configs for R1 and R2 (stage 1), and PROVE that the
only thing that differs from the original run is the training seed.

    cd $WORK_LUKAS/distillkit
    python make_seed_replicates.py            # writes 4 YAMLs + 4 sbatch files
    python make_seed_replicates.py --seeds 43 # only seed 43

Why derive instead of hand-writing: the R1 and R2 YAMLs that actually ran are
committed in this repo.  Copying them by text substitution keeps every comment
and every value, and the check at the end re-loads both files and fails if
anything other than project_name / output_path / training_args.seed differs.

What `training_args.seed` changes, and what it does not
-------------------------------------------------------
* `dataset.seed: 42` stays 42.  It decides WHICH 500,000 documents form the
  pool and WHICH 200 held-out documents are evaluated (main.py, _load_dataset).
  So the pool, the eval set and the tokenize/pack cache are R1's exactly.
* `training_args.seed` goes into trl.SFTConfig -> HF Trainer.  R1-R4 had no
  such key, so they ran with the Trainer default, 42.  It seeds the sampler, so
  a replicate draws a different random ~77 % of the same packed pool
  (epoch 0.7655) in a different order.
* No random initialisation happens at train time: the student's new parameters
  were drawn once, at build time (seed 0), and are loaded from disk.  So these
  replicates measure DATA-ORDER variance (plus GPU non-determinism), not
  initialisation variance.  Say so in the thesis.
"""
import argparse
import os
import re
import sys

import yaml

W = "/leonardo_work/EUHPC_D34_049/WORK_LUKAS"
BASES = {                                   # committed YAML -> short tag
    "r1_stage1_4node.yaml":     "r1",
    "r2_stage1_gdn_4node.yaml": "r2",
}
ALLOWED = {"project_name", "output_path", "training_args.seed"}

SBATCH = r"""#!/bin/bash
#SBATCH -A EUHPC_D34_049
#SBATCH -J {name}
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH -o {W}/distillkit/logs/%x-%j.out
#SBATCH -e {W}/distillkit/logs/%x-%j.err

# SEED REPLICATE of {base_run}: identical config, training_args.seed = {seed}.
# Submit:  sbatch {sbatch_name}
# Pre-registered checks (read them in the .out before trusting the run):
#   * Froze 147 tensors (x16), one train_runtime, held-out curve monotone
#   * the FIRST logged distillation microbatch must DIFFER from {base_run}'s
#     (R1: CE 7.986 / KL 9.286 / hs 0.844).  Identical to 3 decimals means the
#     seed did not reach the sampler -- the run is a duplicate, kill it.

source /leonardo/home/userexternal/lrosenbe/distill_env_cu126/bin/activate
export WORK_LUKAS={W}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
export GPUS_PER_NODE=4
export TOTAL_PROCS=$((SLURM_NNODES * GPUS_PER_NODE))

export DS_CONFIG=$WORK_LUKAS/distillkit/ds_config_zero2.json
export RUN_CONFIG={yaml_name}
export STUDENT={student}
export OUTDIR={outdir}

export HF_HOME=$WORK_LUKAS/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export WANDB_DIR=$WORK_LUKAS/wandb
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

cd $WORK_LUKAS/distillkit

YAML_OUT=$(grep -m1 '^output_path:' $RUN_CONFIG | awk '{{print $2}}')
if [ "$YAML_OUT" != "$OUTDIR" ]; then
  echo "FATAL: yaml output_path='$YAML_OUT' != OUTDIR='$OUTDIR'"; exit 1
fi
if [ -e "$OUTDIR" ]; then
  echo "FATAL: $OUTDIR already exists -- refusing to overwrite"; exit 1
fi
YAML_MODEL=$(grep -m1 '^model:' $RUN_CONFIG | awk '{{print $2}}')
if [ "$YAML_MODEL" != "$STUDENT" ]; then
  echo "FATAL: yaml model='$YAML_MODEL' != STUDENT='$STUDENT'"; exit 1
fi
grep -q '^  seed: {seed}' $RUN_CONFIG || {{ echo "FATAL: training_args.seed {seed} not in $RUN_CONFIG"; exit 1; }}
echo "guards OK: $RUN_CONFIG -> $OUTDIR, student $STUDENT, seed {seed}"

echo "=== provenance ==="
git -C $WORK_LUKAS/distillkit log -1 --oneline
cat $STUDENT/build_info.json
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
print("PREWARM class:", type(m).__name__)
print("PREWARM params:", sum(p.numel() for p in m.parameters()))
PY

srun bash -c '
  export TRITON_CACHE_DIR=/tmp/triton_cache_${{SLURM_JOB_ID}}_${{SLURM_PROCID}}
  export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${{SLURM_JOB_ID}}_${{SLURM_PROCID}}
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
"""


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{p}{k}."))
        else:
            out[f"{p}{k}"] = v
    return out


def sub1(pattern, repl, text):
    new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"FATAL: pattern {pattern!r} not found exactly once")
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[43, 44])
    a = ap.parse_args()
    if 42 in a.seeds:
        sys.exit("FATAL: 42 is the original runs' seed")

    cmds = []
    for base, tag in BASES.items():
        if not os.path.exists(base):
            sys.exit(f"FATAL: {base} not in {os.getcwd()} -- run this inside distillkit/")
        text = open(base).read()
        orig = yaml.safe_load(text)
        if "seed" in orig.get("training_args", {}):
            sys.exit(f"FATAL: {base} already sets training_args.seed")
        base_run = orig["project_name"]
        for seed in a.seeds:
            name = f"{base_run}_s{seed}"
            outdir = f"{W}/output/{name}"
            yname, sname = f"{tag}_stage1_s{seed}.yaml", f"{tag}_stage1_s{seed}.sbatch"
            for f in (yname, sname):
                if os.path.exists(f):
                    sys.exit(f"FATAL: {f} exists -- delete it first if you mean to regenerate")

            new = f"# SEED REPLICATE of {base_run} ({base}): identical except\n" \
                  f"# project_name, output_path and training_args.seed = {seed}.\n" \
                  f"# Generated by make_seed_replicates.py.\n" + text
            new = sub1(r"^project_name:.*$", f"project_name: {name}", new)
            new = sub1(r"^output_path:.*$", f"output_path: {outdir}", new)
            new = sub1(r"^training_args:[ \t]*$",
                       f"training_args:\n  seed: {seed}                         "
                       f"# data ORDER only; dataset.seed (pool + eval docs) stays 42", new)

            # ---- the proof: only the three allowed keys differ ----------------
            f0, f1 = flat(orig), flat(yaml.safe_load(new))
            diff = {k for k in set(f0) | set(f1) if f0.get(k) != f1.get(k)}
            if diff != ALLOWED:
                sys.exit(f"FATAL: {yname} differs from {base} in {sorted(diff)}, "
                         f"expected exactly {sorted(ALLOWED)}")
            assert f1["dataset.seed"] == 42 and f1["training_args.seed"] == seed

            open(yname, "w").write(new)
            open(sname, "w").write(SBATCH.format(
                W=W, name=name, base_run=base_run, seed=seed, yaml_name=yname,
                sbatch_name=sname, student=f1["model"], outdir=outdir))
            print(f"wrote {yname} + {sname}   (differs from {base} only in "
                  f"{', '.join(sorted(diff))}; student {os.path.basename(f1['model'])})")
            cmds.append(f"sbatch {sname}")

    print("\nnext:\n  for f in r[12]_stage1_s*.sbatch; do bash -n $f && echo \"$f ok\"; done")
    print("  git add make_seed_replicates.py r[12]_stage1_s*.yaml r[12]_stage1_s*.sbatch")
    print("  git commit -m 'Stage-1 seed replicates of R1 and R2'")
    print("  " + "; ".join(cmds))


if __name__ == "__main__":
    main()