#!/usr/bin/env python
"""Build a stage-2 training split with ZERO document overlap with stage 1.

R1-R4 trained on DistillKit's  load_from_disk(train).shuffle(seed=42).select(range(500000)).
This script reproduces that exact permutation and saves the NEXT 1,350,000
documents -- permutation positions 500,000 .. 1,849,999 -- as a new dataset.

v2 (fast).  v1 read 1.85 M texts through the shuffled view one random row at a
time and hashed them in one Python process: > 30 min.  v2
  * hashes the SOURCE once, in its stored order, with 16 processes,
  * reads the permutation itself (the shuffle's index table), and looks
    the hashes up by index,
  * writes the stage-2 documents in SOURCE order (sorted indices = sequential
    reads).  Order does not matter: the YAML has `seed: 42`, so DistillKit
    shuffles this split again before training.

v3.  The source contains EXACT DUPLICATE texts (v2 found 6,123 stage-2 texts
that also occur at stage-1 positions, 25 that occur in held-out).  Disjoint
positions are therefore not disjoint documents.  v3 walks the permutation from
position 500,000 on and skips any text already in stage 1, in held-out, or
already kept, until 1,350,000 clean documents are collected.

Then it PROVES the claim instead of assuming it:
  * no stage-2 text (sha1) appears among stage 1's 500,000
  * no stage-2 text appears in the held-out split
  * no text appears twice in stage 2
and writes provenance.json next to the data.  Checking by text hash rather than
by position also catches exact duplicate documents in the source.

1,350,000 docs x 784 effective tokens = 1.06 B tokens >= 15,260 steps x 65,536.
"""
import hashlib
import json
import os
import shutil
import sys
import time

import datasets
from datasets import load_from_disk

W = os.environ["WORK_LUKAS"]
SRC = f"{W}/datasets/fineweb_edu_3B_train"
HELD = f"{W}/datasets/fineweb_edu_3B_heldout"
DST = f"{W}/datasets/fineweb_edu_3B_train_stage2_nooverlap"
TMP = f"{W}/datasets/_stage2_hash_tmp"          # hash columns; deleted at the end
SEED, R1_N, N, NPROC = 42, 500_000, 1_350_000, 16
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f} s] {msg}", flush=True)


def sha(batch):
    return {"h": [hashlib.sha1(t.encode()).hexdigest() for t in batch["text"]]}


def hashes(d, name):
    h = d.map(sha, batched=True, batch_size=2000, num_proc=NPROC,
              remove_columns=d.column_names,
              cache_file_name=f"{TMP}/{name}.arrow", desc=f"hashing {name}")
    # h[:]["h"] is a real Python list.  NOT h["h"]: in datasets >= 4.0 that is a
    # lazy Column, and every hs[i] below would be a separate Arrow lookup --
    # 1.85 M of them is what stalled v2's first run after "heldout: 2,000 hashes".
    out = h[:]["h"]
    assert isinstance(out, list), type(out)
    log(f"{name}: {len(out):,} hashes")
    return out


def main():
    if os.path.exists(os.path.join(DST, "provenance.json")):
        print(f"{DST} already built -- nothing to do.")
        print(open(os.path.join(DST, "provenance.json")).read())
        sys.exit(0)
    if os.path.exists(DST):
        sys.exit(f"FATAL: {DST} exists but has no provenance.json -- a half-written run? "
                 f"Remove it (rm -rf) and rerun.")
    os.makedirs(TMP, exist_ok=True)

    log(f"datasets {datasets.__version__}")
    ds = load_from_disk(SRC)
    if hasattr(ds, "keys"):
        sys.exit("FATAL: train set is a DatasetDict -- check SRC")
    log(f"source: {len(ds):,} documents, columns {ds.column_names}")

    # EXACTLY DistillKit's _load_dataset: `if seed: res = res.shuffle(seed=seed)`
    sh = ds.shuffle(seed=SEED)
    perm = sh._indices.column("indices").to_numpy()
    assert len(perm) == len(ds), (len(perm), len(ds))
    for p in (0, 1, R1_N - 1, R1_N, R1_N + N - 1, len(ds) - 1):        # the index table IS the shuffled view
        assert sh[p]["text"] == ds[int(perm[p])]["text"], f"permutation mismatch at {p}"
    log("permutation read and spot-checked against the shuffled view")

    i1 = perm[:R1_N]
    hs = hashes(ds, "train")
    hh = hashes(load_from_disk(HELD), "heldout")
    k1, kh = {hs[i] for i in i1}, set(hh)
    log(f"stage-1 (positions 0..{R1_N - 1}): {len(k1):,} unique texts "
        f"({R1_N - len(k1):,} exact duplicates inside stage 1 itself)")
    log(f"held-out: {len(kh):,} unique texts;  ALSO in stage 1: {len(kh & k1)}  "
        f"(eval documents R1-R5 had seen in training)")

    # Walk the permutation from position R1_N on.  Keep a document only if its text
    # is new: not in stage 1, not in held-out, not already kept.  Continue until N
    # are kept -- the skipped ones are replaced by later positions, so the split
    # keeps its full size.
    kept = set()
    i2, skip = [], dict(stage1=0, heldout=0, within=0)
    pos = R1_N
    while len(i2) < N:
        if pos >= len(perm):
            sys.exit(f"FATAL: source exhausted after {len(i2):,} clean documents")
        i, h = int(perm[pos]), hs[int(perm[pos])]
        pos += 1
        if h in k1:
            skip["stage1"] += 1
        elif h in kh:
            skip["heldout"] += 1
        elif h in kept:
            skip["within"] += 1
        else:
            kept.add(h)
            i2.append(i)
    last = pos - 1
    log(f"stage-2: positions {R1_N}..{last:,}  ->  {len(i2):,} kept;  skipped {skip}")

    # Proof, recomputed from what will actually be written
    k2 = {hs[i] for i in i2}
    o1, oh = len(k2 & k1), len(k2 & kh)
    log(f"overlap stage-2 / stage-1 : {o1}")
    log(f"overlap stage-2 / held-out: {oh}")
    assert o1 == 0 and oh == 0 and len(k2) == N == len(i2)

    i2.sort()                         # source order = sequential reads; DistillKit reshuffles
    log(f"writing {N:,} documents to {DST} ...")
    stage2 = ds.select(i2)
    stage2.save_to_disk(DST)   # ONE process: v3's 16 worker processes re-ran this script and hit the DST guard
    check = load_from_disk(DST)
    assert len(check) == N, (len(check), N)
    assert hashlib.sha1(check[0]["text"].encode()).hexdigest() == hs[i2[0]]
    log("written and re-read")

    prov = dict(source=SRC, shuffle_seed=SEED, permutation_positions=[R1_N, last],
                documents=N, stored_order="source order (DistillKit's seed-42 shuffle randomises it)",
                excluded_stage1_positions=[0, R1_N - 1], id_key="sha1(text)",
                skipped_text_in_stage1=skip["stage1"], skipped_text_in_heldout=skip["heldout"],
                skipped_duplicate_within_stage2=skip["within"],
                overlap_with_stage1=o1, overlap_with_heldout=oh,
                stage1_internal_duplicates=R1_N - len(k1),
                heldout_texts_also_in_stage1=len(kh & k1),
                datasets_version=datasets.__version__, script="make_stage2_split.py v4")
    json.dump(prov, open(os.path.join(DST, "provenance.json"), "w"), indent=1)
    shutil.rmtree(TMP, ignore_errors=True)
    print(json.dumps(prov, indent=1))
    log("DONE")


# The guard matters: datasets' worker processes may re-import this file.
# Without it they re-ran the whole script (v3: 17 "FATAL ... exists" lines).
if __name__ == "__main__":
    main()