#!/usr/bin/env python3
# Build a single output tree that contains originals AND perturbed copies together.

import os
import shutil
import numpy as np
import pandas as pd

# ===== Config =====
INPUT_ROOT   = "games_train_new"           # your current root
OUTPUT_ROOT  = "games_train_mixed"         # new main folder to create
PERT_SUFFIX  = "_perturbed"                # filename suffix for perturbed copies
NOISE_LOW, NOISE_HIGH = 0.9, 1.1           # multiplicative noise range
EXCLUDE_NUMERIC_COLS = {"chunk", "minute"} # don't perturb these
SEED = 42                                   # set None for non-deterministic

# ==================
if SEED is not None:
    np.random.seed(SEED)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def perturb_csv(src_csv: str, dst_csv: str):
    df = pd.read_csv(src_csv)
    pert = df.copy()

    # numeric columns except exclusions
    num_cols = pert.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in EXCLUDE_NUMERIC_COLS]

    if num_cols:
        factors = np.random.uniform(NOISE_LOW, NOISE_HIGH, size=(len(pert), len(num_cols)))
        pert[num_cols] = pert[num_cols].to_numpy(dtype=float) * factors

    pert.to_csv(dst_csv, index=False)

def main():
    for dirpath, _, filenames in os.walk(INPUT_ROOT):
        rel = os.path.relpath(dirpath, INPUT_ROOT)
        out_dir = os.path.join(OUTPUT_ROOT, rel if rel != "." else "")
        ensure_dir(out_dir)

        for fname in filenames:
            src = os.path.join(dirpath, fname)
            base, ext = os.path.splitext(fname)

            # 1) Copy original file as-is
            dst_orig = os.path.join(out_dir, fname)
            ensure_dir(out_dir)
            shutil.copy2(src, dst_orig)

            # 2) If CSV, also write a perturbed sibling next to it
            if ext.lower() == ".csv":
                dst_pert = os.path.join(out_dir, f"{base}{PERT_SUFFIX}{ext}")
                try:
                    perturb_csv(src, dst_pert)
                except Exception as e:
                    print(f"⚠️ Skipping perturbation for {src}: {e}")

    print(f"✅ Done. Originals and perturbed copies are together under: {OUTPUT_ROOT}/")

if __name__ == "__main__":
    main()
