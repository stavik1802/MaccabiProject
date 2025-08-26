#!/usr/bin/env python3
"""
Convert cumulative per-chunk CSVs into INDEPENDENT per-chunk values.

Input layout (example):
  BASE/
    2023-12-31/
      merged_features_CB_41.csv
      merged_features_DM_15.csv
      ...
    2024-01-07/
      merged_features_CB_10.csv
      ...

Output layout:
  BASE/independent_chunks/
    2023-12-31/
      merged_features_CB_41.csv   # same name, values converted to independent per-chunk
      merged_features_DM_15.csv
      ...
    2024-01-07/
      merged_features_CB_10.csv
      ...

Rules:
- Columns ending with *_sum or *_count are treated as cumulative counters.
  -> We replace them with per-chunk increments using diff(), with reset handling.
- Columns ending with *_mean, or that look like rates (contain "per_" or end with "_rate"),
  or are listed in EXTRA_MEAN_LIKE stay unchanged (already per-chunk).
- All other columns (numeric or not) are preserved as-is.
- 'chunk' column must exist; rows are sorted by 'chunk' before conversion.
"""

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd

# --- patterns to classify columns ---
SUM_LIKE_PAT  = re.compile(r".*(_sum|_count)$", re.IGNORECASE)
MEAN_LIKE_PAT = re.compile(r".*_mean$", re.IGNORECASE)
EXTRA_MEAN_LIKE = {
    "Speed (m/s)",  # add more if you have other rate-like columns without *_mean suffix
}

def is_game_dir(path: str) -> bool:
    return os.path.isdir(path) and bool(glob.glob(os.path.join(path, "merged_features_*_*.csv")))

def iter_game_dirs(base: str) -> Iterable[str]:
    """Yield all game dirs under base. If base itself is a game dir, yield it."""
    if is_game_dir(base):
        yield base
        return
    for name in sorted(os.listdir(base)):
        sub = os.path.join(base, name)
        if is_game_dir(sub):
            yield sub

def _convert_to_independent(num: pd.DataFrame, tol: float) -> pd.DataFrame:
    """
    Convert numeric columns in 'num' to independent per-chunk values:
      - *_sum / *_count -> per-chunk increments (diff with reset & jitter handling)
      - *_mean / rate-like -> unchanged
      - others -> unchanged
    Returns a new DataFrame (sorted by chunk).
    """
    if "chunk" not in num.columns:
        raise ValueError("Missing required 'chunk' column")

    num = num.sort_values("chunk").reset_index(drop=True)

    # classify
    sum_like  = [c for c in num.columns if c != "chunk" and SUM_LIKE_PAT.fullmatch(c)]
    mean_like = [
        c for c in num.columns
        if c != "chunk" and (MEAN_LIKE_PAT.fullmatch(c) or c in EXTRA_MEAN_LIKE
                             or ("per_" in c.lower()) or c.lower().endswith("_rate"))
    ]

    # convert cumulative to increments
    for c in sum_like:
        d = num[c].diff()
        d.iloc[0] = num[c].iloc[0]              # first increment = first observed value
        # treat small negative diffs (jitter) as zero; large negatives as counter reset
        d = d.mask((d < 0) & (d >= -tol), 0.0)  # small negative -> 0
        d = d.where(d >= -tol, num[c])          # big negative -> reset: take current value
        num[c] = d.clip(lower=0)

    # everything else stays as-is
    return num

def process_game_folder(game_dir: Path, out_root: Path, pattern: str, tol: float) -> int:
    """Process one game dir; return number of files written."""
    files = sorted(game_dir.glob(pattern))
    if not files:
        print(f"  (no files matching {pattern})")
        return 0

    out_dir = out_root / game_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"  [WARN] Cannot read {fp.name}: {e}")
            continue

        if "chunk" not in df.columns:
            print(f"  [WARN] Skipping {fp.name}: no 'chunk' column")
            continue

        # Ensure 'chunk' is numeric (handles float like 240.0)
        df["chunk"] = pd.to_numeric(df["chunk"], errors="coerce")
        df = df.dropna(subset=["chunk"])
        # If chunk should be integer indices, cast to int:
        # df["chunk"] = df["chunk"].astype(int)

        # Split numeric/non-numeric, but KEEP 'chunk' in both frames
        num = df.select_dtypes(include="number").copy()
        if "chunk" not in num.columns:
            # extremely rare, but ensure it's there
            num.insert(0, "chunk", df["chunk"].values)

        # Non-numeric: drop only the numeric columns EXCEPT keep 'chunk'
        num_cols = set(num.columns.tolist())
        num_cols.discard("chunk")
        nonnum = df.drop(columns=list(num_cols & set(df.columns)), errors="ignore").copy()

        # Sort and deduplicate right side by chunk to ensure 1:1 merge
        nonnum = nonnum.sort_values("chunk").drop_duplicates(subset=["chunk"], keep="last")

        # Convert cumulative numeric columns to independent increments
        try:
            num_indep = _convert_to_independent(num, tol=tol)
        except Exception as e:
            print(f"  [WARN] Skipping {fp.name}: {e}")
            continue

        # Sort left side by chunk as well
        num_indep = num_indep.sort_values("chunk")

        # 1:1 merge on 'chunk' (both sides have it now)
        out = pd.merge(
            num_indep,
            nonnum,
            on="chunk",
            how="left",
            validate="1:1",  # ensure uniqueness; switch to "m:1" only if you truly have dup chunks
        )

        # Reorder: chunk first
        cols = ["chunk"] + [c for c in out.columns if c != "chunk"]
        out = out[cols]

        out_fp = out_dir / fp.name
        out.to_csv(out_fp, index=False)
        written += 1

    return written

def main():
    ap = argparse.ArgumentParser(description="Make independent per-chunk CSVs from cumulative inputs.")
    ap.add_argument("base_path", type=str,
                    help="Folder that contains per-game subfolders (or a single game folder).")
    ap.add_argument("--pattern", default="merged_features_*_*.csv",
                    help="Glob for files inside each game folder (default: merged_features_*_*.csv)")
    ap.add_argument("--out-name", default="independent_chunks",
                    help="Name of output folder to create inside base (default: independent_chunks)")
    ap.add_argument("--neg-jitter-tol", type=float, default=1e-6,
                    help="Tolerance for tiny negative diffs before treating as reset (default: 1e-6)")
    args = ap.parse_args()

    base = Path(args.base_path)
    out_root = base / args.out_name
    out_root.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for game in iter_game_dirs(str(base)):
        print(f"\n=== Processing game: {game} ===")
        n = process_game_folder(Path(game), out_root, args.pattern, tol=args.neg_jitter_tol)
        print(f"  → wrote {n} file(s)")
        total_written += n

    print(f"\n✅ Done. Total files written: {total_written}")
    print(f"Outputs under: {out_root}")

if __name__ == "__main__":
    main()
