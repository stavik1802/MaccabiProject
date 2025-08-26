#!/usr/bin/env python3
"""
Re-bin 'avg_allgames_group*_*.csv' files into independent 5-minute chunks.

Assumptions:
- Input CSVs each have a numeric 'chunk' column where each row = one base chunk.
- Base chunk length is 15 seconds (change ORIG_CHUNK_SECONDS if different).
- Some columns are cumulative per chunk and end with *_sum or *_count.
  We convert those to per-row increments before summing into 5-minute blocks.

Outputs:
- For every matching file in base_path, creates a sibling file with suffix '_5min.csv'.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re
import argparse

# ---- config (change if needed) ----
ORIG_CHUNK_SECONDS = 15        # length of an input row (sec)
TARGET_MINUTES = 5             # desired block size (minutes)
ROWS_PER_BLOCK = int((TARGET_MINUTES * 60) / ORIG_CHUNK_SECONDS)  # 300/15 = 20

SUM_LIKE_PAT  = re.compile(r".*(_sum|_count)$", re.IGNORECASE)
MEAN_LIKE_PAT = re.compile(r".*_mean$", re.IGNORECASE)

def rebin_df_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Re-bin one avg_allgames dataframe to 5-minute blocks."""
    if df.empty:
        return df.copy()

    # Ensure numeric-only copy for aggregation
    num = df.select_dtypes(include="number").copy()
    if "chunk" in num.columns:
        # Sort by chunk to keep chronological order
        num = num.sort_values("chunk").reset_index(drop=True)
        block = ((num["chunk"] - 1) // ROWS_PER_BLOCK) + 1
    else:
        # Fallback to row order
        num = num.reset_index(drop=True)
        block = (np.arange(len(num)) // ROWS_PER_BLOCK) + 1

    group_key = pd.Series(block, name="block")

    sum_like_cols  = [c for c in num.columns if c != "chunk" and SUM_LIKE_PAT.fullmatch(c)]
    mean_like_cols = [c for c in num.columns if c != "chunk" and MEAN_LIKE_PAT.fullmatch(c)]
    other_cols     = [c for c in num.columns if c not in ["chunk"] + sum_like_cols + mean_like_cols]

    parts = []

    # 1) For *_sum/*_count columns, treat as cumulative -> per-row increments -> sum per 5 min
    if sum_like_cols:
        incr = pd.DataFrame(index=num.index)
        for c in sum_like_cols:
            inc = num[c].diff().fillna(num[c])
            incr[c] = inc.clip(lower=0)  # clamp negatives (resets)
        parts.append(incr.groupby(group_key).sum())

    # 2) For *_mean columns → average
    if mean_like_cols:
        parts.append(num[mean_like_cols].groupby(group_key).mean())

    # 3) For other numeric columns (e.g., speeds not tagged as *_mean) → average
    if other_cols:
        parts.append(num[other_cols].groupby(group_key).mean())

    if not parts:
        out = pd.DataFrame({"chunk": sorted(pd.unique(block))})
        out["chunk"] = np.arange(1, len(out) + 1)
        return out

    out = pd.concat(parts, axis=1).reset_index(drop=True)
    out.insert(0, "chunk", np.arange(1, len(out) + 1))  # new 5-min chunk index
    return out

def process_folder(base_path: Path, pattern: str = "avg_allgames_group*_*.csv", suffix: str = "_5min"):
    files = sorted(base_path.glob(pattern))
    if not files:
        print(f"⚠️ No files matching '{pattern}' in {base_path}")
        return

    print(f"Found {len(files)} files.")
    for fp in files:
        print(f"→ {fp.name}")
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"  ⚠️ Failed to read: {e}")
            continue

        out_df = rebin_df_to_5min(df)
        out_name = fp.with_name(fp.stem + suffix + fp.suffix)
        out_df.to_csv(out_name, index=False)
        print(f"  ✅ Wrote {out_name.name}  ({len(out_df)} rows)")

def main():
    ap = argparse.ArgumentParser(description="Re-bin avg_allgames CSVs into 5-minute chunks.")
    ap.add_argument("base_path", help="Folder containing avg_allgames_group*_*.csv files")
    ap.add_argument("--pattern", default="avg_allgames_group*_*.csv", help="Glob to match input files")
    ap.add_argument("--suffix", default="_5min", help="Suffix for output filenames (before extension)")
    ap.add_argument("--orig-chunk-seconds", type=int, default=ORIG_CHUNK_SECONDS, help="Length of input rows (sec)")
    ap.add_argument("--target-minutes", type=int, default=TARGET_MINUTES, help="Target block length (minutes)")
    args = ap.parse_args()


    process_folder(Path(args.base_path), pattern=args.pattern, suffix=args.suffix)

if __name__ == "__main__":
    main()
