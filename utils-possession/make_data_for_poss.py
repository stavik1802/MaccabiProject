"""Build independent 5‑minute windows from cumulative player data while respecting substitutions."""
import pandas as pd
from pathlib import Path
import math

def compute_player_windows(df, sub_in=None, sub_out=None, chunk_size=15, window_minutes=5):
    """
    Convert cumulative data into independent 5-min windows aligned to game clock.
    Handles:
      - Subbed-in players: time shifted so their first chunk = sub_in.
      - Subbed-out players: trim any chunks after sub_out.
      - Edge case: subs exactly on a 5-min boundary.
    """
    df = df.copy()

    # assign game time for each chunk
    df['game_minute'] = ((df['chunk'] - 1) * chunk_size) / 60.0

    # shift for subbed-in players
    if sub_in is not None:
        df['game_minute'] += sub_in

    # trim for subbed-out players
    if sub_out is not None:
        df = df[df['game_minute'] <= sub_out].reset_index(drop=True)

    if df.empty:
        return pd.DataFrame()

    # split columns: cumulative vs mean
    sum_cols = [c for c in df.columns if c not in ['chunk', 'game_minute'] and not c.endswith('_mean')]
    avg_cols = [c for c in df.columns if c.endswith('_mean')]

    window_rows = []

    # set starting window
    current_start_min = sub_in if sub_in is not None else 0

    # ✅ FIX: handle exact-boundary subs
    if sub_in is not None:
        if sub_in % window_minutes == 0:
            current_end_min = sub_in + window_minutes
        else:
            current_end_min = math.ceil(sub_in / window_minutes) * window_minutes
    else:
        current_end_min = window_minutes

    first_window = True

    while True:
        # adjust for sub_out cutting mid-window
        if sub_out is not None and current_end_min > sub_out:
            current_end_min = sub_out

        # select chunks for this window
        window_chunks = df[(df['game_minute'] >= current_start_min) & (df['game_minute'] < current_end_min)]

        if window_chunks.empty:
            break

        row = {'minute_start': current_start_min, 'minute_end': current_end_min}

        # cumulative stats: subtract baseline
        for col in sum_cols:
            end_val = window_chunks[col].iloc[-1]

            if first_window and sub_in is not None:
                # first window for subbed-in players: start from 0
                prev_val = 0
            else:
                # for all others: subtract the last cumulative value before the window
                prev_val = df.loc[df['game_minute'] < current_start_min, col].iloc[-1] if any(df['game_minute'] < current_start_min) else 0

            row[col] = end_val - prev_val

        # mean stats: weighted average
        for col in avg_cols:
            weighted_sum = (window_chunks[col] * chunk_size).sum()
            total_time = len(window_chunks) * chunk_size
            row[col] = weighted_sum / total_time if total_time > 0 else 0

        window_rows.append(row)

        # stop if sub_out is reached
        if sub_out is not None and current_end_min >= sub_out:
            break

        # move to next window
        current_start_min = current_end_min
        current_end_min += window_minutes
        first_window = False

        # stop if we've gone past the last chunk
        if current_start_min > df['game_minute'].max():
            break

    return pd.DataFrame(window_rows)


def process_game_folder(game_folder: Path, output_root: Path):
    """
    Processes all player files in a single game folder into 5-min window stats.
    """
    subs_file = game_folder / "subs.csv"
    subs_df = pd.read_csv(subs_file) if subs_file.exists() else pd.DataFrame()

    output_game_folder = output_root / game_folder.name
    output_game_folder.mkdir(parents=True, exist_ok=True)

    # copy subs.csv for reference
    if not subs_df.empty:
        subs_df.to_csv(output_game_folder / "subs.csv", index=False)

    for file in game_folder.glob("merged_features_*.csv"):
        player_code = file.stem.replace("merged_features_", "")

        # find sub in/out times
        sub_in = None
        sub_out = None
        if not subs_df.empty:
            in_match = subs_df.loc[subs_df['In Player'] == player_code]
            out_match = subs_df.loc[subs_df['Out Player'] == player_code]
            if not in_match.empty:
                sub_in = in_match['Minute'].values[0]
            if not out_match.empty:
                sub_out = out_match['Minute'].values[0]

        df = pd.read_csv(file)
        processed_df = compute_player_windows(df, sub_in=sub_in, sub_out=sub_out,window_minutes=1)

        processed_df.to_csv(output_game_folder / file.name, index=False)
        print(f"✅ Processed {file.name} for {game_folder.name}")


def main():
    games_root = Path("/home/stav.karasik/MaccabiProject/scripts/games")  # adjust this path
    output_root = games_root.parent / "1min_windows"
    output_root.mkdir(exist_ok=True)

    for game_folder in games_root.iterdir():
        if game_folder.is_dir():
            process_game_folder(game_folder, output_root)


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# Make per-file 5-minute *independent* windows from per-15s player CSVs.

# - Treat *_sum and *_count columns as CUMULATIVE:
#     -> convert to per-chunk increments (safe to resets), then SUM within 5-min blocks.
# - Treat *_mean and rate-like columns as MEANS/RATES:
#     -> AVERAGE within 5-min blocks.
# - All other numeric columns -> AVERAGE (fallback).

# Input:
#   A folder of CSV files. Each file must include a numeric 'chunk' column (0- or 1-based).
#   Each row is one 15s chunk.

# Output:
#   <input_folder>/avg_independent/<same_filename>.csv
#   Columns: minute_start, minute_end, <features...>

# Usage:
#   python make_independent_5min.py /path/to/folder \
#       --pattern "merged_features_*.csv" \
#       --chunk-seconds 15 --window-minutes 5
# """

# import argparse
# import re
# from pathlib import Path
# import numpy as np
# import pandas as pd

# # ---------- Heuristics to classify columns ----------
# SUM_LIKE_PAT  = re.compile(r".*(_sum|_count)$", re.IGNORECASE)
# MEAN_LIKE_PAT = re.compile(r".*_mean$", re.IGNORECASE)

# # Extra mean-like names that don't follow the *_mean suffix
# EXTRA_MEAN_LIKE = {
#     "Speed (m/s)",  # extend this set if you have more
# }

# def is_mean_like(col: str) -> bool:
#     c = col.lower()
#     return (
#         bool(MEAN_LIKE_PAT.fullmatch(col))
#         or col in EXTRA_MEAN_LIKE
#         or ("per_" in c)          # e.g., turns_per_sec
#         or c.endswith("_rate")    # e.g., something_rate
#     )

# # ---------- Core rebin ----------
# def rebin_player_to_independent_windows(
#     df: pd.DataFrame,
#     chunk_seconds: int = 15,
#     window_minutes: int = 5,
# ) -> pd.DataFrame:
#     """
#     Convert a single player's per-chunk table (with 'chunk' column) into
#     independent windows of length `window_minutes`, ignoring substitutions.

#     Returns columns: minute_start, minute_end, <features...>
#     """
#     if "chunk" not in df.columns:
#         raise ValueError("Input must include a 'chunk' column.")

#     # Keep numeric columns only and sort by chunk
#     num = df.select_dtypes(include="number").copy()
#     if num.empty:
#         return pd.DataFrame(columns=["minute_start", "minute_end"])
#     num = num.sort_values("chunk").reset_index(drop=True)

#     # Derive rows per block based on time resolution
#     rows_per_block = int((window_minutes * 60) / chunk_seconds)

#     # Support 0-based or 1-based chunk indexing
#     min_chunk = int(num["chunk"].min())
#     if min_chunk == 0:
#         block = (num["chunk"] // rows_per_block) + 1
#     else:
#         block = ((num["chunk"] - 1) // rows_per_block) + 1
#     num["__block__"] = block

#     # Classify columns
#     sum_like_cols  = [c for c in num.columns if c not in ("chunk","__block__") and SUM_LIKE_PAT.fullmatch(c)]
#     mean_like_cols = [c for c in num.columns if c not in ("chunk","__block__") and is_mean_like(c)]
#     other_cols     = [c for c in num.columns if c not in ("chunk","__block__") + tuple(sum_like_cols) + tuple(mean_like_cols)]

#     parts = []

#     # Cumulative -> per-chunk increments -> sum per block
#     if sum_like_cols:
#         incr = pd.DataFrame(index=num.index)
#         for c in sum_like_cols:
#             d = num[c].diff()
#             # First increment is the first value
#             d.iloc[0] = num[c].iloc[0]
#             # Handle counter resets (negative diffs) -> treat current value as increment from 0
#             d = d.where(d >= 0, num[c])
#             incr[c] = d.clip(lower=0)
#         parts.append(incr.groupby(num["__block__"]).sum())

#     # Means/rates -> average per block
#     if mean_like_cols:
#         parts.append(num.groupby("__block__")[mean_like_cols].mean())

#     # Other numeric -> average per block (fallback)
#     if other_cols:
#         parts.append(num.groupby("__block__")[other_cols].mean())

#     # Combine
#     if parts:
#         out = pd.concat(parts, axis=1)
#     else:
#         out = pd.DataFrame(index=sorted(num["__block__"].unique()))

#     out = out.reset_index().rename(columns={"__block__": "block"})

#     # Map block -> minutes; block 1 => 0–window_minutes
#     out["minute_start"] = (out["block"] - 1) * window_minutes
#     out["minute_end"]   = out["minute_start"] + window_minutes

#     # Reorder columns
#     cols = ["minute_start", "minute_end"] + [c for c in out.columns if c not in ("block","minute_start","minute_end")]
#     return out[cols].sort_values(["minute_start","minute_end"]).reset_index(drop=True)

# # ---------- Batch processing ----------
# def process_folder(
#     input_dir: Path,
#     pattern: str = "*.csv",
#     chunk_seconds: int = 15,
#     window_minutes: int = 5,
#     out_folder_name: str = "avg_independent",
#     recursive: bool = False,
# ) -> None:
#     out_dir = input_dir / out_folder_name
#     out_dir.mkdir(parents=True, exist_ok=True)

#     files = sorted(input_dir.rglob(pattern) if recursive else input_dir.glob(pattern))
#     if not files:
#         print(f"⚠️ No files matching '{pattern}' in {input_dir}")
#         return

#     print(f"Found {len(files)} file(s). Writing to: {out_dir}")
#     for fp in files:
#         try:
#             df = pd.read_csv(fp)
#             df5 = rebin_player_to_independent_windows(
#                 df, chunk_seconds=chunk_seconds, window_minutes=window_minutes
#             )
#             out_fp = out_dir / fp.name
#             df5.to_csv(out_fp, index=False)
#             print(f"✅ {fp.name} → {out_fp.name}  ({len(df5)} rows)")
#         except Exception as e:
#             print(f"❌ {fp.name}: {e}")

# # ---------- CLI ----------
# def main():
#     ap = argparse.ArgumentParser(description="Convert per-15s player CSVs to independent 5-min windows.")
#     ap.add_argument("input_dir", type=str, help="Folder containing CSV files")
#     ap.add_argument("--pattern", default="*.csv", help="Glob pattern (e.g., 'merged_features_*.csv')")
#     ap.add_argument("--chunk-seconds", type=int, default=15, help="Seconds per input row (default: 15)")
#     ap.add_argument("--window-minutes", type=int, default=5, help="Minutes per output window (default: 5)")
#     ap.add_argument("--out-name", default="avg_independent", help="Output subfolder name (default: avg_independent)")
#     ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
#     args = ap.parse_args()

#     process_folder(
#         input_dir=Path(args.input_dir),
#         pattern=args.pattern,
#         chunk_seconds=args.chunk_seconds,
#         window_minutes=args.window_minutes,
#         out_folder_name=args.out_name,
#         recursive=args.recursive,
#     )

# if __name__ == "__main__":
#     main()
