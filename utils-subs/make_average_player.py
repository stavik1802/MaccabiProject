# #!/usr/bin/env python3
# import os
# import re
# import glob
# import argparse
# from pathlib import Path
# from typing import Dict, List, Iterable, Tuple

# import pandas as pd

# # =========================
# # Position groups (edit if needed)
# # =========================
# POSITION_GROUPS: Dict[str, List[str]] = {
#     "group1_CB": ["CB"],
#     "group2_DM_CM": ["DM", "CM"],
#     "group3_FB_W": ["RB", "RM", "RW", "LB", "LM", "LW", "UNK"],
#     "group4_AM_CF": ["AM", "CF"],
# }

# # Heuristics to classify columns by name
# SUM_LIKE_PAT  = re.compile(r".*(_sum|_count)$", re.IGNORECASE)  # cumulative counters
# MEAN_LIKE_PAT = re.compile(r".*_mean$", re.IGNORECASE)          # per-chunk means
# EXTRA_MEAN_LIKE = {
#     "Speed (m/s)",  # add more if you have additional rate-like columns
# }

# def is_mean_like(col: str) -> bool:
#     c = col.lower()
#     return (
#         bool(MEAN_LIKE_PAT.fullmatch(col))
#         or col in EXTRA_MEAN_LIKE
#         or ("per_" in c)       # e.g., turns_per_sec
#         or c.endswith("_rate")
#     )

# # =========================
# # Utilities: find game folders and files
# # =========================
# def is_game_dir(path: str) -> bool:
#     """A directory containing merged_features_*.csv files."""
#     return os.path.isdir(path) and bool(glob.glob(os.path.join(path, "merged_features_*_*.csv")))

# def iter_game_dirs(base: str) -> Iterable[str]:
#     """Yield all game directories under base. If base itself is a game dir, yield it."""
#     if is_game_dir(base):
#         yield base
#         return
#     for name in sorted(os.listdir(base)):
#         sub = os.path.join(base, name)
#         if is_game_dir(sub):
#             yield sub

# def list_group_files_in_dir(folder: str, pos_list: List[str]) -> List[str]:
#     files: List[str] = []
#     for pos in pos_list:
#         for pat in [pos, pos.upper(), pos.lower()]:
#             files.extend(glob.glob(os.path.join(folder, f"merged_features_{pat}_*.csv")))
#     # de-duplicate
#     seen, out = set(), []
#     for f in files:
#         if f not in seen:
#             seen.add(f)
#             out.append(f)
#     return out

# def list_group_files_across_games(base: str, pos_list: List[str]) -> List[str]:
#     files: List[str] = []
#     for game_dir in iter_game_dirs(base):
#         files.extend(list_group_files_in_dir(game_dir, pos_list))
#     seen, out = set(), []
#     for f in files:
#         if f not in seen:
#             seen.add(f)
#             out.append(f)
#     return out

# # =========================
# # Core transforms
# # =========================
# def _to_increments_per_file(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Convert one player's table to per-chunk values:
#       - For *_sum / *_count*: use diff() with reset handling (negatives -> current value)
#       - Leave *_mean and other numeric cols as-is (already per-chunk means/rates)
#     Returns numeric columns only (must include 'chunk'), sorted by 'chunk'.
#     """
#     num = df.select_dtypes(include="number").copy()
#     if "chunk" not in num.columns:
#         raise ValueError("'chunk' column missing")
#     num = num.sort_values("chunk").reset_index(drop=True)

#     sum_like = [c for c in num.columns if c != "chunk" and SUM_LIKE_PAT.fullmatch(c)]
#     for c in sum_like:
#         d = num[c].diff()
#         d.iloc[0] = num[c].iloc[0]       # first increment = first value
#         d = d.where(d >= 0, num[c])      # handle counter resets (negatives)
#         num[c] = d.clip(lower=0)
#     return num

# def average_group_union_across_files(
#     files: List[str],
#     min_players: int = 1,
#     add_count_col: bool = True
# ) -> pd.DataFrame:
#     """
#     Per-chunk *average player* for a position group:
#       1) Convert each file to per-chunk values (increments for *_sum/*_count).
#       2) For each chunk, count how many files (players) have that chunk (n_group_chunk).
#       3) Sum all per-chunk values across files; divide by n_group_chunk (same denominator for all cols).
#     """
#     if not files:
#         return pd.DataFrame()

#     frames = []
#     for pid, fp in enumerate(files):
#         try:
#             raw = pd.read_csv(fp)
#             num = _to_increments_per_file(raw)
#             num["__file_id__"] = pid
#             frames.append(num)
#         except Exception as e:
#             print(f"[WARN] Skipping {fp}: {e}")

#     if not frames:
#         return pd.DataFrame()

#     concat = pd.concat(frames, ignore_index=True)

#     # how many files in the group have each chunk
#     counts = (
#         concat.groupby("chunk")["__file_id__"]
#         .nunique()
#         .rename("n_group_chunk")
#     )

#     # sum all numeric columns across files for each chunk
#     sums = (
#         concat.drop(columns="__file_id__")
#         .groupby("chunk")
#         .sum(numeric_only=True)
#     )

#     # keep only chunks with enough presence
#     valid_idx = counts[counts >= min_players].index
#     if len(valid_idx) == 0:
#         return pd.DataFrame()

#     sums   = sums.loc[valid_idx]
#     counts = counts.loc[valid_idx]

#     # divide by the per-chunk group count
#     averaged = sums.div(counts, axis=0).reset_index()
#     if add_count_col:
#         averaged = averaged.merge(counts.reset_index(), on="chunk", how="left")

#     cols = ["chunk"] + [c for c in averaged.columns if c != "chunk"]
#     return averaged[cols].sort_values("chunk").reset_index(drop=True)

# # ----- rebin per-chunk group averages to independent 5-min windows -----
# def rebin_group_avg_to_5min(
#     avg_df: pd.DataFrame,
#     chunk_seconds: int = 15,
#     window_minutes: int = 5
# ) -> pd.DataFrame:
#     """
#     From per-chunk group averages (already increments for *_sum/*_count):
#       - *_sum/*_count -> SUM over the window (because they are per-chunk increments now)
#       - *_mean/rates  -> AVERAGE over the window
#       - other numeric -> AVERAGE (fallback)

#     Returns: minute_start, minute_end, <features...>
#     """
#     if avg_df.empty:
#         return pd.DataFrame(columns=["minute_start", "minute_end"])

#     num = avg_df.select_dtypes(include="number").copy()
#     if "chunk" not in num.columns:
#         raise ValueError("avg_df must include 'chunk' column")

#     num = num.sort_values("chunk").reset_index(drop=True)

#     rows_per_block = int((window_minutes * 60) / chunk_seconds)
#     min_chunk = int(num["chunk"].min())
#     if min_chunk == 0:
#         block = (num["chunk"] // rows_per_block) + 1
#     else:
#         block = ((num["chunk"] - 1) // rows_per_block) + 1
#     num["__block__"] = block

#     # classify columns (ignore helper count column)
#     feature_cols = [c for c in num.columns if c not in ("chunk", "__block__", "n_group_chunk")]
#     sum_like_cols  = [c for c in feature_cols if SUM_LIKE_PAT.fullmatch(c)]
#     mean_like_cols = [c for c in feature_cols if is_mean_like(c)]
#     other_cols     = [c for c in feature_cols if c not in set(sum_like_cols + mean_like_cols)]

#     parts = []
#     if sum_like_cols:
#         parts.append(num.groupby("__block__")[sum_like_cols].sum())
#     if mean_like_cols:
#         parts.append(num.groupby("__block__")[mean_like_cols].mean())
#     if other_cols:
#         parts.append(num.groupby("__block__")[other_cols].mean())

#     out = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=sorted(num["__block__"].unique()))
#     out = out.reset_index().rename(columns={"__block__": "block"})

#     out["minute_start"] = (out["block"] - 1) * window_minutes
#     out["minute_end"]   = out["minute_start"] + window_minutes

#     cols = ["minute_start", "minute_end"] + [c for c in out.columns if c not in ("block","minute_start","minute_end")]
#     return out[cols].sort_values(["minute_start","minute_end"]).reset_index(drop=True)

# # =========================
# # Counts helper & CLI printer
# # =========================
# def group_chunk_counts(files: List[str]) -> pd.Series:
#     """Return a Series indexed by 'chunk' with the number of files in the group that contain that chunk."""
#     frames = []
#     for pid, fp in enumerate(files):
#         try:
#             df = pd.read_csv(fp, usecols=["chunk"])
#             tmp = df[["chunk"]].copy()
#             tmp["__file_id__"] = pid
#             frames.append(tmp)
#         except Exception as e:
#             print(f"[WARN] Skipping {fp}: {e}")

#     if not frames:
#         return pd.Series(dtype="int64", name="n_group_chunk")

#     concat = pd.concat(frames, ignore_index=True)
#     counts = (
#         concat.groupby("chunk")["__file_id__"]
#         .nunique()
#         .rename("n_group_chunk")
#     )
#     return counts

# def print_counts_for_range(base: str, group_key: str, start_chunk: int, end_chunk: int) -> pd.DataFrame:
#     pos_list = POSITION_GROUPS[group_key]
#     files = list_group_files_across_games(base, pos_list)
#     counts = group_chunk_counts(files)
#     idx = pd.Index(range(start_chunk, end_chunk + 1), name="chunk")
#     counts_range = counts.reindex(idx, fill_value=0)
#     print(f"\n=== {group_key}: n_group_chunk for chunks {start_chunk}..{end_chunk} ===")
#     print(counts_range.to_string())
#     return counts_range.reset_index()

# # =========================
# # Writers
# # =========================
# def write_group_outputs(
#     base: str,
#     out_prefix: str = "avg_allgames",
#     min_players: int = 1,
#     also_5min: bool = True,
#     chunk_seconds: int = 15,
#     window_minutes: int = 5
# ) -> List[Tuple[str, int]]:
#     """
#     For each group:
#       - Build per-chunk average (increments-based) and write <out_prefix>_<group>_perchunk.csv
#       - Optionally rebin to 5-min independent windows and write <out_prefix>_<group>_5min.csv
#     """
#     results: List[Tuple[str, int]] = []
#     print(f"=== Building group averages from base: {base} ===")
#     for group_name, pos_list in POSITION_GROUPS.items():
#         files = list_group_files_across_games(base, pos_list)
#         print(f"  -> {group_name}: found {len(files)} files across games")
#         if not files:
#             print(f"  !! Skipping {group_name}: no files")
#             continue

#         perchunk = average_group_union_across_files(files, min_players=min_players, add_count_col=True)
#         if perchunk.empty:
#             print(f"  !! Skipping {group_name}: no valid per-chunk data")
#             continue

#         out_perchunk = os.path.join(base, f"{out_prefix}_{group_name}_perchunk.csv")
#         perchunk.to_csv(out_perchunk, index=False)
#         print(f"  Saved per-chunk: {out_perchunk}  ({len(perchunk)} rows)")
#         results.append((out_perchunk, len(perchunk)))

#         if also_5min:
#             df5 = rebin_group_avg_to_5min(perchunk, chunk_seconds=chunk_seconds, window_minutes=window_minutes)
#             out_5min = os.path.join(base, f"{out_prefix}_{group_name}_5min.csv")
#             df5.to_csv(out_5min, index=False)
#             print(f"  Saved 5-min:    {out_5min}  ({len(df5)} rows)")
#             results.append((out_5min, len(df5)))

#     return results

# # =========================
# # CLI
# # =========================
# def main():
#     parser = argparse.ArgumentParser(description="Build group average players (per-chunk and 5-min independent).")
#     sub = parser.add_subparsers(dest="cmd", required=True)

#     # global-avg
#     p_glob = sub.add_parser("global-avg", help="Build per-chunk (and optional 5-min) averages per group across ALL games")
#     p_glob.add_argument("base_path", help="Base folder containing multiple game folders (or a single game folder)")
#     p_glob.add_argument("--out-prefix", default="avg_allgames", help="Prefix for output CSV filenames")
#     p_glob.add_argument("--min-players", type=int, default=1, help="Min files (players) required for a chunk to be kept")
#     p_glob.add_argument("--no-5min", action="store_true", help="Do NOT write 5-min windows")
#     p_glob.add_argument("--chunk-seconds", type=int, default=15, help="Seconds per input chunk (default: 15)")
#     p_glob.add_argument("--window-minutes", type=int, default=5, help="Minutes per output window (default: 5)")

#     # counts
#     p_counts = sub.add_parser("counts", help="Print per-chunk group counts for a chunk range")
#     p_counts.add_argument("base_path", help="Base folder containing games")
#     p_counts.add_argument("--group", required=True, choices=list(POSITION_GROUPS.keys()),
#                           help="Position group key (e.g., group3_FB_W)")
#     p_counts.add_argument("--start", type=int, default=240, help="Start chunk (inclusive)")
#     p_counts.add_argument("--end", type=int, default=320, help="End chunk (inclusive)")

#     args = parser.parse_args()

#     if args.cmd == "global-avg":
#         write_group_outputs(
#             base=args.base_path,
#             out_prefix=args.out_prefix,
#             min_players=args.min_players,
#             also_5min=(not args.no_5min),
#             chunk_seconds=args.chunk_seconds,
#             window_minutes=args.window_minutes,
#         )
#     elif args.cmd == "counts":
#         print_counts_for_range(args.base_path, args.group, args.start, args.end)

# if __name__ == "__main__":
#     main()









#!/usr/bin/env python3
import os
import re
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Iterable, Tuple, Optional

import pandas as pd

# =========================
# Position groups (edit if needed)
# =========================
POSITION_GROUPS: Dict[str, List[str]] = {
    "group1_CB": ["CB"],
    "group2_DM_CM": ["DM", "CM"],
    "group3_FB_W": ["RB", "RM", "RW", "LB", "LM", "LW", "UNK"],
    "group4_AM_CF": ["AM", "CF"],
}

# Heuristics to classify columns by name
SUM_LIKE_PAT  = re.compile(r".*(_sum|_count)$", re.IGNORECASE)  # cumulative counters
MEAN_LIKE_PAT = re.compile(r".*_mean$", re.IGNORECASE)          # per-chunk means
EXTRA_MEAN_LIKE = {
    "Speed (m/s)",  # add more if you have additional rate-like columns
}

def is_mean_like(col: str) -> bool:
    c = col.lower()
    return (
        bool(MEAN_LIKE_PAT.fullmatch(col))
        or col in EXTRA_MEAN_LIKE
        or ("per_" in c)       # e.g., turns_per_sec
        or c.endswith("_rate")
    )

# Distance column detection
DIST_CANDIDATES = [
    "inst_dist_m_sum",
    "inst_dist_m",
    "inst_distance_m_sum",
    "inst_distance_m",
    "distance_m_sum",
    "distance_m",
]
DIST_REGEX = re.compile(r"^inst.*dist.*m.*", re.IGNORECASE)

def find_dist_col(df: pd.DataFrame, prefer: Optional[str] = None) -> Optional[str]:
    if prefer and prefer in df.columns:
        return prefer
    for c in DIST_CANDIDATES:
        if c in df.columns:
            return c
    matches = [c for c in df.columns if DIST_REGEX.match(c)]
    if matches:
        sums = [c for c in matches if c.endswith("_sum")]
        return sums[0] if sums else matches[0]
    return None

# =========================
# Utilities: find game folders and files
# =========================
def is_game_dir(path: str) -> bool:
    """A directory containing merged_features_*.csv files."""
    return os.path.isdir(path) and bool(glob.glob(os.path.join(path, "merged_features_*_*.csv")))

def iter_game_dirs(base: str) -> Iterable[str]:
    """Yield all game directories under base. If base itself is a game dir, yield it."""
    if is_game_dir(base):
        yield base
        return
    for name in sorted(os.listdir(base)):
        sub = os.path.join(base, name)
        if is_game_dir(sub):
            yield sub

def list_group_files_in_dir(folder: str, pos_list: List[str]) -> List[str]:
    files: List[str] = []
    for pos in pos_list:
        for pat in [pos, pos.upper(), pos.lower()]:
            files.extend(glob.glob(os.path.join(folder, f"merged_features_{pat}_*.csv")))
    # de-duplicate
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out

def list_group_files_across_games(base: str, pos_list: List[str]) -> List[str]:
    files: List[str] = []
    for game_dir in iter_game_dirs(base):
        files.extend(list_group_files_in_dir(game_dir, pos_list))
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out

# =========================
# Core transforms
# =========================
def _to_increments_per_file(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert one player's table to per-chunk values:
      - For *_sum / *_count*: use diff() with reset handling (negatives -> current value)
      - Leave *_mean and other numeric cols as-is (already per-chunk means/rates)
    Returns numeric columns only (must include 'chunk'), sorted by 'chunk'.
    """
    num = df.select_dtypes(include="number").copy()
    if "chunk" not in num.columns:
        raise ValueError("'chunk' column missing")
    num = num.sort_values("chunk").reset_index(drop=True)

    sum_like = [c for c in num.columns if c != "chunk" and SUM_LIKE_PAT.fullmatch(c)]
    for c in sum_like:
        d = num[c].diff()
        d.iloc[0] = num[c].iloc[0]       # first increment = first value
        d = d.where(d >= 0, num[c])      # handle counter resets (negatives)
        num[c] = d.clip(lower=0)
    return num

def average_group_union_across_files(
    files: List[str],
    min_players: int = 1,
    add_count_col: bool = True,
    min_dist: float = 1.0,
    max_dist: float = 100.0,
    prefer_dist_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Per-chunk *average player* for a position group, using ONLY rows where per-chunk distance is in [min_dist, max_dist].

      1) Convert each file to per-chunk values (increments for *_sum/*_count).
      2) Detect distance column; filter rows to keep only min_dist <= dist <= max_dist.
      3) For each chunk, count how many files (players) remain after filtering (n_group_chunk).
      4) Sum per-chunk values across files; divide by n_group_chunk (same denominator for all cols).

    Files without a recognizable distance column are skipped (with a warning).
    """
    if not files:
        return pd.DataFrame()

    frames = []
    skipped_no_dist = 0
    skipped_empty_after_filter = 0

    for pid, fp in enumerate(files):
        try:
            raw = pd.read_csv(fp)
            num = _to_increments_per_file(raw)
            dist_col = find_dist_col(num, prefer=prefer_dist_col)
            if not dist_col:
                print(f"[WARN] Skipping {fp}: no distance-like column found")
                skipped_no_dist += 1
                continue

            # filter by distance range (inclusive)
            dist = pd.to_numeric(num[dist_col], errors="coerce")
            mask = dist.between(min_dist, max_dist, inclusive="both")
            num = num.loc[mask].copy()
            if num.empty:
                skipped_empty_after_filter += 1
                continue

            num["__file_id__"] = pid
            frames.append(num)
        except Exception as e:
            print(f"[WARN] Skipping {fp}: {e}")

    if not frames:
        print(f"[INFO] No data after filtering (skipped_no_dist={skipped_no_dist}, empty_after_filter={skipped_empty_after_filter})")
        return pd.DataFrame()

    concat = pd.concat(frames, ignore_index=True)

    # how many files in the group have each chunk **after filtering**
    counts = (
        concat.groupby("chunk")["__file_id__"]
        .nunique()
        .rename("n_group_chunk")
    )

    # sum all numeric columns across files for each chunk
    sums = (
        concat.drop(columns="__file_id__")
        .groupby("chunk")
        .sum(numeric_only=True)
    )

    # keep only chunks with enough presence
    valid_idx = counts[counts >= min_players].index
    if len(valid_idx) == 0:
        print("[INFO] No chunks meet min_players after filtering.")
        return pd.DataFrame()

    sums   = sums.loc[valid_idx]
    counts = counts.loc[valid_idx]

    # divide by the per-chunk group count
    averaged = sums.div(counts, axis=0).reset_index()
    if add_count_col:
        averaged = averaged.merge(counts.reset_index(), on="chunk", how="left")

    cols = ["chunk"] + [c for c in averaged.columns if c != "chunk"]
    return averaged[cols].sort_values("chunk").reset_index(drop=True)

# ----- rebin per-chunk group averages to independent 5-min windows -----
def rebin_group_avg_to_5min(
    avg_df: pd.DataFrame,
    chunk_seconds: int = 15,
    window_minutes: int = 5
) -> pd.DataFrame:
    """
    From per-chunk group averages (already increments for *_sum/*_count):
      - *_sum/*_count -> SUM over the window (because they are per-chunk increments now)
      - *_mean/rates  -> AVERAGE over the window
      - other numeric -> AVERAGE (fallback)

    Returns: minute_start, minute_end, <features...>
    """
    if avg_df.empty:
        return pd.DataFrame(columns=["minute_start", "minute_end"])

    num = avg_df.select_dtypes(include="number").copy()
    if "chunk" not in num.columns:
        raise ValueError("avg_df must include 'chunk' column")

    num = num.sort_values("chunk").reset_index(drop=True)

    rows_per_block = int((window_minutes * 60) / chunk_seconds)
    min_chunk = int(num["chunk"].min())
    if min_chunk == 0:
        block = (num["chunk"] // rows_per_block) + 1
    else:
        block = ((num["chunk"] - 1) // rows_per_block) + 1
    num["__block__"] = block

    # classify columns (ignore helper count column)
    feature_cols = [c for c in num.columns if c not in ("chunk", "__block__", "n_group_chunk")]
    sum_like_cols  = [c for c in feature_cols if SUM_LIKE_PAT.fullmatch(c)]
    mean_like_cols = [c for c in feature_cols if is_mean_like(c)]
    other_cols     = [c for c in feature_cols if c not in set(sum_like_cols + mean_like_cols)]

    parts = []
    if sum_like_cols:
        parts.append(num.groupby("__block__")[sum_like_cols].sum())
    if mean_like_cols:
        parts.append(num.groupby("__block__")[mean_like_cols].mean())
    if other_cols:
        parts.append(num.groupby("__block__")[other_cols].mean())

    out = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=sorted(num["__block__"].unique()))
    out = out.reset_index().rename(columns={"__block__": "block"})

    out["minute_start"] = (out["block"] - 1) * window_minutes
    out["minute_end"]   = out["minute_start"] + window_minutes

    cols = ["minute_start", "minute_end"] + [c for c in out.columns if c not in ("block","minute_start","minute_end")]
    return out[cols].sort_values(["minute_start","minute_end"]).reset_index(drop=True)

# =========================
# Counts helper & CLI printer (unchanged)
# =========================
def group_chunk_counts(files: List[str]) -> pd.Series:
    """Return a Series indexed by 'chunk' with the number of files in the group that contain that chunk."""
    frames = []
    for pid, fp in enumerate(files):
        try:
            df = pd.read_csv(fp, usecols=["chunk"])
            tmp = df[["chunk"]].copy()
            tmp["__file_id__"] = pid
            frames.append(tmp)
        except Exception as e:
            print(f"[WARN] Skipping {fp}: {e}")

    if not frames:
        return pd.Series(dtype="int64", name="n_group_chunk")

    concat = pd.concat(frames, ignore_index=True)
    counts = (
        concat.groupby("chunk")["__file_id__"]
        .nunique()
        .rename("n_group_chunk")
    )
    return counts

def print_counts_for_range(base: str, group_key: str, start_chunk: int, end_chunk: int) -> pd.DataFrame:
    pos_list = POSITION_GROUPS[group_key]
    files = list_group_files_across_games(base, pos_list)
    counts = group_chunk_counts(files)
    idx = pd.Index(range(start_chunk, end_chunk + 1), name="chunk")
    counts_range = counts.reindex(idx, fill_value=0)
    print(f"\n=== {group_key}: n_group_chunk for chunks {start_chunk}..{end_chunk} ===")
    print(counts_range.to_string())
    return counts_range.reset_index()

# =========================
# Writers
# =========================
def write_group_outputs(
    base: str,
    out_prefix: str = "avg_allgames",
    min_players: int = 1,
    also_5min: bool = True,
    chunk_seconds: int = 15,
    window_minutes: int = 5,
    min_dist: float = 1.0,
    max_dist: float = 100.0,
    prefer_dist_col: Optional[str] = None,
) -> List[Tuple[str, int]]:
    """
    For each group:
      - Build per-chunk average (increments-based, WITH distance filtering) and write <out_prefix>_<group>_perchunk.csv
      - Optionally rebin to 5-min independent windows and write <out_prefix>_<group>_5min.csv
    """
    results: List[Tuple[str, int]] = []
    print(f"=== Building group averages from base: {base} ===")
    for group_name, pos_list in POSITION_GROUPS.items():
        files = list_group_files_across_games(base, pos_list)
        print(f"  -> {group_name}: found {len(files)} files across games")
        if not files:
            print(f"  !! Skipping {group_name}: no files")
            continue

        perchunk = average_group_union_across_files(
            files,
            min_players=min_players,
            add_count_col=True,
            min_dist=min_dist,
            max_dist=max_dist,
            prefer_dist_col=prefer_dist_col,
        )
        if perchunk.empty:
            print(f"  !! Skipping {group_name}: no valid per-chunk data after filtering")
            continue

        out_perchunk = os.path.join(base, f"{out_prefix}_{group_name}_perchunk.csv")
        perchunk.to_csv(out_perchunk, index=False)
        print(f"  Saved per-chunk: {out_perchunk}  ({len(perchunk)} rows)")
        results.append((out_perchunk, len(perchunk)))

        if also_5min:
            df5 = rebin_group_avg_to_5min(perchunk, chunk_seconds=chunk_seconds, window_minutes=window_minutes)
            out_5min = os.path.join(base, f"{out_prefix}_{group_name}_5min.csv")
            df5.to_csv(out_5min, index=False)
            print(f"  Saved 5-min:    {out_5min}  ({len(df5)} rows)")
            results.append((out_5min, len(df5)))

    return results

# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(description="Build group average players with distance filtering (per-chunk and 5-min independent).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # global-avg
    p_glob = sub.add_parser("global-avg", help="Build per-chunk (and optional 5-min) averages per group across ALL games")
    p_glob.add_argument("base_path", help="Base folder containing multiple game folders (or a single game folder)")
    p_glob.add_argument("--out-prefix", default="avg_allgames", help="Prefix for output CSV filenames")
    p_glob.add_argument("--min-players", type=int, default=1, help="Min files (players) required for a chunk to be kept")
    p_glob.add_argument("--no-5min", action="store_true", help="Do NOT write 5-min windows")
    p_glob.add_argument("--chunk-seconds", type=int, default=15, help="Seconds per input chunk (default: 15)")
    p_glob.add_argument("--window-minutes", type=int, default=5, help="Minutes per output window (default: 5)")
    p_glob.add_argument("--min-dist", type=float, default=15.0, help="Minimum per-chunk distance to include (inclusive)")
    p_glob.add_argument("--max-dist", type=float, default=100.0, help="Maximum per-chunk distance to include (inclusive)")
    p_glob.add_argument("--prefer-dist-col", default=None, help="Exact distance column to prefer (e.g., inst_dist_m_sum)")

    # counts (unchanged utility)
    p_counts = sub.add_parser("counts", help="Print per-chunk group counts for a chunk range")
    p_counts.add_argument("base_path", help="Base folder containing games")
    p_counts.add_argument("--group", required=True, choices=list(POSITION_GROUPS.keys()),
                          help="Position group key (e.g., group3_FB_W)")
    p_counts.add_argument("--start", type=int, default=240, help="Start chunk (inclusive)")
    p_counts.add_argument("--end", type=int, default=320, help="End chunk (inclusive)")

    args = parser.parse_args()

    if args.cmd == "global-avg":
        write_group_outputs(
            base=args.base_path,
            out_prefix=args.out_prefix,
            min_players=args.min_players,
            also_5min=(not args.no_5min),
            chunk_seconds=args.chunk_seconds,
            window_minutes=args.window_minutes,
            min_dist=args.min_dist,
            max_dist=args.max_dist,
            prefer_dist_col=args.prefer_dist_col,
        )
    elif args.cmd == "counts":
        print_counts_for_range(args.base_path, args.group, args.start, args.end)

if __name__ == "__main__":
    main()
