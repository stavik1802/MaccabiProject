#!/usr/bin/env python3
"""
Experiment: sensitivity of recommended substitutions to perturbations of specific physical features.

Selection rule (NO 60-minute CSV used):
- Scan every subfolder in --games_root.
- Include a game iff it has EXACTLY 10 players whose merged_features_*.csv
  contain chunk==1 AND chunk==241 (15s chunks, 1-based).

Then randomly sample K games (default 6) from the valid set (or all if fewer than K).

Pipeline per selected game:
- Determine starters = those 10 players.
- Predict baseline team possession for minutes 60..75 via:
  TCN (15s) -> rebin to 5-min -> grouped CatBoost.
- For EACH starter and EACH group (G1..G4), substitute the group 5-min "average player".
- For EACH listed feature and EACH multiplier, perturb ONLY that feature on the replacement
  player and recompute the recommended group (count only positive-delta SUBs).
- Aggregate SUB counts per (feature, multiplier, group).

Outputs (under --out_dir):
- subs_counts_by_feature_multiplier.csv   (tidy totals across all selected games)
- plots/subs_counts_<feature>.png         (per-feature grouped bar chart)

Also prints:
- How many valid games were found, and which random K were used.
"""

import argparse
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tcn import TCN

# ---------------------------- Defaults you can edit ----------------------------

# Root folder of game folders (each contains merged_features_*.csv)
DEFAULT_GAMES_ROOT = "/home/stav.karasik/MaccabiProject/scripts/5min_windows"

# 5-min average player per group files directory (contains avg_allgames_group{1..4}_5min.csv or similar)
DEFAULT_AVG_GROUP_5MIN_DIR = "/home/stav.karasik/MaccabiProject/scripts/5min_windows_avg"

# Models
DEFAULT_TCN_MODEL = "saved_model/even_better_model/player_tcn_model.keras"
DEFAULT_CAT_MODEL = "possession_catboost_grouped.pkl"
DEFAULT_CAT_FEATURES = "catboost_grouped_features.pkl"

# Start minute & horizon
START_MINUTE = 60
HORIZON_MINUTES = 30  # 60..90
INPUT_WINDOW = 130    # TCN history length (must match training)

# TCN targets & alias map (must match training)
TCN_TARGET_COLS = [
    "inst_dist_m_sum", "Speed (m/s)_sum", "hsr_m_sum",
    "vha_count_1s_sum", "avg_jerk_1s_sum", "turns_per_sec_sum", "playerload_1s_sum",
    "walk_time_sum", "jog_time_sum", "run_time_sum", "sprint_time_sum",
    "total_sprints_sum", "sprint_attack_sum", "sprint_defense_sum",
    "dist_attack_sum", "dist_defense_sum", "time_attack_sum", "time_defense_sum",
    "attacking_third_time_sum", "middle_third_time_sum", "defending_third_time_sum"
]
ALIAS_MAP = {
    "Speed (m/s)_mean": "Speed (m/s)_sum",
    "avg_jerk_1s_mean": "avg_jerk_1s_sum",
}

# Features to perturb (in the replacement player's 5-min windows)
DEFAULT_PERTURB_FEATURES = [
    "inst_dist_m_sum",
    "Speed (m/s)_sum",
    "hsr_m_sum",
    "vha_count_1s_sum",
    "avg_jerk_1s_sum",
    "turns_per_sec_sum",
    "total_sprints_sum",
    "sprint_attack_sum",
    "sprint_defense_sum",
    "dist_attack_sum",
    "dist_defense_sum",
    "playerload_1s_sum",
]
DEFAULT_PERTURB_MULTIPLIERS = [1.0, 1.1, 1.2]   # include 1.0 as baseline

# EXACTLY 10 starters required (per your request)
REQUIRE_EXACT_10 = True

# ---------------------------- Mappings ----------------------------

PREFIX_TO_GROUP_ID = {
    "CB": 1,
    "CM": 2, "DM": 2,
    "LB": 3, "LM": 3, "LW": 3, "RB": 3, "RM": 3, "RW": 3, "UNK": 3,
    "AM": 4, "CF": 4,
}
GROUP_ID_TO_HUMAN = {1: "group1_CB", 2: "group2_CM_DM", 3: "group3_FB_W", 4: "group4_AM_CF"}
GID_TO_GROUPLBL = {1: "G1", 2: "G2", 3: "G3", 4: "G4"}

SUM_LIKE_PAT  = re.compile(r".*(_sum|_count)$", re.IGNORECASE)
MEAN_LIKE_PAT = re.compile(r".*_mean$", re.IGNORECASE)

# ---------------------------- Utilities ----------------------------

def get_prefix(code: str) -> str:
    return code.split("_", 1)[0]

def infer_group_id(code: str) -> int:
    return PREFIX_TO_GROUP_ID.get(get_prefix(code), 3)

def infer_group_label(code: str) -> str:
    return GID_TO_GROUPLBL[infer_group_id(code)]

def minute_blocks(start_minute: int, horizon_min: int):
    blocks = []
    for m in range(start_minute, start_minute + horizon_min, 5):
        block = (m // 5) + 1
        blocks.append((block, m, m + 5))
    return blocks

def chunk_range_for_minutes(start_minute: int, horizon_min: int):
    # 1-based 15s chunks: minute m starts at m*4 + 1  → 60 → 241
    start_chunk = start_minute * 4 + 1
    end_chunk   = start_chunk + horizon_min * 4 - 1
    return start_chunk, end_chunk

def read_csv_robust(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception as e2:
            print(f"[!] Failed to read {path}: {e2}")
            return None

def player_has_chunks_1_and_241(csv_path: Path, verbose: bool = False) -> bool:
    """
    Starter detection (1-based): True iff the file has chunk==1 AND chunk==241.
    """
    df = read_csv_robust(csv_path)
    if df is None or "chunk" not in df.columns:
        if verbose:
            print(f"    [debug] {csv_path.name}: unreadable or missing 'chunk' column")
        return False
    ch = pd.to_numeric(df["chunk"], errors="coerce").dropna().astype(int)
    has1   = (ch == 1).any()
    has241 = (ch == 241).any()
    if verbose:
        mn = int(ch.min()) if len(ch) else None
        mx = int(ch.max()) if len(ch) else None
        print(f"    [debug] {csv_path.name}: chunk[min,max]=({mn},{mx}) -> has1={has1}, has241={has241}")
    return bool(has1 and has241)

def resolve_columns(df: pd.DataFrame, wanted: List[str], alias: Dict[str, str]):
    used = []
    inv = {v: k for k, v in alias.items()}
    for c in wanted:
        if c in df.columns:
            used.append(c)
        elif c in alias and alias[c] in df.columns:
            used.append(alias[c])
        elif c in inv and inv[c] in df.columns:
            used.append(inv[c])
        else:
            raise ValueError(f"Missing feature for TCN: '{c}' (and alias).")
    return df[used].copy(), used

def autoreg_predict_15s(raw: pd.DataFrame, used_cols: List[str], model, start_chunk: int, end_chunk: int, input_window: int):
    if "chunk" not in raw.columns:
        raise ValueError("Dataframe must have a 'chunk' column.")
    df = raw.sort_values("chunk").reset_index(drop=True)
    idx_list = df.index[df["chunk"] == start_chunk].tolist()
    if not idx_list:
        raise ValueError(f"start_chunk {start_chunk} not present.")
    start_idx = idx_list[0]
    if start_idx < input_window:
        raise ValueError(f"Need {input_window} history rows before chunk {start_chunk}; have {start_idx}.")
    scaler = StandardScaler()
    feats_all = df[used_cols].to_numpy()
    feats_norm = scaler.fit_transform(feats_all)
    hist = feats_norm[start_idx - input_window:start_idx, :]
    buf = hist.copy()
    steps = (end_chunk - start_chunk) + 1
    preds_norm = []
    for _ in range(steps):
        model_input = buf[-input_window:].reshape(1, input_window, -1)
        yhat = model.predict(model_input, verbose=0)[0]
        preds_norm.append(yhat)
        next_row = buf[-1].copy()
        next_row[:len(used_cols)] = yhat
        buf = np.vstack([buf, next_row])[1:]
    preds_norm = np.array(preds_norm)
    y_pred = preds_norm * np.sqrt(scaler.var_) + scaler.mean_
    out = pd.DataFrame(y_pred, columns=used_cols)
    out.insert(0, "chunk", np.arange(start_chunk, end_chunk + 1))
    return out

def rebin_15s_to_5min(df15: pd.DataFrame) -> pd.DataFrame:
    """Re-bin 15s predictions to independent 5-min blocks (1-based chunk)."""
    if df15.empty:
        return df15.copy()
    num = df15.select_dtypes(include="number").copy()
    num = num.sort_values("chunk").reset_index(drop=True)

    # 1-based: chunks 1..20 -> block 1, 21..40 -> block 2, etc.
    block = ((num["chunk"] - 1) // 20) + 1
    group_key = pd.Series(block, name="block")

    parts = []
    sum_like_cols  = [c for c in num.columns if c != "chunk" and SUM_LIKE_PAT.fullmatch(c)]
    mean_like_cols = [c for c in num.columns if c != "chunk" and MEAN_LIKE_PAT.fullmatch(c)]
    other_cols     = [c for c in num.columns if c not in ["chunk"] + sum_like_cols + mean_like_cols]

    if sum_like_cols:
        incr = pd.DataFrame(index=num.index)
        for c in sum_like_cols:
            inc = num[c].diff().fillna(num[c])
            incr[c] = inc.clip(lower=0)
        parts.append(incr.groupby(group_key).sum())
    if mean_like_cols:
        parts.append(num[mean_like_cols].groupby(group_key).mean())
    if other_cols:
        parts.append(num[other_cols].groupby(group_key).mean())

    out = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=sorted(pd.unique(block)))
    out["block"] = out.index
    out = out.reset_index(drop=True)
    out["minute_start"] = (out["block"] - 1) * 5
    out["minute_end"]   = out["minute_start"] + 5
    cols = ["minute_start","minute_end"] + [c for c in out.columns if c not in ("block","minute_start","minute_end")]
    return out[cols]

def find_avg_file_for_group(group_dir: Path, gid: int) -> Path:
    patterns = [
        f"avg_allgames_group{gid}_5min.csv",
        f"*group{gid}*_5min.csv",
        f"*group{gid}*5min*.csv",
        f"avg_group{gid}*_5min*.csv",
    ]
    for pat in patterns:
        cands = sorted(group_dir.glob(pat))
        if cands:
            return cands[0]
    raise FileNotFoundError(f"No 5-min average file found for group {gid} in {group_dir}")

def _slice_group_file_to_blocks(gdf: pd.DataFrame, block_ids: List[int]) -> pd.DataFrame:
    """
    Accepts group avg file that has either:
      - 'chunk' (5-min block id, 1-based), or
      - 'block' (1-based), or
      - 'minute_start' (then we match minutes directly).
    """
    df = gdf.copy()
    cols = df.columns.str.lower()

    if "chunk" in cols:
        key = df.columns[cols.get_loc("chunk")]
        need = df[df[key].isin(block_ids)].copy()
        need["minute_start"] = (need[key] - 1) * 5
        need["minute_end"]   = need["minute_start"] + 5
        drop_cols = [key]
    elif "block" in cols:
        key = df.columns[cols.get_loc("block")]
        need = df[df[key].isin(block_ids)].copy()
        need["minute_start"] = (need[key] - 1) * 5
        need["minute_end"]   = need["minute_start"] + 5
        drop_cols = [key]
    elif "minute_start" in cols:
        key = df.columns[cols.get_loc("minute_start")]
        minute_starts = [(b - 1) * 5 for b in block_ids]
        need = df[df[key].isin(minute_starts)].copy()
        if "minute_end" not in df.columns:
            need["minute_end"] = need[key] + 5
        drop_cols = []
    else:
        raise ValueError("Group avg file must contain 'chunk', 'block', or 'minute_start'.")

    rem = [c for c in need.columns if c not in ["minute_start","minute_end"] + drop_cols]
    need = need[["minute_start","minute_end"] + rem]
    return need.sort_values(["minute_start","minute_end"]).reset_index(drop=True)

def load_group_avg_5min_slice(avg_dir: Path, gid: int, block_ids: List[int]) -> pd.DataFrame:
    fp = find_avg_file_for_group(avg_dir, gid)
    gdf = read_csv_robust(fp)
    if gdf is None:
        raise FileNotFoundError(f"Cannot read {fp}")
    return _slice_group_file_to_blocks(gdf, block_ids)

def build_team_features_grouped(players_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["minute_start", "minute_end", "player_id", "position", "group"]
    feature_cols = [c for c in players_df.columns if c not in key_cols]
    ALL_GROUPS = ["G1", "G2", "G3", "G4"]

    rows = []
    for (start, end), window in players_df.groupby(["minute_start", "minute_end"], sort=True):
        row = {"minute_start": start, "minute_end": end}
        for g in ALL_GROUPS:
            row[f"count_{g}"] = int((window["group"] == g).sum())
        for g in ALL_GROUPS:
            sub = window[window["group"] == g]
            if len(sub) == 0:
                for c in feature_cols:
                    row[f"{c}__{g}"] = 0.0
            else:
                means = sub[feature_cols].mean(numeric_only=True)
                for c in feature_cols:
                    row[f"{c}__{g}"] = float(means.get(c, 0.0))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["minute_start", "minute_end"]).reset_index(drop=True)

def ensure_catboost_features(df_team_grouped: pd.DataFrame, cb_features: List[str]) -> pd.DataFrame:
    out = df_team_grouped.copy()
    missing = [c for c in cb_features if c not in out.columns]
    for c in missing:
        out[c] = 0.0
    return out[cb_features].copy()

# ---------------------------- Core experiment ----------------------------

def choose_starters_for_game(game_dir: Path, verbose: bool = False) -> List[str]:
    """Return list of player codes with chunk==1 and chunk==241 present (1-based)."""
    starters = []
    files = sorted(game_dir.glob("merged_features_*.csv"))
    if verbose:
        print(f"  [debug] scanning {len(files)} player files in {game_dir.name}")
    for f in files:
        if player_has_chunks_1_and_241(f, verbose=verbose):
            starters.append(f.stem.replace("merged_features_", ""))
    if verbose:
        print(f"  [debug] starters: {starters}")
    return starters

def predict_baseline_for_game(
    game_dir: Path,
    starters: List[str],
    tcn_model,
    cb_model,
    cb_features: List[str],
    avg_dir: Path,
    start_chunk: int,
    end_chunk: int,
    block_ids: List[int],
) -> Tuple[pd.DataFrame, float]:
    """Return (baseline_players_df, baseline_avg_possession)."""
    player_windows = []
    for code in starters:
        fp = game_dir / f"merged_features_{code}.csv"
        raw = read_csv_robust(fp)
        if raw is None:
            print(f"  [!] Missing/unreadable player file: {fp.name} — skipping in baseline.")
            continue

        feats_df, used_cols = resolve_columns(raw, TCN_TARGET_COLS, ALIAS_MAP)

        df15 = autoreg_predict_15s(
            raw=pd.concat([raw[["chunk"]], feats_df], axis=1),
            used_cols=feats_df.columns.tolist(),
            model=tcn_model,
            start_chunk=start_chunk,
            end_chunk=end_chunk,
            input_window=INPUT_WINDOW,
        )
        df5 = rebin_15s_to_5min(df15)

        gid_self = infer_group_id(code)
        try:
            g5_self = load_group_avg_5min_slice(avg_dir, gid_self, block_ids)
            merged = g5_self.merge(df5, on=["minute_start","minute_end"], how="left", suffixes=("_grp",""))
            for c in df5.columns:
                if c in ("minute_start","minute_end"):
                    continue
                if c in merged.columns and (c + "_grp") in merged.columns:
                    merged[c] = merged[c].where(merged[c].notna(), merged[c + "_grp"])
                    merged.drop(columns=[c + "_grp"], inplace=True)
        except Exception:
            merged = df5.copy()

        merged["player_id"] = code
        merged["position"]  = get_prefix(code)
        merged["group"]     = infer_group_label(code)
        player_windows.append(merged)

    if not player_windows:
        raise RuntimeError("No valid players for baseline.")

    baseline_players_df = pd.concat(player_windows, ignore_index=True)
    baseline_team_grouped = build_team_features_grouped(baseline_players_df)
    baseline_X = ensure_catboost_features(baseline_team_grouped, cb_features)
    baseline_preds = cb_model.predict(baseline_X)
    baseline_avg = float(np.mean(baseline_preds))
    return baseline_players_df, baseline_avg

def simulate_replacement_with_perturbation(
    baseline_players_df: pd.DataFrame,
    target_player_code: str,
    gid: int,
    avg_dir: Path,
    block_ids: List[int],
    cb_model,
    cb_features: List[str],
    feature_to_scale: Optional[str],
    multiplier: float,
) -> Optional[float]:
    """Return scenario_avg (mean possession) or None if failed."""
    try:
        g5_rep = load_group_avg_5min_slice(avg_dir, gid, block_ids).copy()
    except Exception:
        return None

    if feature_to_scale and feature_to_scale in g5_rep.columns and abs(multiplier - 1.0) > 1e-12:
        g5_rep[feature_to_scale] = g5_rep[feature_to_scale] * multiplier

    g5_rep["player_id"] = f"group{gid}_AVG"
    g5_rep["position"]  = GROUP_ID_TO_HUMAN[gid]
    g5_rep["group"]     = GID_TO_GROUPLBL[gid]

    repl_players = []
    for pcode in baseline_players_df["player_id"].unique():
        if pcode == target_player_code:
            repl_players.append(g5_rep)
        else:
            repl_players.append(baseline_players_df[baseline_players_df["player_id"] == pcode])
    scenario_players_df = pd.concat(repl_players, ignore_index=True)

    scenario_team_grouped = build_team_features_grouped(scenario_players_df)
    scenario_X = ensure_catboost_features(scenario_team_grouped, cb_features)
    scenario_preds = cb_model.predict(scenario_X)
    return float(np.mean(scenario_preds))

def run_experiment(
    games_root: Path,
    avg_group_dir: Path,
    tcn_model_path: Path,
    cat_model_path: Path,
    cat_features_path: Path,
    perturb_features: List[str],
    perturb_multipliers: List[float],
    out_dir: Path,
    num_games: int,
    random_seed: int,
):
    # Load models and features
    print("Loading models…")
    tcn = load_model(tcn_model_path, custom_objects={"TCN": TCN})
    cat = joblib.load(cat_model_path)
    cb_features = joblib.load(cat_features_path)

    # Prepare time ranges
    blocks = minute_blocks(START_MINUTE, HORIZON_MINUTES)
    block_ids = [b for (b,_,_) in blocks]
    start_chunk, end_chunk = chunk_range_for_minutes(START_MINUTE, HORIZON_MINUTES)

    # --- Scan game folders and select by EXACT 10 starters with chunks 1 & 241 ---
    valid_games: List[Tuple[str, List[str]]] = []
    all_game_dirs = [d for d in sorted(games_root.iterdir()) if d.is_dir()]
    for game_dir in all_game_dirs:
        starters = choose_starters_for_game(game_dir, verbose=False)
        if REQUIRE_EXACT_10 and len(starters) == 10:
            valid_games.append((game_dir.name, starters))
        elif (not REQUIRE_EXACT_10) and len(starters) >= 10:
            valid_games.append((game_dir.name, starters[:10]))

    print("\n=== Selection summary (valid pool) ===")
    print(f"Valid games found: {len(valid_games)}")
    if valid_games:
        print("  " + ", ".join(name for name, _ in valid_games))

    # --- Randomly pick K games from valid set (or all if fewer) ---
    random.seed(random_seed)
    if len(valid_games) > num_games:
        selected_games = random.sample(valid_games, num_games)
    else:
        selected_games = valid_games

    print(f"\nSampling {len(selected_games)} game(s) (requested {num_games}) with seed={random_seed}:")
    if selected_games:
        print("  " + ", ".join(name for name, _ in selected_games))
    else:
        print("  (none)")

    # Aggregation dict: (feature, multiplier, group_label) -> count
    counts: Dict[Tuple[str, float, str], int] = {}

    # --- Run experiment over selected games ---
    for game_name, starters in selected_games:
        game_dir = games_root / game_name
        print(f"\n=== Game {game_name} ===")
        print(f"  Starters ({len(starters)}): {', '.join(starters)}")

        # Baseline once per game
        try:
            baseline_players_df, baseline_avg = predict_baseline_for_game(
                game_dir, starters, tcn, cat, cb_features, avg_group_dir,
                start_chunk, end_chunk, block_ids
            )
        except Exception as e:
            print(f"  [!] Baseline failed: {e}")
            continue

        # For each perturbation setting
        for feat in perturb_features:
            for mult in perturb_multipliers:
                local_counts = {"G1": 0, "G2": 0, "G3": 0, "G4": 0}
                for code in starters:
                    scenario_stats = []
                    for gid in [1,2,3,4]:
                        scenario_avg = simulate_replacement_with_perturbation(
                            baseline_players_df=baseline_players_df,
                            target_player_code=code,
                            gid=gid,
                            avg_dir=avg_group_dir,
                            block_ids=block_ids,
                            cb_model=cat,
                            cb_features=cb_features,
                            feature_to_scale=feat,
                            multiplier=mult,
                        )
                        if scenario_avg is None:
                            continue
                        delta = scenario_avg - baseline_avg
                        scenario_stats.append((gid, delta))

                    if not scenario_stats:
                        continue
                    best_gid, best_delta = max(scenario_stats, key=lambda x: x[1])
                    if best_delta > 0:
                        local_counts[GID_TO_GROUPLBL[best_gid]] += 1

                # Add to global totals
                for g_lbl, c in local_counts.items():
                    key = (feat, mult, g_lbl)
                    counts[key] = counts.get(key, 0) + c

    # Save tidy totals CSV
    rows = []
    for (feat, mult, g_lbl), c in sorted(counts.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        rows.append({"feature": feat, "multiplier": mult, "group": g_lbl, "subs_count": c})
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "subs_counts_by_feature_multiplier.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n✅ Wrote {out_csv}")

    # Plots: one figure per feature (x: groups, multiple bars = multipliers)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _safe_name(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

    features_sorted = sorted(set([r["feature"] for r in rows]))
    for feat in features_sorted:
        df_feat = pd.DataFrame([r for r in rows if r["feature"] == feat])
        if df_feat.empty:
            continue
        groups = ["G1","G2","G3","G4"]
        multipliers = sorted(df_feat["multiplier"].unique())
        x = np.arange(len(groups))
        width = 0.8 / max(1, len(multipliers))
        plt.figure(figsize=(10, 5))
        for i, m in enumerate(multipliers):
            vals = [int(df_feat[(df_feat["group"] == g) & (df_feat["multiplier"] == m)]["subs_count"].sum()) for g in groups]
            plt.bar(x + i*width, vals, width, label=f"x{m:g}")
        plt.xticks(x + (len(multipliers)-1)*width/2.0, groups)
        plt.xlabel("Recommended group")
        plt.ylabel("# of SUB recommendations (total)")
        plt.title(f"SUB recommendations per group vs perturbation — {feat}")
        plt.legend()
        out_png = plots_dir / f"subs_counts_{_safe_name(feat)}_.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"  📈 {out_png}")

# ---------------------------- CLI ----------------------------

def parse_multipliers(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_features(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games_root", default=DEFAULT_GAMES_ROOT, help="Root folder containing game date folders")
    ap.add_argument("--avg_group_dir", default=DEFAULT_AVG_GROUP_5MIN_DIR, help="Folder with avg_allgames_group{1..4}_5min.csv")
    ap.add_argument("--tcn_model", default=DEFAULT_TCN_MODEL)
    ap.add_argument("--cat_model", default=DEFAULT_CAT_MODEL)
    ap.add_argument("--cat_features", default=DEFAULT_CAT_FEATURES)
    ap.add_argument("--perturb_features", default=",".join(DEFAULT_PERTURB_FEATURES),
                    help="Comma-separated list of base feature names to perturb (in replacement player)")
    ap.add_argument("--perturb_multipliers", default=",".join([str(x) for x in DEFAULT_PERTURB_MULTIPLIERS]),
                    help="Comma-separated multipliers, e.g., '1.0,1.1,1.2'")
    ap.add_argument("--out_dir", default="subs_perturbation_out")
    ap.add_argument("--num_games", type=int, default=6, help="Random number of valid games to sample")
    ap.add_argument("--random_seed", type=int, default=42, help="Seed for random sampling of games")
    args = ap.parse_args()

    run_experiment(
        games_root=Path(args.games_root),
        avg_group_dir=Path(args.avg_group_dir),
        tcn_model_path=Path(args.tcn_model),
        cat_model_path=Path(args.cat_model),
        cat_features_path=Path(args.cat_features),
        perturb_features=parse_features(args.perturb_features),
        perturb_multipliers=parse_multipliers(args.perturb_multipliers),
        out_dir=Path(args.out_dir),
        num_games=args.num_games,
        random_seed=args.random_seed,
    )

if __name__ == "__main__":
    main()
