#!/usr/bin/env python3
"""
Experiment: do per-player (on-pitch) feature perturbations make starters less subbable?

For each selected game (exactly 10 starters with chunk==1 and chunk==241):
  1) Compute baseline decision per starter:
       - Build baseline team (TCN 15s -> 5-min -> grouped CatBoost).
       - For that starter, test replacement with each group (G1..G4) average player.
       - best_delta_baseline = max_group(avg_scenario - baseline_avg).
       - If best_delta_baseline > 0 → SUB is recommended; else KEEP.
  2) For each feature and each multiplier:
       - Multiply ONLY THIS STARTER'S 5-min predicted feature by the multiplier.
       - Rebuild team, recompute baseline_avg_pert.
       - Re-evaluate replacement with each (G1..G4) group.
       - best_delta_pert <= 0 means KEEP under perturbation.
       - Count a **conversion** if (baseline recommended SUB) and (perturbed says KEEP).
       - Also track final KEEP counts (optional).

Outputs (under --out_dir):
  - conversions_to_keep_by_feature_multiplier.csv (feature, multiplier, group, conversions)
  - (optional) final_keep_counts_by_feature_multiplier.csv
  - plots_conversions/subs_to_keep_<feature>.png   (bars per group, one bar per multiplier)

Game selection:
  - Scans --games_root; include game iff it has EXACTLY 10 starters (chunk==1 & 241).
  - Randomly sample --num_games from the valid pool (or all if fewer).
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

# ---------------------------- Paths & defaults ----------------------------

DEFAULT_GAMES_ROOT = "/home/stav.karasik/MaccabiProject/scripts/5min_windows"
DEFAULT_AVG_GROUP_5MIN_DIR = "/home/stav.karasik/MaccabiProject/scripts/5min_windows_avg"

DEFAULT_TCN_MODEL = "saved_model/even_better_model/player_tcn_model.keras"
DEFAULT_CAT_MODEL = "possession_catboost_grouped.pkl"
DEFAULT_CAT_FEATURES = "catboost_grouped_features.pkl"

START_MINUTE = 60
HORIZON_MINUTES = 30
INPUT_WINDOW = 130

# TCN targets & alias map (must match your TCN training)
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

# Features to perturb on the STARTER (one at a time)
PERTURB_FEATURES_DEFAULT = [
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
PERTURB_MULTIPLIERS_DEFAULT = [1.0, 1.1, 1.2]

# EXACTLY 10 starters required
REQUIRE_EXACT_10 = True

# Label map for nicer plot titles (edit as you like)
LABEL_MAP = {
    "inst_dist_m_sum":  "Distance (m)",
    "Speed (m/s)_sum":  "Speed (m/s)",
    "hsr_m_sum":        "High-speed running (m)",
    "vha_count_1s_sum": "Very high acceleration count",
    "avg_jerk_1s_sum":  "Avg jerk",
    "turns_per_sec_sum":"Turns count",
    "total_sprints_sum":"Total sprints",
    "sprint_attack_sum":"Sprints (attack)",
    "sprint_defense_sum":"Sprints (defense)",
    "dist_attack_sum":  "Distance (attack)",
    "dist_defense_sum": "Distance (defense)",
    "playerload_1s_sum":"PlayerLoad",
}

GROUP_NICE = {"G1":"G1 (CB)", "G2":"G2 (CM/DM)", "G3":"G3 (FB/W)", "G4":"G4 (AM/CF)"}

# ---------------------------- Mappings ----------------------------

PREFIX_TO_GROUP_ID = {
    "CB": 1,
    "CM": 2, "DM": 2,
    "LB": 3, "LM": 3, "LW": 3, "RB": 3, "RM": 3, "RW": 3, "UNK": 3,
    "AM": 4, "CF": 4,
}
GROUP_ID_TO_HUMAN = {1: "group1_CB", 2: "group2_CM_DM", 3: "group3_FB_W", 4: "group4_AM_CF"}
GID_TO_GROUPLBL    = {1: "G1", 2: "G2", 3: "G3", 4: "G4"}

SUM_LIKE_PAT  = re.compile(r".*(_sum|_count)$", re.IGNORECASE)
MEAN_LIKE_PAT = re.compile(r".*_mean$", re.IGNORECASE)

# ---------------------------- Utils ----------------------------

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
    # 1-based chunks: minute 60 starts at 241
    start_chunk = start_minute * 4 + 1
    end_chunk   = start_chunk + horizon_min * 4 - 1
    return start_chunk, end_chunk

def read_csv_robust(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, engine="python", on_bad_lines="skip")
        except Exception:
            return None

def player_has_chunks_1_and_241(csv_path: Path) -> bool:
    df = read_csv_robust(csv_path)
    if df is None or "chunk" not in df.columns:
        return False
    ch = pd.to_numeric(df["chunk"], errors="coerce").dropna().astype(int)
    return bool((ch == 1).any() and (ch == 241).any())

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
    if df15.empty:
        return df15.copy()
    num = df15.select_dtypes(include="number").copy()
    num = num.sort_values("chunk").reset_index(drop=True)
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

# ---------------------------- Core helpers ----------------------------

def choose_starters_for_game(game_dir: Path) -> List[str]:
    starters = []
    for f in sorted(game_dir.glob("merged_features_*.csv")):
        if player_has_chunks_1_and_241(f):
            starters.append(f.stem.replace("merged_features_", ""))
    return starters

def predict_player_df5(game_dir: Path, code: str, tcn_model, start_chunk: int, end_chunk: int) -> pd.DataFrame:
    fp = game_dir / f"merged_features_{code}.csv"
    raw = read_csv_robust(fp)
    if raw is None:
        raise RuntimeError(f"Cannot read {fp}")
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
    return df5

def baseline_players_windows(game_dir: Path, starters: List[str], tcn_model, avg_dir: Path, block_ids: List[int]) -> pd.DataFrame:
    parts = []
    for code in starters:
        df5 = predict_player_df5(game_dir, code, tcn_model, *chunk_range_for_minutes(START_MINUTE, HORIZON_MINUTES))
        # group fallback fill (optional)
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
        parts.append(merged)
    if not parts:
        raise RuntimeError("No valid players for baseline.")
    return pd.concat(parts, ignore_index=True)

def team_avg_possession(players_df: pd.DataFrame, cb_model, cb_features: List[str]) -> float:
    team = build_team_features_grouped(players_df)
    X = ensure_catboost_features(team, cb_features)
    preds = cb_model.predict(X)
    return float(np.mean(preds))

def best_delta_for_player_replacement(players_df: pd.DataFrame, target_code: str, avg_dir: Path, block_ids: List[int], cb_model, cb_features: List[str]) -> float:
    """Return best_delta (>0 means SUB recommended) for replacing target_code with G1..G4 avg."""
    base_avg = team_avg_possession(players_df, cb_model, cb_features)
    deltas = []
    for gid in [1,2,3,4]:
        try:
            g5 = load_group_avg_5min_slice(avg_dir, gid, block_ids).copy()
        except Exception:
            continue
        g5["player_id"] = f"group{gid}_AVG"
        g5["position"]  = GROUP_ID_TO_HUMAN[gid]
        g5["group"]     = GID_TO_GROUPLBL[gid]
        # swap
        repl = []
        for pcode in players_df["player_id"].unique():
            repl.append(g5 if pcode == target_code else players_df[players_df["player_id"] == pcode])
        scen_df = pd.concat(repl, ignore_index=True)
        scen_avg = team_avg_possession(scen_df, cb_model, cb_features)
        deltas.append(scen_avg - base_avg)
    return max(deltas) if deltas else float("-inf")  # if no groups available, treat as no improvement

def apply_player_perturbation(players_df: pd.DataFrame, target_code: str, feature: str, mult: float) -> pd.DataFrame:
    out = players_df.copy()
    mask = out["player_id"] == target_code
    if feature in out.columns and abs(mult - 1.0) > 1e-12:
        out.loc[mask, feature] = out.loc[mask, feature] * mult
    return out

# ---------------------------- Main experiment ----------------------------

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
    # Models
    print("Loading models…")
    tcn = load_model(tcn_model_path, custom_objects={"TCN": TCN})
    cat = joblib.load(cat_model_path)
    cb_features = joblib.load(cat_features_path)

    # Time windows & blocks
    blocks = minute_blocks(START_MINUTE, HORIZON_MINUTES)
    block_ids = [b for (b,_,_) in blocks]
    start_chunk, end_chunk = chunk_range_for_minutes(START_MINUTE, HORIZON_MINUTES)

    # Select valid games (exactly 10 starters 1 & 241) and sample K
    valid: List[Tuple[str, List[str]]] = []
    for game_dir in sorted([d for d in games_root.iterdir() if d.is_dir()]):
        starters = choose_starters_for_game(game_dir)
        if REQUIRE_EXACT_10 and len(starters) == 10:
            valid.append((game_dir.name, starters))
    print("\n=== Valid games pool ===")
    print(f"Found {len(valid)} valid games.")
    if valid:
        print("  " + ", ".join(name for name,_ in valid))

    random.seed(random_seed)
    chosen = random.sample(valid, min(num_games, len(valid)))
    print(f"\nSampling {len(chosen)} game(s) (requested {num_games}) with seed={random_seed}:")
    if chosen:
        print("  " + ", ".join(name for name,_ in chosen))
    else:
        print("  (none)")

    # Aggregation
    conversions: Dict[Tuple[str, float, str], int] = {}  # (feature, mult, player_group) -> count
    final_keep: Dict[Tuple[str, float, str], int] = {}   # optional

    # Process each chosen game
    for game_name, starters in chosen:
        game_dir = games_root / game_name
        print(f"\n=== Game {game_name} ===")
        print(f"  Starters ({len(starters)}): {', '.join(starters)}")

        # Baseline per-player windows (df5 for all starters)
        try:
            base_players_df = baseline_players_windows(game_dir, starters, tcn, avg_group_dir, block_ids)
        except Exception as e:
            print(f"  [!] Baseline windows failed: {e}")
            continue

        # Compute baseline best_delta per starter
        base_best_delta = {}
        for code in starters:
            try:
                base_best_delta[code] = best_delta_for_player_replacement(
                    base_players_df, code, avg_group_dir, block_ids, cat, cb_features
                )
            except Exception as e:
                print(f"  [!] Baseline best-delta failed for {code}: {e}")
                base_best_delta[code] = float("nan")

        # Perturbation loop
        for feat in perturb_features:
            for mult in perturb_multipliers:
                # For each starter independently
                for code in starters:
                    # Group label for aggregation
                    g_lbl = infer_group_label(code)

                    # Perturb ONLY this starter's windows
                    try:
                        pert_df = apply_player_perturbation(base_players_df, code, feat, mult)
                        best_delta_pert = best_delta_for_player_replacement(
                            pert_df, code, avg_group_dir, block_ids, cat, cb_features
                        )
                    except Exception:
                        continue

                    # Optional: record final KEEP
                    if best_delta_pert <= 0:
                        final_keep[(feat, mult, g_lbl)] = final_keep.get((feat, mult, g_lbl), 0) + 1

                    # Count conversion: baseline SUB → perturbed KEEP
                    b = base_best_delta.get(code, float("nan"))
                    if (not np.isnan(b)) and (b > 0.0) and (best_delta_pert <= 0.0):
                        conversions[(feat, mult, g_lbl)] = conversions.get((feat, mult, g_lbl), 0) + 1

    # ----- Write CSVs -----
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_conv = [
        {"feature": f, "multiplier": m, "group": g, "conversions_to_keep": c}
        for (f,m,g), c in sorted(conversions.items(), key=lambda x: (x[0][0], x[0][1], x[0][2]))
    ]
    conv_csv = out_dir / "conversions_to_keep_by_feature_multiplier.csv"
    pd.DataFrame(rows_conv).to_csv(conv_csv, index=False)
    print(f"\n✅ Wrote {conv_csv}")

    rows_final = [
        {"feature": f, "multiplier": m, "group": g, "final_keep_count": c}
        for (f,m,g), c in sorted(final_keep.items(), key=lambda x: (x[0][0], x[0][1], x[0][2]))
    ]
    final_csv = out_dir / "final_keep_counts_by_feature_multiplier.csv"
    pd.DataFrame(rows_final).to_csv(final_csv, index=False)
    print(f"✅ Wrote {final_csv}")

    # ----- Plots (conversions) -----
    plots_dir = out_dir / "plots_conversions"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _safe(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", s)

    df_conv = pd.DataFrame(rows_conv)
    if not df_conv.empty:
        for feat in sorted(df_conv["feature"].unique()):
            sub = df_conv[df_conv["feature"] == feat].copy()
            disp = LABEL_MAP.get(feat, feat)
            groups = ["G1","G2","G3","G4"]
            xticks = [GROUP_NICE.get(g, g) for g in groups]
            multipliers = sorted(sub["multiplier"].unique())
            x = np.arange(len(groups))
            width = 0.8 / max(1, len(multipliers))

            plt.figure(figsize=(10, 5))
            for i, m in enumerate(multipliers):
                vals = [int(sub[(sub["group"] == g) & (sub["multiplier"] == m)]["conversions_to_keep"].sum()) for g in groups]
                plt.bar(x + i*width, vals, width, label=f"x{m:g}")

            plt.xticks(x + (len(multipliers)-1)*width/2.0, xticks)
            plt.xlabel("Starter group")
            plt.ylabel("Conversions to KEEP (SUB → KEEP)")
            plt.title(f"Fewer subs after perturbation — {disp}")
            plt.legend()
            out_png = plots_dir / f"subs_to_keep_{_safe(disp)}.png"
            plt.tight_layout()
            plt.savefig(out_png, dpi=200)
            plt.close()
            print(f"  📈 {out_png}")
    else:
        print("No conversions recorded; skip plots.")

# ---------------------------- CLI ----------------------------

def parse_multipliers(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_features(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games_root", default=DEFAULT_GAMES_ROOT)
    ap.add_argument("--avg_group_dir", default=DEFAULT_AVG_GROUP_5MIN_DIR)
    ap.add_argument("--tcn_model", default=DEFAULT_TCN_MODEL)
    ap.add_argument("--cat_model", default=DEFAULT_CAT_MODEL)
    ap.add_argument("--cat_features", default=DEFAULT_CAT_FEATURES)
    ap.add_argument("--perturb_features", default=",".join(PERTURB_FEATURES_DEFAULT),
                    help="Comma-separated base feature names to scale on the STARTER")
    ap.add_argument("--perturb_multipliers", default=",".join(str(x) for x in PERTURB_MULTIPLIERS_DEFAULT),
                    help="Comma-separated multipliers, e.g., '1.0,1.1,1.2'")
    ap.add_argument("--out_dir", default="subs_onpitch_perturb_out")
    ap.add_argument("--num_games", type=int, default=6, help="Random number of valid games to sample")
    ap.add_argument("--random_seed", type=int, default=42, help="Seed for random sampling")
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
