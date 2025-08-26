#!/usr/bin/env python3
"""
Simulate possession over the next 15 minutes (e.g., minutes 60–75):
- Baseline: current starters (TCN 15s -> 5-min windows -> CatBoost grouped features)
- What-if: for EACH starter, replace him with an average player from EACH of the 4 groups (1..4)

Decision rule per player:
- If best delta (max over 4 groups) <= 0 → "he needs to stay"
- Else recommend the group that yields the highest positive delta

Groups by code prefix (file name like merged_features_<POS>_<ID>.csv):
  group1: CB
  group2: CM, DM
  group3: LB, LM, LW, RB, RM, RW, UNK
  group4: AM, CF
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tcn import TCN
import joblib


# ============================ USER CONFIG ============================

# GAME folder that contains: merged_features_<CODE>.csv for each player
GAME_DIR = Path("games_train_new/2024-04-20")

# Current starters (10 or 11 codes) — must match file suffix after "merged_features_"
STARTERS = ["CB_41","CB_3","LB_13","RB_9","DM_15","CM_18","AM_22","CF_34","RM_23","LM_28"]

# Start/horizon (minutes)
START_MINUTE = 60
HORIZON_MINUTES = 15   # 3 × 5-min windows

# Models (GROUPED version)
TCN_MODEL_PATH = Path("saved_model/even_better_model/player_tcn_model.keras")
CATBOOST_MODEL_PATH = Path("possession_catboost_grouped.pkl")
CATBOOST_FEATURES_PATH = Path("catboost_grouped_features.pkl")  # list of feature names used by grouped CatBoost

# Average-per-group 5-min folder (should contain files like "avg_allgames_group1_5min.csv", etc.)
AVG_GROUP_5MIN_DIR = Path("/home/stav.karasik/MaccabiProject/scripts/5min_windows_avg")

# TCN feature list & aliases (match your TCN training)
TCN_TARGET_COLS = [
    "inst_dist_m_sum", "Speed (m/s)_sum", "hsr_m_sum",
    "vha_count_1s_sum", "avg_jerk_1s_sum", "turns_per_sec_sum", "playerload_1s_sum",
    "walk_time_sum", "jog_time_sum", "run_time_sum", "sprint_time_sum",
    "total_sprints_sum", "sprint_attack_sum", "sprint_defense_sum",
    "dist_attack_sum", "dist_defense_sum", "time_attack_sum", "time_defense_sum",
    "attacking_third_time_sum", "middle_third_time_sum", "defending_third_time_sum"
]
# If a TCN column is missing, try alias name:
ALIAS_MAP = {
    "Speed (m/s)_mean": "Speed (m/s)_sum",
    "avg_jerk_1s_mean": "avg_jerk_1s_sum",
}

# History window length for TCN model (must match training)
INPUT_WINDOW = 130

# Output
OUT_DIR = Path("sim_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================ HELPERS ============================

SUM_LIKE_PAT  = re.compile(r".*(_sum|_count)$", re.IGNORECASE)
MEAN_LIKE_PAT = re.compile(r".*_mean$", re.IGNORECASE)

PREFIX_TO_GROUP_ID = {
    "CB": 1,
    "CM": 2, "DM": 2,
    "LB": 3, "LM": 3, "LW": 3, "RB": 3, "RM": 3, "RW": 3, "UNK": 3,
    "AM": 4, "CF": 4,
}
GROUP_ID_TO_HUMAN = {1: "group1_CB", 2: "group2_CM_DM", 3: "group3_FB_W", 4: "group4_AM_CF"}

# CatBoost TRAINING labels (G1..G4)
GID_TO_GROUPLBL = {1: "G1", 2: "G2", 3: "G3", 4: "G4"}
POS2GROUPLBL = {
    "CB": "G1",
    "CM": "G2", "DM": "G2",
    "RB": "G3", "RW": "G3", "RM": "G3", "LB": "G3", "LW": "G3", "LM": "G3", "UNK": "G3",
    "AM": "G4", "CF": "G4",
}

def get_prefix(code: str) -> str:
    return code.split("_", 1)[0]

def infer_group_id(code: str) -> int:
    return PREFIX_TO_GROUP_ID.get(get_prefix(code), 3)

def infer_group_label(code: str) -> str:
    return POS2GROUPLBL.get(get_prefix(code), "G3")

def minute_blocks(start_minute: int, horizon_min: int) -> List[Tuple[int,int,int]]:
    """Return list of (block_idx, minute_start, minute_end)."""
    blocks = []
    for m in range(start_minute, start_minute + horizon_min, 5):
        block = (m // 5) + 1
        blocks.append((block, m, m + 5))
    return blocks

def chunk_range_for_minutes(start_minute: int, horizon_min: int) -> Tuple[int, int]:
    """Convert minute range to [start_chunk, end_chunk] inclusive, 15s chunks (4/min)."""
    start_chunk = start_minute * 4
    end_chunk   = start_chunk + horizon_min * 4 - 1
    return start_chunk, end_chunk

def resolve_columns(df: pd.DataFrame, wanted: List[str], alias: Dict[str, str]) -> Tuple[pd.DataFrame, List[str]]:
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

def autoreg_predict_15s(raw: pd.DataFrame, used_cols: List[str], model, start_chunk: int, end_chunk: int, input_window: int) -> pd.DataFrame:
    """Autoregressive prediction (15s steps) on used_cols between [start_chunk..end_chunk]."""
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
        yhat = model.predict(model_input, verbose=0)[0]  # (len(used_cols),)
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
    """Re-bin 15s predictions to independent 5-min blocks."""
    if df15.empty:
        return df15.copy()
    num = df15.select_dtypes(include="number").copy()
    num = num.sort_values("chunk").reset_index(drop=True)

    # 20 rows = 5 min
    block = ((num["chunk"] - 1) // 20) + 1
    group_key = pd.Series(block, name="block")

    sum_like_cols  = [c for c in num.columns if c != "chunk" and SUM_LIKE_PAT.fullmatch(c)]
    mean_like_cols = [c for c in num.columns if c != "chunk" and MEAN_LIKE_PAT.fullmatch(c)]
    other_cols     = [c for c in num.columns if c not in ["chunk"] + sum_like_cols + mean_like_cols]

    parts = []
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

# ---------- group-average 5-min files ----------

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

def load_group_avg_5min_slice(gid: int, block_ids: List[int]) -> pd.DataFrame:
    fp = find_avg_file_for_group(AVG_GROUP_5MIN_DIR, gid)
    gdf = pd.read_csv(fp)
    return _slice_group_file_to_blocks(gdf, block_ids)

# ---------- grouped team features for CatBoost (G1..G4) ----------

def build_team_features_grouped(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: rows per player per 5-min window with:
      ['minute_start','minute_end','player_id','position','group', <feature columns...>]
    Output: one row per window with:
      count_G1..count_G4 and per-feature means per group: <feat>__G1, ..., <feat>__G4
    """
    key_cols = ["minute_start", "minute_end", "player_id", "position", "group"]
    feature_cols = [c for c in players_df.columns if c not in key_cols]

    ALL_GROUPS = ["G1", "G2", "G3", "G4"]
    rows = []
    for (start, end), window in players_df.groupby(["minute_start", "minute_end"], sort=True):
        row = {"minute_start": start, "minute_end": end}
        # counts per group
        for g in ALL_GROUPS:
            row[f"count_{g}"] = int((window["group"] == g).sum())

        # per-group MEAN for every feature
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


# ============================ MAIN ============================

def main():
    # Load models
    tcn = load_model(TCN_MODEL_PATH, custom_objects={"TCN": TCN})
    cat = joblib.load(CATBOOST_MODEL_PATH)
    cb_features = joblib.load(CATBOOST_FEATURES_PATH)  # ordered feature list for CatBoost

    # Compute indices
    blocks = minute_blocks(START_MINUTE, HORIZON_MINUTES)
    block_ids = [b for (b,_,_) in blocks]
    start_chunk, end_chunk = chunk_range_for_minutes(START_MINUTE, HORIZON_MINUTES)

    # Predict baseline per-player 15s -> 5-min
    player_windows = []
    for code in STARTERS:
        fp = GAME_DIR / f"merged_features_{code}.csv"
        if not fp.exists():
            print(f"⚠️ Missing player file: {fp.name} — skipping this player in baseline.")
            continue

        raw = pd.read_csv(fp)
        feats_df, used_cols = resolve_columns(raw, TCN_TARGET_COLS, ALIAS_MAP)

        df15 = autoreg_predict_15s(
            raw=pd.concat([raw[["chunk"]], feats_df], axis=1),
            used_cols=feats_df.columns.tolist(),
            model=tcn,
            start_chunk=start_chunk,
            end_chunk=end_chunk,
            input_window=INPUT_WINDOW,
        )
        df5 = rebin_15s_to_5min(df15)

        # Fallback fill using this player's group average (optional)
        gid_self = infer_group_id(code)
        try:
            g5_self = load_group_avg_5min_slice(gid_self, block_ids)
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
        raise SystemExit("No players available for baseline.")

    baseline_players_df = pd.concat(player_windows, ignore_index=True)
    baseline_team_grouped = build_team_features_grouped(baseline_players_df)
    baseline_X = ensure_catboost_features(baseline_team_grouped, cb_features)
    baseline_preds = cat.predict(baseline_X)
    baseline_avg_poss = float(np.mean(baseline_preds))

    # ---- Simulate one-for-one subs against avg of EACH group (1..4)
    per_player_results = []
    csv_rows = []

    for code in STARTERS:
        scenario_stats = []
        for gid in [1,2,3,4]:
            try:
                g5_rep = load_group_avg_5min_slice(gid, block_ids)
            except Exception as e:
                scenario_stats.append({"gid": gid, "avg": None, "delta": None, "per_window": None, "error": str(e)})
                continue

            g5_rep = g5_rep.copy()
            g5_rep["player_id"] = f"group{gid}_AVG"
            g5_rep["position"]  = GROUP_ID_TO_HUMAN[gid]   # not used by training, but ok to keep
            g5_rep["group"]     = GID_TO_GROUPLBL[gid]     # CRITICAL: label CatBoost expects (G1..G4)

            # Replace only this player's rows, keep others from baseline
            repl_players = []
            for pcode in STARTERS:
                if pcode == code:
                    repl_players.append(g5_rep)
                else:
                    repl_players.append(baseline_players_df[baseline_players_df["player_id"] == pcode])
            scenario_players_df = pd.concat(repl_players, ignore_index=True)

            scenario_team_grouped = build_team_features_grouped(scenario_players_df)
            scenario_X = ensure_catboost_features(scenario_team_grouped, cb_features)
            scenario_preds = cat.predict(scenario_X)
            scenario_avg = float(np.mean(scenario_preds))
            delta = scenario_avg - baseline_avg_poss

            scenario_stats.append({
                "gid": gid,
                "avg": scenario_avg,
                "delta": delta,
                "per_window": [float(x) for x in scenario_preds],
                "error": None
            })

        # choose best delta among available scenarios
        available = [s for s in scenario_stats if s["delta"] is not None]
        if available:
            best = max(available, key=lambda d: d["delta"])
            best_gid = best["gid"]
            best_delta = best["delta"]
            decision = (
                f"he needs to stay (best Δ={best_delta:+.4f} with {GROUP_ID_TO_HUMAN[best_gid]})"
                if best_delta <= 0.0 else
                f"replace with {GROUP_ID_TO_HUMAN[best_gid]} (Δ={best_delta:+.4f})"
            )
        else:
            best_gid, best_delta, decision = None, None, "no valid group avg files found — cannot decide"

        per_player_results.append({
            "player": code,
            "baseline_avg_possession_next15": baseline_avg_poss,
            "scenarios": [
                {
                    "group": GROUP_ID_TO_HUMAN[s["gid"]],
                    "avg_possession_next15": s["avg"],
                    "delta_vs_baseline": s["delta"],
                    "per_window_possession": s["per_window"],
                    "error": s["error"],
                } for s in scenario_stats
            ],
            "best_decision": decision,
            "best_group": GROUP_ID_TO_HUMAN[best_gid] if best_gid else None,
            "best_delta": best_delta
        })

    # ---- Save JSON
    summary = {
        "game": GAME_DIR.name,
        "start_minute": START_MINUTE,
        "horizon_minutes": HORIZON_MINUTES,
        "blocks": block_ids,
        "baseline_per_window": [float(x) for x in baseline_preds],
        "baseline_avg_possession_next15": baseline_avg_poss,
        "per_player_decisions": per_player_results
    }
    out_json = OUT_DIR / f"subs_possession_allgroups_{GAME_DIR.name}_m{START_MINUTE}_{START_MINUTE+HORIZON_MINUTES}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Wrote {out_json}")

    # ---- Save CSV with per-group deltas only (plus decision)
    rows = []
    for r in per_player_results:
        row = {
            "player": r["player"],
            "baseline_avg_possession_next15": r["baseline_avg_possession_next15"],
            "best_group": r["best_group"] or "",
            "best_delta": r["best_delta"] if r["best_delta"] is not None else "",
            "decision": r["best_decision"],
        }
        # add delta for each group 1..4
        for gid in [1, 2, 3, 4]:
            s = next((x for x in r["scenarios"] if x["group"] == GROUP_ID_TO_HUMAN[gid]), None)
            row[f"delta_group{gid}"] = s["delta_vs_baseline"] if (s and s["delta_vs_baseline"] is not None) else ""
        rows.append(row)

    cols = [
        "player",
        "baseline_avg_possession_next15",
        "delta_group1", "delta_group2", "delta_group3", "delta_group4",
        "best_group", "best_delta", "decision",
    ]
    out_csv = OUT_DIR / f"subs_possession_allgroups_{GAME_DIR.name}_m{START_MINUTE}_{START_MINUTE+HORIZON_MINUTES}.csv"
    pd.DataFrame(rows)[cols].to_csv(out_csv, index=False)
    print(f"✅ Wrote {out_csv}")

    # ---- Console report
    print("\n📝 Per-player decisions:")
    for r in per_player_results:
        print(f"  - {r['player']}: {r['best_decision']}")


if __name__ == "__main__":
    main()
