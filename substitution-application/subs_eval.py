"""Evaluate substitution scenarios by replacing players with group-average profiles and
measure predicted possession impact.

Purpose:
- Quantify how substituting a specific starter with a group-average player (G1..G4) changes
  5‑minute possession predictions across game windows.

Inputs:
- Root folder of games containing independent 5‑minute per‑player feature CSVs
  (e.g., merged_features_<POS>_<ID>.csv)
- Optional true possession per window (poss.csv) for calibration/validation
- Pretrained possession model (joblib) with the same POS2GROUP mapping as training

Workflow:
- Build per-window features from raw player-level rows
- For each player and window, form scenarios by swapping the player’s features with the
  average profile of a chosen group (G1..G4)
- Predict possession for baseline and each scenario and compute delta

Outputs:
- Per-window CSVs with baseline, scenario predictions, and deltas
- Aggregated metrics summarizing the impact per player and group

Notes:
- Ensure windowing and feature naming conventions match training/evaluation pipelines.
- Group-average profiles must be prepared consistently with training data.
"""
# subs_eval_clean.py
# Analyze impact of substituting each player with an average player on predicted possession
# Uses dynamic feature discovery; no manual remapping of _sum/_mean

import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# ======================
# 1. Build per-window aggregates (game-level)
# ======================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # df: raw player-level rows with minute_start, minute_end, feature columns
    feature_cols = [c for c in df.columns if c not in ['minute_start','minute_end','player_id']]
    rows = []
    for (start, end), group in df.groupby(['minute_start','minute_end']):
        # for each window, average features across all rows (players or segments)
        mean_vals = group[feature_cols].mean()
        row = {'minute_start': start, 'minute_end': end}
        row.update(mean_vals.to_dict())
        rows.append(row)
    return pd.DataFrame(rows)

# ======================
# 2. Load all games and true possession labels
# ======================
def load_all_games(root_folder: str) -> pd.DataFrame:
    all_windows = []
    for game_folder in Path(root_folder).iterdir():
        if not game_folder.is_dir():
            continue
        # read and tag all raw features in this game
        parts = []
        for f in game_folder.glob('merged_features_*.csv'):
            df = pd.read_csv(f)
            df['player_id'] = f.stem.replace('merged_features_','')
            parts.append(df)
        if not parts:
            continue
        raw = pd.concat(parts, ignore_index=True)
        agg = build_features(raw)
        # get true possession for each window
        poss = pd.read_csv(game_folder/'poss.csv')
        poss['minute_start'] = poss['time_start_sec'] // 60
        poss['minute_end']   = poss['time_end_sec']   // 60
        merged = agg.merge(
            poss[['minute_start','minute_end','maccabi_haifa_possession_percent']],
            on=['minute_start','minute_end'], how='inner'
        )
        merged['game'] = game_folder.name
        all_windows.append(merged)
    return pd.concat(all_windows, ignore_index=True) if all_windows else pd.DataFrame()

# ======================
# 3. Compute global average player feature vector
# ======================
def compute_avg_player_vector(root_folder: str, feat_names: list) -> np.ndarray:
    vectors = []
    for game_folder in Path(root_folder).iterdir():
        if not game_folder.is_dir(): continue
        for f in game_folder.glob('merged_features_*.csv'):
            df = pd.read_csv(f)
            df['player_id'] = f.stem.replace('merged_features_','')
            agg = build_features(df)
            # average across this player's windows (fill missing features)
            agg_full = agg.reindex(columns=['minute_start','minute_end'] + feat_names, fill_value=0)
            # drop window cols, take mean of feature vectors
            vec = agg_full[feat_names].mean().values
            vectors.append(vec)
    if not vectors:
        raise RuntimeError("No player data to compute average vector.")
    return np.mean(vectors, axis=0)

# ======================
# 4. Main substitution analysis
# ======================
if __name__ == '__main__':
    games_root = '5min_eval'
    model_path = 'possession_catboost_all_features.pkl'
    out_csv    = 'substitution_impact.csv'

    # load model
    print("Loading model…")
    model = joblib.load(model_path)

    # load and aggregate all windows to get feature names
    print("Loading all games for feature discovery…")
    data = load_all_games(games_root)
    if data.empty:
        print("No data found in", games_root)
        exit(1)

    # determine feature columns automatically
    feat_names = [c for c in data.columns if c not in ['game','minute_start','minute_end','maccabi_haifa_possession_percent']]
    print("Discovered features:", feat_names)

    # compute global avg player vector
    print("Computing global average player vector…")
    avg_vec = compute_avg_player_vector(games_root, feat_names)
    print(avg_vec)

    results = []
    # process each game separately
    for game_folder in Path(games_root).iterdir():
        if not game_folder.is_dir(): continue
        print("Processing game:", game_folder.name)

        # per-game aggregated windows
        game_data = data[data['game']==game_folder.name].copy()
        agg_orig = game_data[['minute_start','minute_end'] + feat_names]
        X_orig = agg_orig[feat_names].values
        y_orig = model.predict(X_orig)

        # load individual player raw and aggregated features
        players = {}
        for f in game_folder.glob('merged_features_*.csv'):
            pid = f.stem.replace('merged_features_','')
            df_raw = pd.read_csv(f)
            df_raw['player_id'] = pid
            players[pid] = df_raw

        n_players = len(players)
        if n_players == 0:
            continue

        # simulate each substitution
        for pid, df_raw in players.items():
            # aggregated features for this player
            agg_p = build_features(df_raw)
            # merge with game windows
            merged = agg_orig.merge(
                agg_p[['minute_start','minute_end'] + feat_names],
                on=['minute_start','minute_end'],
                suffixes=('','_p'), how='inner'
            )
            if merged.empty:
                continue
            sim_X = []
            for _, row in merged.iterrows():
                orig = row[feat_names].values.astype(float)
                player = row[[f + '_p' for f in feat_names]].values.astype(float)
                # new average vector after substitution
                new_avg = (orig * n_players - player + avg_vec) / n_players
                sim_X.append(new_avg)
            sim_X = np.vstack(sim_X)
            y_sim = model.predict(sim_X)

            # compute mean change
            delta = y_sim - y_orig[:len(y_sim)]
            results.append({
                'game': game_folder.name,
                'player_id': pid,
                'mean_delta': float(delta.mean())
            })

    # save results
    df_res = pd.DataFrame(results)
    df_res.to_csv(out_csv, index=False)
    print("Saved substitution impact to", out_csv)
    print(df_res)
