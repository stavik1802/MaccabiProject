"""Evaluate the grouped-features CatBoost possession model on held-out games/windows.

Purpose:
- Mirror the training-time group aggregation to generate features and evaluate a saved model
  using MAE/RMSE/R2 and diagnostic plots.

Inputs:
- Directory with game folders containing per-player 5‑minute window CSVs (merged_features_*.csv)
- Ground-truth possession per window (poss.csv) when available
- A trained CatBoostRegressor saved via joblib

Workflow:
- Parse player position from filename → map to group (G1..G4)
- Aggregate per-window features by group (means + counts)
- Concatenate into a row per window and predict possession
- Compare predictions against ground truth and compute metrics

Outputs:
- Metrics printed and saved (CSV)
- Plots comparing predicted vs. actual possession

Notes:
- POS2GROUP mapping must match the training file.
- Ensure independent 5‑minute windows preprocessing was applied to player files.
"""
#!/usr/bin/env python3
# Evaluate the grouped-features CatBoost possession model
# Mirrors the training pipeline's per-group aggregation (means + counts)

import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------- Role groups (match training) --------------------
POS2GROUP = {
    "CB": "G1",
    "CM": "G2", "DM": "G2",
    "RB": "G3", "RW": "G3", "RM": "G3", "LB": "G3", "LW": "G3", "LM": "G3", "UNK": "G3",
    "AM": "G4", "CF": "G4",
}
ALL_GROUPS = ["G1", "G2", "G3", "G4"]

def parse_pos_from_stem(stem: str) -> str:
    # "merged_features_AM_21" -> "AM"
    try:
        parts = stem.replace("merged_features_", "").split("_")
        return parts[0]
    except Exception:
        return "UNK"

# -------------------- Feature building (match training) --------------------
def build_features_grouped(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    players_df columns (per player window):
      ['minute_start','minute_end','player_id','position','group', <features...>]
    Returns one row per [minute_start, minute_end] with:
      - count_G1..G4
      - For every original feature c: c__G1..c__G4 (per-group MEANs)
    """
    key_cols = ["minute_start", "minute_end", "player_id", "position", "group"]
    feature_cols = [c for c in players_df.columns if c not in key_cols]

    rows = []
    for (start, end), window in players_df.groupby(["minute_start", "minute_end"], sort=True):
        row = {"minute_start": start, "minute_end": end}

        # counts per group
        for g in ALL_GROUPS:
            row[f"count_{g}"] = int((window["group"] == g).sum())

        # per-group MEAN for every feature (fill zeros if group absent)
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

# -------------------- Load games (match training) --------------------
def load_all_games_grouped(root_folder: str) -> pd.DataFrame:
    """
    Directory layout:
      root/
        YYYY-MM-DD/
          merged_features_*.csv
          poss.csv  (with time_start_sec, time_end_sec, maccabi_haifa_possession_percent)
    """
    root = Path(root_folder)
    all_games = []

    for game in root.iterdir():
        if not game.is_dir():
            continue

        parts = []
        for f in game.glob("merged_features_*.csv"):
            df = pd.read_csv(f)
            stem = f.stem  # e.g. merged_features_AM_21
            pos = parse_pos_from_stem(stem)
            grp = POS2GROUP.get(pos, "UNK")
            df["player_id"] = stem.replace("merged_features_", "")
            df["position"]  = pos
            df["group"]     = grp
            parts.append(df)

        if not parts:
            continue

        players_df = pd.concat(parts, ignore_index=True)
        feats = build_features_grouped(players_df)

        poss = pd.read_csv(game / "poss.csv")
        poss["minute_start"] = (poss["time_start_sec"] // 60).astype(int)
        poss["minute_end"]   = (poss["time_end_sec"]   // 60).astype(int)

        merged = feats.merge(
            poss[["minute_start", "minute_end", "maccabi_haifa_possession_percent"]],
            on=["minute_start", "minute_end"],
            how="inner"
        )
        merged["game"] = game.name
        all_games.append(merged)

    return pd.concat(all_games, ignore_index=True) if all_games else pd.DataFrame()

# -------------------- Evaluation --------------------
def evaluate_model(
    model_path: str = "possession_catboost_grouped.pkl",
    feature_names_path: str = "catboost_grouped_features.pkl",
    games_root: str = "/home/stav.karasik/MaccabiProject/scripts/train_poss_subs/5min_windows_eval",
    out_master_csv: str = "catboost_grouped_predictions.csv",
):
    print("📥 Loading model and feature list…")
    model = joblib.load(model_path)
    feature_names = joblib.load(feature_names_path)

    print(f"📥 Loading grouped features from: {games_root}")
    data = load_all_games_grouped(games_root)
    if data.empty:
        print("❌ No data to evaluate. Check the path.")
        return

    # Align to training feature set 1:1 (missing cols -> 0.0)
    X = data.reindex(columns=feature_names, fill_value=0.0)
    y_true = data["maccabi_haifa_possession_percent"].astype(float).values

    print("🤖 Predicting…")
    y_pred = model.predict(X)

    # Metrics
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print("\n📊 Overall Regression Metrics:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE : {mae:.4f}")
    print(f"   R²  : {r2:.4f}")

    # Master CSV
    out = data[["game", "minute_start", "minute_end", "maccabi_haifa_possession_percent"]].copy()
    out["predicted_possession"] = y_pred
    out = out.sort_values(["game", "minute_start", "minute_end"])
    out.to_csv(out_master_csv, index=False)
    print(f"✅ Master CSV saved → {out_master_csv}")

    # Per-game CSVs + plots (saved inside each game folder under games_root)
    for game, gdf in out.groupby("game"):
        folder = Path(games_root) / game
        folder.mkdir(parents=True, exist_ok=True)

        csv_path = folder / f"{game}_grouped_predictions.csv"
        gdf.to_csv(csv_path, index=False)

        plt.figure(figsize=(10, 4))
        plt.plot(gdf["minute_start"], gdf["maccabi_haifa_possession_percent"], marker="o", label="True")
        plt.plot(gdf["minute_start"], gdf["predicted_possession"], marker="x", linestyle="--", label="Pred")
        plt.title(f"Possession % – {game} (Grouped Features)")
        plt.xlabel("Minute"); plt.ylabel("Possession %")
        plt.ylim(0, 1); plt.legend(); plt.tight_layout()

        plot_path = folder / f"{game}_grouped_plot.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"📈 Saved plot → {plot_path}")

if __name__ == "__main__":
    evaluate_model()
