# train_possession_grouped_features.py
# Predict possession with CatBoost using per-group aggregated features

import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor

# -------------------- Role groups --------------------
# map position code (from filename) -> group label
POS2GROUP = {
    "CB": "G1",
    "CM": "G2", "DM": "G2",
    "RB": "G3", "RW": "G3", "RM": "G3", "LB": "G3", "LW": "G3", "LM": "G3","UNK":"G3",
    "AM": "G4", "CF": "G4",
}
ALL_GROUPS = ["G1", "G2", "G3", "G4"]

def parse_pos_from_stem(stem: str) -> str:
    # stem looks like "merged_features_AM_21" -> "AM"
    # fallback to last-but-one token if naming differs
    try:
        parts = stem.replace("merged_features_", "").split("_")
        return parts[0]
    except Exception:
        return "UNK"

# -------------------- Feature building --------------------
def build_features_grouped(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    players_df: concatenation of all player CSVs for a single game,
    with columns ['minute_start','minute_end','player_id','position','group', <features...>]
    Returns one row per [minute_start, minute_end] with per-group aggregated features + counts.
    """
    key_cols = ["minute_start", "minute_end", "player_id", "position", "group"]
    feature_cols = [c for c in players_df.columns if c not in key_cols]

    rows = []
    for (start, end), window in players_df.groupby(["minute_start", "minute_end"], sort=True):
        row = {"minute_start": start, "minute_end": end}
        # counts per group
        for g in ALL_GROUPS:
            row[f"count_{g}"] = int((window["group"] == g).sum())

        # per-group MEAN for every feature (consistent with your earlier averaging)
        for g in ALL_GROUPS:
            sub = window[window["group"] == g]
            if len(sub) == 0:
                # fill zeros to keep column set consistent
                for c in feature_cols:
                    row[f"{c}__{g}"] = 0.0
            else:
                means = sub[feature_cols].mean(numeric_only=True)
                for c in feature_cols:
                    row[f"{c}__{g}"] = float(means.get(c, 0.0))

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["minute_start", "minute_end"]).reset_index(drop=True)

def load_all_games_grouped(root_folder: str) -> pd.DataFrame:
    """
    root_folder layout:
      /.../5min_windows/
        YYYY-MM-DD/               # game folder
          merged_features_AM_21.csv
          merged_features_CB_2.csv
          ...
          poss.csv
    """
    root = Path(root_folder)
    all_games = []

    for game in root.iterdir():
        if not game.is_dir():
            continue

        parts = []
        for f in game.glob("merged_features_*.csv"):
            df = pd.read_csv(f)
            stem = f.stem  # e.g., merged_features_AM_21
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
        poss["minute_start"] = poss["time_start_sec"] // 60
        poss["minute_end"]   = poss["time_end_sec"]   // 60

        merged = feats.merge(
            poss[["minute_start", "minute_end", "maccabi_haifa_possession_percent"]],
            on=["minute_start", "minute_end"],
            how="inner"
        )
        merged["game"] = game.name
        all_games.append(merged)

    return pd.concat(all_games, ignore_index=True) if all_games else pd.DataFrame()

# -------------------- Train --------------------
if __name__ == "__main__":
    DATA_ROOT = "/home/stav.karasik/MaccabiProject/scripts/train_poss_subs/5min_windows"

    data = load_all_games_grouped(DATA_ROOT)
    if data.empty:
        print("❌ No data loaded. Check paths."); exit(1)

    # Drop keys & target from features
    target_col = "maccabi_haifa_possession_percent"
    drop_cols = ["minute_start", "minute_end", "game", target_col]
    X = data.drop(columns=drop_cols)
    y = data[target_col].values
    feature_names = X.columns.tolist()

    # (Optional) emphasize low-possession windows
    weight_scale = 0.8
    weights = np.where(y < 0.5, 1.0 + weight_scale * (0.5 - y) / 0.5, 1.0)

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )

    model = CatBoostRegressor(
        iterations=1500,
        learning_rate=0.05,
        depth=8,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        early_stopping_rounds=100,
        verbose=100
    )

    model.fit(X_train, y_train, sample_weight=w_train, eval_set=(X_test, y_test))

    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds)
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    print(f"\n📊 Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

    joblib.dump(model, "possession_catboost_grouped.pkl")
    joblib.dump(feature_names, "catboost_grouped_features.pkl")
    print("✅ Saved model → possession_catboost_grouped.pkl")

    # Quick example plot on a single game
    sample = data[data["game"] == data["game"].iloc[0]].copy()
    X_s = sample[feature_names]
    y_true = sample[target_col].values
    y_pred = model.predict(X_s)

    plt.figure(figsize=(10, 4))
    plt.plot(sample["minute_start"], y_true, label="True")
    plt.plot(sample["minute_start"], y_pred, "--", label="Pred")
    plt.xlabel("Minute"); plt.ylabel("Possession %"); plt.legend()
    plt.tight_layout()
    plt.savefig("catboost_grouped_features_plot.png")
    plt.close()
    print("📈 Example plot → catboost_grouped_features_plot.png")
