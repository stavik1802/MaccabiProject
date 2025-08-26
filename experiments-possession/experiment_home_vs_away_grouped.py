#!/usr/bin/env python3
"""
Home/Away CV for possession regression (CatBoost) using GROUPED FEATURES.
- Player positions are parsed from filenames (merged_features_{POS}_{id}.csv)
- Positions are mapped to 4 groups; UNK is assigned to Group 3.
- For each window [minute_start, minute_end], we aggregate per-group MEANS
  for every numeric feature and include per-group player COUNTS.

Usage:
  python eval_home_away_cv_userdict_grouped.py \
      --games_root 5min_windows \
      --games_map_json home_away_map.json \
      --out_dir experiment_home_vs_away_grouped \
      --fold_size 5 \
      --seed 42

JSON format (folder names must match subfolders under --games_root):
{
  "home": ["2023-09-03", "2023-10-01", "..."],
  "away": ["2023-09-24", "2023-10-08", "..."]
}
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

# ======================
# Group mapping (NEW) — UNK -> G3
# ======================
POS2GROUP = {
    "CB": "G1",
    "CM": "G2", "DM": "G2",
    "RB": "G3", "RW": "G3", "RM": "G3", "LB": "G3", "LW": "G3", "LM": "G3",
    "AM": "G4", "CF": "G4",
    "UNK": "G3",
}
ALL_GROUPS = ["G1", "G2", "G3", "G4"]

def parse_pos_from_stem(stem: str) -> str:
    # stem example: "merged_features_AM_21" -> "AM"
    try:
        return stem.replace("merged_features_", "").split("_")[0]
    except Exception:
        return "error stem"

# ===== Helpers: MAE when true<0.5 vs true>0.5 (overall & late window) =====
def _mae_true_halves_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[group_col, "mae_true_lt_0_5", "mae_true_gt_0_5"])
    tmp = df.copy()
    tmp["abs_err"] = (tmp["true"] - tmp["pred"]).abs()
    tmp["true_bin"] = np.where(tmp["true"] < 0.5, "true<0.5", "true>0.5")
    agg = (tmp.groupby([group_col, "true_bin"])["abs_err"].mean().reset_index())
    pivot = agg.pivot(index=group_col, columns="true_bin", values="abs_err").reset_index()
    pivot = pivot.rename(columns={"true<0.5": "mae_true_lt_0_5", "true>0.5": "mae_true_gt_0_5"})
    for c in ["mae_true_lt_0_5", "mae_true_gt_0_5"]:
        if c not in pivot.columns:
            pivot[c] = np.nan
    return pivot

def _order_for_plot(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    pref = {"venue": ["home", "away"]}.get(group_col, list(df[group_col]))
    order = [g for g in pref if g in set(df[group_col])]
    return df.set_index(group_col).loc[order].reset_index() if order else df

def plot_mae_true_halves(all_preds_df: pd.DataFrame,
                         group_col: str,
                         out_path: Path,
                         prefix: str,
                         start_min: int | None = None,
                         end_min: int | None = None) -> None:
    df = all_preds_df
    if start_min is not None and end_min is not None:
        df = df[(df["minute_start"] >= start_min) & (df["minute_start"] < end_min)].copy()
        suffix = f"_{start_min}_{end_min}"
        title_suffix = f" (minutes {start_min}–{end_min})"
    else:
        suffix = ""
        title_suffix = ""

    tbl = _mae_true_halves_table(df, group_col)
    csv_name = f"{prefix}_mae_true_halves{suffix}.csv"
    png_name = f"{prefix}_mae_true_halves{suffix}.png"
    tbl.to_csv(out_path / csv_name, index=False)

    if tbl.empty:
        print(f"ℹ️ No data for {png_name}; skipped.")
        return

    plot_df = _order_for_plot(tbl, group_col)
    x = np.arange(len(plot_df[group_col])); width = 0.35

    plt.figure(figsize=(7,4))
    plt.bar(x - width/2, plot_df["mae_true_lt_0_5"], width, label="true < 0.5")
    plt.bar(x + width/2, plot_df["mae_true_gt_0_5"], width, label="true > 0.5")
    plt.xticks(x, plot_df[group_col])
    plt.ylabel("MAE")
    title_prefix = {"venue": "Home vs Away"}.get(group_col, group_col.capitalize())
    plt.title(f"{title_prefix} — MAE by true side of 0.5{title_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / png_name, dpi=150)
    plt.close()
    print(f"🖼️ Saved plot → {out_path / png_name}")

# ======================
# 1) GROUPED feature builder (NEW)
# ======================
def build_features_grouped(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    players_df: concat of all player CSVs for one game with columns
      ['minute_start','minute_end',<feature...>,'player_id','position','group']
    Returns one row per [minute_start, minute_end] with:
      - count_Gk for k=1..4
      - per-group MEAN for each numeric feature: {feature}__Gk
    """
    key_cols = ["minute_start", "minute_end", "player_id", "position", "group"]
    feature_cols = [c for c in players_df.columns if c not in key_cols]

    rows = []
    for (start, end), window in players_df.groupby(["minute_start", "minute_end"], sort=True):
        row = {"minute_start": start, "minute_end": end}
        # counts per group
        for g in ALL_GROUPS:
            row[f"count_{g}"] = int((window["group"] == g).sum())
        # per-group means for every numeric feature
        for g in ALL_GROUPS:
            sub = window[window["group"] == g]
            means = sub[feature_cols].mean(numeric_only=True) if len(sub) else pd.Series(dtype=float)
            for c in feature_cols:
                row[f"{c}__{g}"] = float(means.get(c, 0.0))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["minute_start", "minute_end"]).reset_index(drop=True)

def load_game_df_grouped(game_folder: Path) -> pd.DataFrame:
    parts = []
    for f in game_folder.glob("merged_features_*.csv"):
        df = pd.read_csv(f)
        stem = f.stem
        pos = parse_pos_from_stem(stem) or "UNK"
        pos = pos if pos in POS2GROUP else "UNK"
        grp = POS2GROUP[pos]  # UNK -> G3 here
        df["player_id"] = stem.replace("merged_features_", "")
        df["position"]  = pos
        df["group"]     = grp
        parts.append(df)
    if not parts:
        return pd.DataFrame()

    players_df = pd.concat(parts, ignore_index=True)
    feats = build_features_grouped(players_df)

    poss_path = game_folder / "poss.csv"
    if not poss_path.exists():
        return pd.DataFrame()
    poss = pd.read_csv(poss_path)
    poss["minute_start"] = poss["time_start_sec"] // 60
    poss["minute_end"]   = poss["time_end_sec"]   // 60

    merged = feats.merge(
        poss[["minute_start", "minute_end", "maccabi_haifa_possession_percent"]],
        on=["minute_start", "minute_end"],
        how="inner"
    )
    return merged

def load_all_games_grouped(root_folder: str) -> pd.DataFrame:
    root = Path(root_folder)
    dfs = []
    for game_folder in root.iterdir():
        if not game_folder.is_dir():
            continue
        gdf = load_game_df_grouped(game_folder)
        if gdf.empty:
            continue
        gdf["game"] = game_folder.name
        dfs.append(gdf)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ======================
# 2) CV helpers (unchanged)
# ======================
def get_feature_columns(df: pd.DataFrame) -> List[str]:
    drop_cols = ["minute_start","minute_end","maccabi_haifa_possession_percent","game"]
    return [c for c in df.columns if c not in drop_cols]

def make_model(seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=1200,
        learning_rate=0.05,
        depth=8,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=seed,
        early_stopping_rounds=100,
        verbose=False,
    )

def sample_weights(y: np.ndarray, weight_scale: float = 0.8) -> np.ndarray:
    return np.where(y < 0.5, 1.0 + weight_scale * (0.5 - y) / 0.5, 1.0)

def train_eval_once(train_df: pd.DataFrame,
                    test_df: pd.DataFrame,
                    feature_cols: List[str],
                    seed: int,
                    return_preds: bool = True):
    train_X = train_df.reindex(columns=feature_cols, fill_value=0.0)
    test_X  = test_df.reindex(columns=feature_cols, fill_value=0.0)
    y_train = train_df["maccabi_haifa_possession_percent"].values
    y_test  = test_df["maccabi_haifa_possession_percent"].values

    w_train = sample_weights(y_train, 0.0)  # turn on if you want weighting
    model = make_model(seed)
    model.fit(train_X, y_train, sample_weight=w_train, eval_set=(test_X, y_test))

    y_pred = model.predict(test_X)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))
    metrics = {"rmse": rmse, "mae": mae, "r2": r2, "n_windows": int(len(test_df))}

    preds_df = test_df[["game","minute_start","minute_end"]].copy()
    preds_df["true"] = y_test
    preds_df["pred"] = y_pred
    return metrics, preds_df

def chunk_folds(items: List[str], fold_size: int) -> List[List[str]]:
    return [items[i:i+fold_size] for i in range(0, len(items) - (len(items) % fold_size), fold_size)]

# ======================
# 3) Main routine (uses GROUPED loader)
# ======================
def run(
    games_root: str,
    games_map_json: str,
    out_dir: str = "experiment_home_vs_away_grouped",
    fold_size: int = 5,
    seed: int = 42,
    save_preds: bool = False
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    preds_dir = out_path / "predictions"
    if save_preds:
        preds_dir.mkdir(parents=True, exist_ok=True)

    # Load all windows once (GROUPED)
    print(f"📥 Loading ALL games (grouped features) from {games_root}…")
    data_all = load_all_games_grouped(games_root)
    if data_all.empty:
        print("❌ No data found under games_root."); return

    feature_cols = sorted(get_feature_columns(data_all))

    # Load user-provided mapping and validate
    print(f"📥 Loading user game mapping from {games_map_json}…")
    with open(games_map_json, "r", encoding="utf-8") as f:
        games_map = json.load(f)

    existing_games = set(data_all["game"].unique().tolist())
    effective_map = {k: sorted([g for g in set(v) if g in existing_games])
                     for k, v in games_map.items() if k in ("home","away")}

    with open(out_path / "effective_games.json", "w", encoding="utf-8") as f:
        json.dump(effective_map, f, ensure_ascii=False, indent=2)
    print(f"🗂️ Saved effective map → {out_path/'effective_games.json'}")

    rng = np.random.RandomState(seed)
    fold_rows, all_preds = [], []

    for venue in ["home", "away"]:
        if venue not in effective_map or len(effective_map[venue]) < fold_size:
            print(f"⚠️ Skipping venue '{venue}' (need ≥{fold_size} games)."); continue

        games = effective_map[venue][:]
        rng.shuffle(games)
        folds = chunk_folds(games, fold_size)
        print(f"🏟️ {venue}: {len(effective_map[venue])} games → {len(folds)} folds of {fold_size}")

        for fold_idx, test_games in enumerate(folds, start=1):
            test_mask = data_all["game"].isin(test_games)
            test_df   = data_all.loc[test_mask].copy()
            train_df  = data_all.loc[~test_mask].copy()

            metrics, preds_df = train_eval_once(train_df, test_df, feature_cols, seed, return_preds=True)
            preds_df["venue"] = venue; preds_df["fold_idx"] = fold_idx
            all_preds.append(preds_df.copy())

            fold_rows.append({
                "venue": venue, "fold_idx": fold_idx,
                "n_test_games": len(test_games),
                "test_games": ";".join(test_games), **metrics
            })
            print(f"  • {venue} fold {fold_idx}: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}, n={metrics['n_windows']}")

            if save_preds and preds_df is not None:
                preds_df.to_csv(preds_dir / f"{venue}_fold{fold_idx}_preds.csv", index=False)

    if not fold_rows:
        print("❌ No folds evaluated."); return

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(out_path / "folds.csv", index=False)
    print(f"✅ Per-fold metrics → {out_path/'folds.csv'}")

    summary = (folds_df.groupby("venue")
               .agg(mean_mae=("mae","mean"), std_mae=("mae","std"),
                    mean_rmse=("rmse","mean"), std_rmse=("rmse","std"),
                    mean_r2=("r2","mean"), n_folds=("fold_idx","count"))
               .reset_index())
    summary[["std_mae","std_rmse"]] = summary[["std_mae","std_rmse"]].fillna(0.0)
    summary.to_csv(out_path / "summary.csv", index=False)
    print(f"📊 Summary → {out_path/'summary.csv'}")

    # Plot Home vs Away MAE with error bars
    try:
        if {"home","away"}.issubset(set(summary["venue"])):
            plot_df = summary.set_index("venue").loc[["home","away"]].reset_index()
            plt.figure(figsize=(6,4))
            plt.bar(plot_df["venue"], plot_df["mean_mae"], yerr=plot_df["std_mae"], capsize=6)
            plt.ylabel("Mean MAE")
            plt.title("Home vs Away – Mean MAE (±1 std)")
            for x, y in enumerate(plot_df["mean_mae"]):
                plt.text(x, y, f"{y:.3f}", ha="center", va="bottom")
            plt.tight_layout()
            plt.savefig(out_path / "home_away_mae.png", dpi=150)
            plt.close()
            print(f"🖼️ Saved plot → {out_path/'home_away_mae.png'}")
    except Exception as e:
        print(f"⚠️ MAE plot failed: {e}")

    # Print quick comparison
    if {"home","away"}.issubset(set(summary["venue"])):
        home_mae = float(summary.loc[summary["venue"]=="home","mean_mae"].iloc[0])
        away_mae = float(summary.loc[summary["venue"]=="away","mean_mae"].iloc[0])
        diff = away_mae - home_mae
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("Mean MAE by venue:")
        print(f"  Home: {home_mae:.4f}")
        print(f"  Away: {away_mae:.4f}")
        print(f"  Away − Home: {diff:+.4f}")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # Collect all predictions for extra graphs
    if not all_preds:
        return
    all_preds_df = pd.concat(all_preds, ignore_index=True)
    all_preds_df = all_preds_df[["venue","game","minute_start","minute_end","true","pred","fold_idx"]]
    out_path.joinpath("all_fold_predictions.csv").write_text(all_preds_df.to_csv(index=False))
    print(f"🧾 Saved per-interval predictions → {out_path/'all_fold_predictions.csv'}")

    # Mid-range MAE and late-window analyses (reuse helpers)
    try:
        # overall mid-range
        mask_mid = (all_preds_df["true"] >= 0.25) & (all_preds_df["true"] <= 0.75)
        mid_df = all_preds_df.loc[mask_mid].copy()
        if not mid_df.empty:
            mae_mid = (mid_df.assign(abs_err=(mid_df["true"] - mid_df["pred"]).abs())
                       .groupby("venue")["abs_err"].mean().reset_index()
                       .rename(columns={"abs_err":"mae_mid"}))
            mae_mid.to_csv(out_path / "home_away_mae_true_025_075.csv", index=False)
            plt.figure(figsize=(6,4))
            plt.bar(mae_mid["venue"], mae_mid["mae_mid"], capsize=6)
            for x, y in enumerate(mae_mid["mae_mid"]):
                plt.text(x, y, f"{y:.3f}", ha="center", va="bottom")
            plt.ylabel("MAE on 0.25 ≤ true ≤ 0.75")
            plt.title("Home vs Away — Mid-range MAE")
            plt.tight_layout(); plt.savefig(out_path / "home_away_interval_mae_025_075.png", dpi=150); plt.close()

        # late-window suite
        late = all_preds_df[(all_preds_df["minute_start"] >= 60) & (all_preds_df["minute_start"] < 90)].copy()
        if not late.empty:
            late["abs_err"] = (late["true"] - late["pred"]).abs()
            late["sq_err"]  = (late["true"] - late["pred"])**2
            folds_60_90 = (late.groupby(["venue","fold_idx"])
                           .apply(lambda d: pd.Series({
                               "mae": float(d["abs_err"].mean()),
                               "rmse": float(np.sqrt(d["sq_err"].mean())),
                               "n_windows": int(len(d))
                           })).reset_index())
            folds_60_90.to_csv(out_path / "folds_60_90.csv", index=False)

            summary_60_90 = (folds_60_90.groupby("venue")
                             .agg(mean_mae=("mae","mean"), std_mae=("mae","std"),
                                  mean_rmse=("rmse","mean"), std_rmse=("rmse","std"),
                                  n_folds=("fold_idx","count")).reset_index())
            summary_60_90[["std_mae","std_rmse"]] = summary_60_90[["std_mae","std_rmse"]].fillna(0.0)
            summary_60_90.to_csv(out_path / "summary_60_90.csv", index=False)

            if {"home","away"}.issubset(set(summary_60_90["venue"])):
                plot_df = summary_60_90.set_index("venue").loc[["home","away"]].reset_index()
                plt.figure(figsize=(6,4))
                plt.bar(plot_df["venue"], plot_df["mean_mae"], yerr=plot_df["std_mae"], capsize=6)
                plt.ylabel("Mean MAE – minutes 60–90")
                plt.title("Home vs Away – Mean MAE (±1 std), 60–90")
                for x, y in enumerate(plot_df["mean_mae"]):
                    plt.text(x, y, f"{y:.3f}", ha="center", va="bottom")
                plt.tight_layout(); plt.savefig(out_path / "home_away_mae_60_90.png", dpi=150); plt.close()

                plt.figure(figsize=(6,4))
                plt.bar(plot_df["venue"], plot_df["mean_rmse"], yerr=plot_df["std_rmse"], capsize=6)
                plt.ylabel("Mean RMSE – minutes 60–90")
                plt.title("Home vs Away – Mean RMSE (±1 std), 60–90")
                for x, y in enumerate(plot_df["mean_rmse"]):
                    plt.text(x, y, f"{y:.3f}", ha="center", va="bottom")
                plt.tight_layout(); plt.savefig(out_path / "home_away_rmse_60_90.png", dpi=150); plt.close()

        # MAE by true side of 0.5 (overall and 60–90)
        plot_mae_true_halves(all_preds_df, group_col="venue", out_path=out_path, prefix="home_away")
        plot_mae_true_halves(all_preds_df, group_col="venue", out_path=out_path, prefix="home_away",
                             start_min=60, end_min=90)
    except Exception as e:
        print(f"⚠️ Extra graphs failed: {e}")

# ======================
# Entrypoint
# ======================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--games_root", type=str, default="5min_windows",
                   help="Root folder with one subfolder per game (e.g., dates)")
    p.add_argument("--games_map_json", type=str, required=True,
                   help="Path to JSON with {'home': [...], 'away': [...]} folder names")
    p.add_argument("--out_dir", type=str, default="experiment_home_vs_away_grouped",
                   help="Output directory for artifacts")
    p.add_argument("--fold_size", type=int, default=5, help="Games per test fold")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--save_preds", action="store_true", help="Save per-fold prediction CSVs")
    args = p.parse_args()

    run(
        games_root=args.games_root,
        games_map_json=args.games_map_json,
        out_dir=args.out_dir,
        fold_size=args.fold_size,
        seed=args.seed,
        save_preds=args.save_preds
    )
