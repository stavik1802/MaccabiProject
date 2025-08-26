#!/usr/bin/env python3
"""
Season1 vs Season2 cross-validation for possession regression (CatBoost, GROUPED features).

Grouping:
  G1: CB
  G2: CM, DM
  G3: RB, RW, RM, LB, LW, LM, UNK
  G4: AM, CF

For each [minute_start, minute_end] window we create:
  - count_G1..count_G4               (players per group in that window)
  - f__G1, f__G2, f__G3, f__G4       (mean of base feature f across players in each group)

Protocol per season:
  - Shuffle deterministically and make disjoint folds of size `fold_size`.
  - If not divisible by `fold_size`, the LAST FOLD IS SMALLER so every game is evaluated.
  - Train on ALL remaining games (both seasons), excluding the test fold.
  - Evaluate on the held-out fold.

Inputs:
- --games_root: root folder with one subfolder per game (e.g., dates)
- --games_map_json: JSON {"season1": [...], "season2": [...]}

Outputs (under --out_dir):
- folds.csv, summary.csv, effective_games.json
- season_mae.png, season_rmse.png, season_mae_boxplot.png
- all_fold_predictions.csv
- season_mae_true_025_075.csv, season_interval_mae_025_075.png
- mismatch_counts_per_game.csv, mismatch_avg_flips_by_season.csv,
  season_mismatch_avg_flips.png
- late-window equivalents (60–90) and per-side-of-0.5 plots
- predictions/*.csv  (if --save_preds)
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor

# ===== Helpers: MAE when true<0.5 vs true>0.5 (overall & late window) =====
def _mae_true_halves_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    df must contain columns: group_col, true, pred
    Returns wide table with columns: group_col, mae_true_lt_0_5, mae_true_gt_0_5
    """
    if df.empty:
        return pd.DataFrame(columns=[group_col, "mae_true_lt_0_5", "mae_true_gt_0_5"])

    tmp = df.copy()
    tmp["abs_err"] = (tmp["true"] - tmp["pred"]).abs()
    tmp["true_bin"] = np.where(tmp["true"] < 0.5, "true<0.5", "true>0.5")
    agg = (tmp.groupby([group_col, "true_bin"])["abs_err"]
              .mean()
              .reset_index())
    pivot = agg.pivot(index=group_col, columns="true_bin", values="abs_err").reset_index()
    pivot = pivot.rename(columns={
        "true<0.5": "mae_true_lt_0_5",
        "true>0.5": "mae_true_gt_0_5"
    })
    for c in ["mae_true_lt_0_5", "mae_true_gt_0_5"]:
        if c not in pivot.columns:
            pivot[c] = np.nan
    return pivot

def _order_for_plot(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    pref = {
        "venue": ["home", "away"],
        "group": ["top", "other"],
        "season": ["season1", "season2"],
    }.get(group_col, list(df[group_col]))
    order = [g for g in pref if g in set(df[group_col])]
    return df.set_index(group_col).loc[order].reset_index() if order else df

def plot_mae_true_halves(all_preds_df: pd.DataFrame,
                         group_col: str,
                         out_path: Path,
                         prefix: str,
                         start_min: int | None = None,
                         end_min: int | None = None) -> None:
    """
    If start_min/end_min are provided, restrict to minute_start in [start_min, end_min).
    Saves CSV and PNG (bar with two bars per group).
    """
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
    title_prefix = {
        "venue": "Home vs Away",
        "group": "Top vs Other",
        "season": "Season1 vs Season2",
    }.get(group_col, group_col.capitalize())
    plt.title(f"{title_prefix} — MAE by true side of 0.5{title_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / png_name, dpi=150)
    plt.close()
    print(f"🖼️ Saved plot → {out_path / png_name}")

# ======================
# Grouping config
# ======================
POS2GROUP = {
    "CB": "G1",
    "CM": "G2", "DM": "G2",
    "RB": "G3", "RW": "G3", "RM": "G3", "LB": "G3", "LW": "G3", "LM": "G3",
    "AM": "G4", "CF": "G4",
    "UNK": "G3",  # UNK treated as Group 3
}
ALL_GROUPS = ["G1", "G2", "G3", "G4"]

def _pos_from_stem(stem: str) -> str:
    # e.g. "merged_features_AM_21" -> "AM"
    try:
        p = stem.replace("merged_features_", "").split("_")[0]
        return p if p else "UNK"
    except Exception:
        return "UNK"

# ======================
# 1) Grouped feature builder
# ======================
def build_features_grouped(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each [minute_start, minute_end]:
      1) collapse multiple rows per player within the window by summing base features
      2) aggregate per group:
         - count_Gk = number of players of group k
         - f__Gk     = mean of base feature f across players in group k
    """
    key_cols = ["minute_start", "minute_end", "player_id", "group", "position"]
    base_cols = [c for c in players_df.columns if c not in key_cols]

    rows = []
    for (start, end), window in players_df.groupby(["minute_start", "minute_end"], sort=True):
        # collapse per player (sum across fragments inside the same window)
        per_player = (window.groupby(["player_id", "group"], as_index=False)[base_cols]
                             .sum(numeric_only=True))
        out = {"minute_start": start, "minute_end": end}

        # counts
        for g in ALL_GROUPS:
            out[f"count_{g}"] = int((per_player["group"] == g).sum())

        # means per base feature per group
        for g in ALL_GROUPS:
            sub = per_player[per_player["group"] == g]
            if len(sub):
                means = sub[base_cols].mean(numeric_only=True)
            else:
                means = pd.Series({c: 0.0 for c in base_cols}, dtype=float)
            for c in base_cols:
                out[f"{c}__{g}"] = float(means.get(c, 0.0))
        rows.append(out)

    return (pd.DataFrame(rows)
            .sort_values(["minute_start", "minute_end"])
            .reset_index(drop=True))

def load_game_df_grouped(game_folder: Path) -> pd.DataFrame:
    # collect player files + attach position/group from filename
    parts = []
    for f in game_folder.glob('merged_features_*.csv'):
        df = pd.read_csv(f)
        stem = f.stem  # e.g. merged_features_AM_21
        pos  = _pos_from_stem(stem)
        pos  = pos if pos in POS2GROUP else "UNK"
        grp  = POS2GROUP[pos]
        df["player_id"] = stem.replace("merged_features_", "")
        df["position"]  = pos
        df["group"]     = grp
        parts.append(df)
    if not parts:
        return pd.DataFrame()

    feats = build_features_grouped(pd.concat(parts, ignore_index=True))

    # possession labels
    poss_path = game_folder / 'poss.csv'
    if not poss_path.exists():
        return pd.DataFrame()
    poss = pd.read_csv(poss_path)
    poss['minute_start'] = poss['time_start_sec'] // 60
    poss['minute_end']   = poss['time_end_sec']   // 60

    merged = feats.merge(
        poss[['minute_start', 'minute_end', 'maccabi_haifa_possession_percent']],
        on=['minute_start', 'minute_end'],
        how='inner'
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
        gdf['game'] = game_folder.name
        dfs.append(gdf)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ======================
# 2) CV helpers
# ======================
def get_feature_columns(df: pd.DataFrame) -> List[str]:
    drop_cols = ['minute_start','minute_end','maccabi_haifa_possession_percent','game']
    return [c for c in df.columns if c not in drop_cols]

def make_model(seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=seed,
        early_stopping_rounds=50,
        verbose=False
    )

def sample_weights(y: np.ndarray, weight_scale: float = 0.8) -> np.ndarray:
    # same weighting strategy (you can set weight_scale=0.0 to disable)
    return np.where(y < 0.5, 1.0 + weight_scale * (0.5 - y) / 0.5, 1.0)

def train_eval_once(train_df: pd.DataFrame,
                    test_df: pd.DataFrame,
                    feature_cols: List[str],
                    seed: int,
                    return_preds: bool = False) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
    # Align feature columns
    train_X = train_df.reindex(columns=feature_cols, fill_value=0.0)
    test_X  = test_df.reindex(columns=feature_cols, fill_value=0.0)
    y_train = train_df['maccabi_haifa_possession_percent'].values
    y_test  = test_df['maccabi_haifa_possession_percent'].values

    # You used weights=0.0 in your last script; keep that behavior here:
    w_train = sample_weights(y_train, 0.0)

    model = make_model(seed)
    model.fit(train_X, y_train, sample_weight=w_train, eval_set=(test_X, y_test))

    y_pred = model.predict(test_X)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    metrics = {"rmse": rmse, "mae": mae, "r2": r2, "n_windows": int(len(test_df))}
    preds_df = None
    if return_preds:
        preds_df = test_df[['game','minute_start','minute_end']].copy()
        preds_df['true'] = y_test
        preds_df['pred'] = y_pred

    return metrics, preds_df

def folds_cover_all(items: List[str], fold_size: int) -> List[List[str]]:
    """
    Disjoint folds of size `fold_size`; keep the remainder as a final smaller fold.
    Ensures every item appears exactly once across folds.
    """
    if fold_size <= 0:
        raise ValueError("fold_size must be >= 1")
    folds = [items[i:i+fold_size] for i in range(0, len(items), fold_size)]
    return [f for f in folds if len(f) > 0]

# ======================
# 3) Main routine
# ======================
def run(
    games_root: str,
    games_map_json: str,
    out_dir: str = "experiment_season_vs_season_grouped",
    fold_size: int = 5,
    seed: int = 42,
    save_preds: bool = False
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    preds_dir = out_path / "predictions"
    if save_preds:
        preds_dir.mkdir(parents=True, exist_ok=True)

    # Load once (GROUPED features)
    print(f"📥 Loading ALL games (grouped) from {games_root}…")
    data_all = load_all_games_grouped(games_root)
    if data_all.empty:
        print("❌ No data found under games_root.")
        return

    feature_cols = sorted(get_feature_columns(data_all))

    # Mapping
    print(f"📥 Loading season mapping from {games_map_json}…")
    with open(games_map_json, "r", encoding="utf-8") as f:
        games_map = json.load(f)

    # Keep only existing game folders
    existing = set(data_all['game'].unique().tolist())
    groups = ["season1", "season2"]
    effective_map = {g: sorted([x for x in set(games_map.get(g, [])) if x in existing])
                     for g in groups}

    with open(out_path / "effective_games.json", "w", encoding="utf-8") as f:
        json.dump(effective_map, f, ensure_ascii=False, indent=2)
    print(f"🗂️ Saved effective map → {out_path/'effective_games.json'}")

    rng = np.random.RandomState(seed)
    fold_rows = []
    all_preds: List[pd.DataFrame] = []

    for group in groups:
        group_games = effective_map.get(group, [])
        if len(group_games) == 0:
            print(f"⚠️ Skipping '{group}' (no games).")
            continue

        shuffled = group_games[:]
        rng.shuffle(shuffled)
        folds = folds_cover_all(shuffled, fold_size)
        print(f"🏟️ {group}: {len(group_games)} games → {len(folds)} folds "
              f"({', '.join(str(len(f)) for f in folds)} games per fold)")

        for fold_idx, test_games in enumerate(folds, start=1):
            test_mask = data_all['game'].isin(test_games)
            test_df   = data_all.loc[test_mask].copy()
            train_df  = data_all.loc[~test_mask].copy()

            # Always get predictions (we may or may not write them)
            metrics, preds_df = train_eval_once(
                train_df, test_df, feature_cols, seed, return_preds=True
            )

            # Tag and gather predictions
            preds_df["season"] = group
            preds_df["fold_idx"] = fold_idx
            all_preds.append(preds_df.copy())

            fold_rows.append({
                "season": group,
                "fold_idx": fold_idx,
                "n_test_games": len(test_games),
                "test_games": ";".join(test_games),
                **metrics
            })
            print(f"  • {group} fold {fold_idx}: "
                  f"MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
                  f"R²={metrics['r2']:.4f}, n={metrics['n_windows']}")

            if save_preds and preds_df is not None:
                preds_df.to_csv(preds_dir / f"{group}_fold{fold_idx}_preds.csv", index=False)

    if not fold_rows:
        print("❌ No folds evaluated.")
        return

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(out_path / "folds.csv", index=False)
    print(f"✅ Per-fold metrics → {out_path/'folds.csv'}")

    # Gather all per-interval predictions across folds
    if len(all_preds) == 0:
        print("ℹ️ No predictions captured; skipping extra graphs.")
        return

    all_preds_df = pd.concat(all_preds, ignore_index=True)
    all_preds_df = all_preds_df[["season","game","minute_start","minute_end","true","pred","fold_idx"]]
    all_preds_df.to_csv(out_path / "all_fold_predictions.csv", index=False)
    print(f"🧾 Saved per-interval predictions → {out_path/'all_fold_predictions.csv'}")

    # Summary table
    summary = (folds_df
               .groupby('season')
               .agg(mean_mae=('mae','mean'),
                    std_mae=('mae','std'),
                    mean_rmse=('rmse','mean'),
                    std_rmse=('rmse','std'),
                    mean_r2=('r2','mean'),
                    n_folds=('fold_idx','count'))
               .reset_index())
    # safety for NaN std (single fold)
    summary['std_mae'] = summary['std_mae'].fillna(0.0)
    summary['std_rmse'] = summary['std_rmse'].fillna(0.0)
    summary.to_csv(out_path / "summary.csv", index=False)
    print(f"📊 Summary → {out_path/'summary.csv'}")

    # ---- Plots: Season1 vs Season2 (MAE, RMSE, boxplot)
    try:
        needed = {"season1", "season2"}
        present = set(summary["season"].tolist())
        if needed.issubset(present):
            plot_df = summary.set_index("season").loc[["season1","season2"]].reset_index()

            # MAE bar
            plt.figure(figsize=(6,4))
            plt.bar(plot_df["season"], plot_df["mean_mae"], yerr=plot_df["std_mae"], capsize=6)
            plt.ylabel("Mean MAE (mistake)")
            plt.title("Season1 vs Season2 – Mean MAE (±1 std)")
            for x, y in enumerate(plot_df["mean_mae"]):
                plt.text(x, y, f"{y:.3f}", ha="center", va="bottom")
            plt.tight_layout()
            plt.savefig(out_path / "season_mae.png", dpi=150)
            plt.close()
            print(f"🖼️ Saved plot → {out_path/'season_mae.png'}")

            # RMSE bar
            plt.figure(figsize=(6,4))
            plt.bar(plot_df["season"], plot_df["mean_rmse"], yerr=plot_df["std_rmse"], capsize=6)
            plt.ylabel("Mean RMSE")
            plt.title("Season1 vs Season2 – Mean RMSE (±1 std)")
            for x, y in enumerate(plot_df["mean_rmse"]):
                plt.text(x, y, f"{y:.3f}", ha="center", va="bottom")
            plt.tight_layout()
            plt.savefig(out_path / "season_rmse.png", dpi=150)
            plt.close()
            print(f"🖼️ Saved plot → {out_path/'season_rmse.png'}")

            # Per-fold MAE boxplot
            plt.figure(figsize=(6,4))
            folds_df.boxplot(column="mae", by="season")
            plt.suptitle("")
            plt.title("Per-fold MAE by Season")
            plt.ylabel("MAE")
            plt.tight_layout()
            plt.savefig(out_path / "season_mae_boxplot.png", dpi=150)
            plt.close()
            print(f"🖼️ Saved plot → {out_path/'season_mae_boxplot.png'}")
        else:
            print("ℹ️ Skipping MAE/RMSE/box plots (need both 'season1' and 'season2').")
    except Exception as e:
        print(f"⚠️ Plotting failed: {e}")

    # Print MAE comparison
    if set(["season1","season2"]).issubset(set(summary['season'])):
        s1 = float(summary.loc[summary['season']=="season1","mean_mae"].iloc[0])
        s2 = float(summary.loc[summary['season']=="season2","mean_mae"].iloc[0])
        diff = s1 - s2
        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("Mean MAE (mistake) by season:")
        print(f"  Season1: {s1:.4f}")
        print(f"  Season2: {s2:.4f}")
        print(f"  Season1 − Season2: {diff:+.4f}")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    # ===== Graph 1: MAE restricted to true in [0.25, 0.75] per season =====
    try:
        mask_mid = (all_preds_df["true"] >= 0.25) & (all_preds_df["true"] <= 0.75)
        mid_df = all_preds_df.loc[mask_mid].copy()
        if not mid_df.empty:
            mae_mid = (mid_df
                       .assign(abs_err=(mid_df["true"] - mid_df["pred"]).abs())
                       .groupby("season")["abs_err"]
                       .mean()
                       .reset_index()
                       .rename(columns={"abs_err":"mae_mid"}))
            mae_mid.to_csv(out_path / "season_mae_true_025_075.csv", index=False)

            plt.figure(figsize=(6,4))
            plt.bar(mae_mid["season"], mae_mid["mae_mid"], capsize=6)
            for x, y in enumerate(mae_mid["mae_mid"]):
                plt.text(x, y, f"{y:.3f}", ha="center", va="bottom")
            plt.ylabel("MAE on 0.25 ≤ true ≤ 0.75")
            plt.title("Season1 vs Season2 — Mid-range MAE")
            plt.tight_layout()
            plt.savefig(out_path / "season_interval_mae_025_075.png", dpi=150)
            plt.close()
            print(f"🖼️ Saved plot → {out_path/'season_interval_mae_025_075.png'}")
        else:
            print("ℹ️ No intervals with true in [0.25, 0.75]; skipping mid-range MAE plot.")
    except Exception as e:
        print(f"⚠️ Failed to create mid-range MAE plot: {e}")

    # ===== Graph 2: Average mismatch flips per season (mean per game) =====
    try:
        df = all_preds_df.copy()
        df["over_pred_under_true"] = ((df["pred"] > 0.5) & (df["true"] < 0.5)).astype(int)
        df["under_pred_over_true"] = ((df["pred"] < 0.5) & (df["true"] > 0.5)).astype(int)

        # 1) Per-game counts
        game_counts = (df.groupby(["season","game"])
                        .agg(over_pred_under_true=("over_pred_under_true","sum"),
                             under_pred_over_true=("under_pred_over_true","sum"))
                        .reset_index())
        game_counts.to_csv(out_path / "mismatch_counts_per_game.csv", index=False)

        # 2) Per-season averages (mean per game) + std for error bars
        season_avg = (game_counts.groupby("season")[["over_pred_under_true","under_pred_over_true"]]
                                .mean()
                                .reset_index()
                                .rename(columns={
                                    "over_pred_under_true": "avg_over_pred_under_true",
                                    "under_pred_over_true": "avg_under_pred_over_true"
                                }))
        season_std = (game_counts.groupby("season")[["over_pred_under_true","under_pred_over_true"]]
                                .std()
                                .reset_index()
                                .rename(columns={
                                    "over_pred_under_true": "std_over_pred_under_true",
                                    "under_pred_over_true": "std_under_pred_over_true"
                                }))

        season_avg.to_csv(out_path / "mismatch_avg_flips_by_season.csv", index=False)
        season_std.to_csv(out_path / "mismatch_avg_flips_by_season_std.csv", index=False)

        # 3) Plot grouped bars: avg flips per game by season
        plot_df = season_avg.merge(season_std, on="season", how="left")
        if set(plot_df["season"]) >= {"season1","season2"}:
            plot_df = plot_df.set_index("season").loc[["season1","season2"]].reset_index()

        x = np.arange(len(plot_df["season"]))
        width = 0.35

        plt.figure(figsize=(7,4))
        plt.bar(
            x - width/2,
            plot_df["avg_over_pred_under_true"],
            width,
            yerr=plot_df["std_over_pred_under_true"].fillna(0.0),
            capsize=6,
            label="pred>0.5 & true<0.5"
        )
        plt.bar(
            x + width/2,
            plot_df["avg_under_pred_over_true"],
            width,
            yerr=plot_df["std_under_pred_over_true"].fillna(0.0),
            capsize=6,
            label="pred<0.5 & true>0.5"
        )
        plt.xticks(x, plot_df["season"])
        plt.ylabel("Avg flips per game")
        plt.title("Average threshold flips per game by season")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path / "season_mismatch_avg_flips.png", dpi=150)
        plt.close()
        print(f"🖼️ Saved plot → {out_path/'season_mismatch_avg_flips.png'}")
    except Exception as e:
        print(f"⚠️ Failed to create average flips plot: {e}")

    # ===== NEW: Late-window (minutes 60–90) graphs =====
    try:
        late = all_preds_df[(all_preds_df["minute_start"] >= 60) & (all_preds_df["minute_start"] < 90)].copy()
        if late.empty:
            print("ℹ️ No intervals with minute_start in [60,90); skipping late-window graphs.")
        else:
            # --- Per-fold metrics on late windows
            late["abs_err"] = (late["true"] - late["pred"]).abs()
            late["sq_err"]  = (late["true"] - late["pred"])**2
            folds_60_90 = (late.groupby(["season","fold_idx"])
                              .apply(lambda d: pd.Series({
                                  "mae": float(d["abs_err"].mean()),
                                  "rmse": float(np.sqrt(d["sq_err"].mean())),
                                  "n_windows": int(len(d))
                              }))
                              .reset_index())
            folds_60_90.to_csv(out_path / "folds_60_90.csv", index=False)

            # --- Summary (mean ± std across folds)
            summary_60_90 = (folds_60_90.groupby("season")
                             .agg(mean_mae=("mae","mean"),
                                  std_mae=("mae","std"),
                                  mean_rmse=("rmse","mean"),
                                  std_rmse=("rmse","std"),
                                  n_folds=("fold_idx","count"))
                             .reset_index())
            summary_60_90[["std_mae","std_rmse"]] = summary_60_90[["std_mae","std_rmse"]].fillna(0.0)
            summary_60_90.to_csv(out_path / "summary_60_90.csv", index=False)
            print(f"📊 Late-window summary → {out_path/'summary_60_90.csv'}")

            # --- 3 core plots for 60–90
            present = set(summary_60_90["season"])
            if {"season1","season2"}.issubset(present):
                plot_df = summary_60_90.set_index("season").loc[["season1","season2"]].reset_index()

                # MAE bar (60–90)
                plt.figure(figsize=(6,4))
                plt.bar(plot_df["season"], plot_df["mean_mae"], yerr=plot_df["std_mae"], capsize=6)
                plt.ylabel("Mean MAE (mistake) – minutes 60–90")
                plt.title("Season1 vs Season2 – Mean MAE (±1 std), 60–90")
                for x, y in enumerate(plot_df["mean_mae"]):
                    plt.text(x, y, f"{y:.3f}", ha="center", va="bottom")
                plt.tight_layout()
                plt.savefig(out_path / "season_mae_60_90.png", dpi=150)
                plt.close()
                print(f"🖼️ Saved plot → {out_path/'season_mae_60_90.png'}")

                # RMSE bar (60–90)
                plt.figure(figsize=(6,4))
                plt.bar(plot_df["season"], plot_df["mean_rmse"], yerr=plot_df["std_rmse"], capsize=6)
                plt.ylabel("Mean RMSE – minutes 60–90")
                plt.title("Season1 vs Season2 – Mean RMSE (±1 std), 60–90")
                for x, y in enumerate(plot_df["mean_rmse"]):
                    plt.text(x, y, f"{y:.3f}", ha="center", va="bottom")
                plt.tight_layout()
                plt.savefig(out_path / "season_rmse_60_90.png", dpi=150)
                plt.close()
                print(f"🖼️ Saved plot → {out_path/'season_rmse_60_90.png'}")

                # Per-fold MAE boxplot (60–90)
                plt.figure(figsize=(6,4))
                folds_60_90.boxplot(column="mae", by="season")
                plt.suptitle("")
                plt.title("Per-fold MAE by Season (minutes 60–90)")
                plt.ylabel("MAE")
                plt.tight_layout()
                plt.savefig(out_path / "season_mae_boxplot_60_90.png", dpi=150)
                plt.close()
                print(f"🖼️ Saved plot → {out_path/'season_mae_boxplot_60_90.png'}")
            else:
                print("ℹ️ Skipping late-window MAE/RMSE/box plots (need both 'season1' and 'season2').")

            # --- Late-window Graph A: Mid-range MAE (true ∈ [0.25, 0.75])
            late_mid = late[(late["true"] >= 0.25) & (late["true"] <= 0.75)].copy()
            if not late_mid.empty:
                mae_mid_60_90 = (late_mid
                                 .assign(abs_err=(late_mid["true"] - late_mid["pred"]).abs())
                                 .groupby("season")["abs_err"]
                                 .mean()
                                 .reset_index()
                                 .rename(columns={"abs_err": "mae_mid"}))
                mae_mid_60_90.to_csv(out_path / "season_mae_true_025_075_60_90.csv", index=False)

                plt.figure(figsize=(6,4))
                plt.bar(mae_mid_60_90["season"], mae_mid_60_90["mae_mid"], capsize=6)
                for x, y in enumerate(mae_mid_60_90["mae_mid"]):
                    plt.text(x, y, f"{y:.3f}", ha="center", va="bottom")
                plt.ylabel("MAE on 0.25 ≤ true ≤ 0.75 (60–90)")
                plt.title("Season1 vs Season2 — Mid-range MAE (minutes 60–90)")
                plt.tight_layout()
                plt.savefig(out_path / "season_interval_mae_025_075_60_90.png", dpi=150)
                plt.close()
                print(f"🖼️ Saved plot → {out_path/'season_interval_mae_025_075_60_90.png'}")
            else:
                print("ℹ️ No late-window intervals with true in [0.25, 0.75]; skipping mid-range MAE (60–90).")

            # --- Late-window Graph B: Average flips per game around 0.5
            df_late = late.copy()
            df_late["over_pred_under_true"] = ((df_late["pred"] > 0.5) & (df_late["true"] < 0.5)).astype(int)
            df_late["under_pred_over_true"] = ((df_late["pred"] < 0.5) & (df_late["true"] > 0.5)).astype(int)

            # Per-game counts (60–90)
            game_counts_60_90 = (df_late.groupby(["season","game"])
                                   .agg(over_pred_under_true=("over_pred_under_true","sum"),
                                        under_pred_over_true=("under_pred_over_true","sum"))
                                   .reset_index())
            game_counts_60_90.to_csv(out_path / "mismatch_counts_per_game_60_90.csv", index=False)

            # Per-season averages (mean per game) + std (60–90)
            season_avg_60_90 = (game_counts_60_90.groupby("season")[["over_pred_under_true","under_pred_over_true"]]
                                .mean()
                                .reset_index()
                                .rename(columns={
                                    "over_pred_under_true": "avg_over_pred_under_true",
                                    "under_pred_over_true": "avg_under_pred_over_true"
                                }))
            season_std_60_90 = (game_counts_60_90.groupby("season")[["over_pred_under_true","under_pred_over_true"]]
                                .std()
                                .reset_index()
                                .rename(columns={
                                    "over_pred_under_true": "std_over_pred_under_true",
                                    "under_pred_over_true": "std_under_pred_over_true"
                                }))

            season_avg_60_90.to_csv(out_path / "mismatch_avg_flips_by_season_60_90.csv", index=False)
            season_std_60_90.to_csv(out_path / "mismatch_avg_flips_by_season_60_90_std.csv", index=False)

            # Plot grouped bars: avg flips per game by season (60–90)
            plot_df = season_avg_60_90.merge(season_std_60_90, on="season", how="left")
            if set(["season1","season2"]).issubset(set(plot_df["season"])):
                plot_df = plot_df.set_index("season").loc[["season1","season2"]].reset_index()

            x = np.arange(len(plot_df["season"]))
            width = 0.35

            plt.figure(figsize=(7,4))
            plt.bar(
                x - width/2,
                plot_df["avg_over_pred_under_true"],
                width,
                yerr=plot_df["std_over_pred_under_true"].fillna(0.0),
                capsize=6,
                label="pred>0.5 & true<0.5"
            )
            plt.bar(
                x + width/2,
                plot_df["avg_under_pred_over_true"],
                width,
                yerr=plot_df["std_under_pred_over_true"].fillna(0.0),
                capsize=6,
                label="pred<0.5 & true>0.5"
            )
            plt.xticks(x, plot_df["season"])
            plt.ylabel("Avg flips per game (60–90)")
            plt.title("Average threshold flips per game by season (minutes 60–90)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_path / "season_mismatch_avg_flips_60_90.png", dpi=150)
            plt.close()
            print(f"🖼️ Saved plot → {out_path/'season_mismatch_avg_flips_60_90.png'}")
    except Exception as e:
        print(f"⚠️ Failed to create late-window graphs: {e}")

    # --- MAE by true side of 0.5 (overall and 60–90) ---
    plot_mae_true_halves(all_preds_df, group_col="season",
                         out_path=out_path, prefix="season")
    plot_mae_true_halves(all_preds_df, group_col="season",
                         out_path=out_path, prefix="season",
                         start_min=60, end_min=90)

# ======================
# Entrypoint
# ======================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--games_root", type=str, default="5min_windows",
                   help="Root folder with one subfolder per game (e.g., dates)")
    p.add_argument("--games_map_json", type=str, required=True,
                   help="Path to JSON with {'season1': [...], 'season2': [...]} game folder names")
    p.add_argument("--out_dir", type=str, default="experiment_season_vs_season_grouped",
                   help="Output directory for all artifacts")
    p.add_argument("--fold_size", type=int, default=5, help="Games per test fold (last fold may be smaller)")
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
