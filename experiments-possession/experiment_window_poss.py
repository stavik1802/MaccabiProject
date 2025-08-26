#!/usr/bin/env python3
"""
Window-size comparison (1m / 3m / 5m) for possession regression (CatBoost, GROUPED features).

Grouping:
  G1: CB
  G2: CM, DM
  G3: RB, RW, RM, LB, LW, LM, UNK
  G4: AM, CF

Inputs:
  --root_1min    : folder with ALL games processed at 1-minute windows (not split)
  --root_3min    : folder with ALL games processed at 3-minute windows (not split)
  --train_5min   : folder with TRAIN games processed at 5-minute windows
  --eval_5min    : folder with EVAL  games processed at 5-minute windows

The eval game list is taken from --eval_5min folder names.
For 1m/3m: eval = intersection(with their roots, by folder name); train = others.

Outputs under --out_dir (default: experiment_window_comparison_grouped):
  - predictions/<win>_preds.csv
  - summary.csv (one row per window size with all metrics)
  - mae_by_window.png
  - flips_by_window.png   (two bars per window: over_pred_under_true, under_pred_over_true)
  - mae_midrange_by_window.png     (true in [0.25, 0.75])
  - mae_60_90_by_window.png        (minutes 60–90 only)
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import argparse
import sys

# -----------------------------
# Position → Group mapping
# -----------------------------
POS2GROUP = {
    "CB": "G1",
    "CM": "G2", "DM": "G2",
    "RB": "G3", "RW": "G3", "RM": "G3", "LB": "G3", "LW": "G3", "LM": "G3",
    "AM": "G4", "CF": "G4",
    "UNK": "G3",  # default bucket
}
ALL_GROUPS = ["G1", "G2", "G3", "G4"]


def _pos_from_stem(stem: str) -> str:
    # e.g., "merged_features_AM_21" -> "AM"
    try:
        p = stem.replace("merged_features_", "").split("_")[0]
        return p if p else "UNK"
    except Exception:
        return "UNK"


# -----------------------------
# Grouped feature building
# -----------------------------
def build_features_grouped(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each [minute_start, minute_end]:
      1) collapse rows per player inside the window by summing base features
      2) aggregate per group:
         - count_Gk
         - mean(base_feature)_per_group -> as columns `feature__Gk`
    """
    key_cols = ["minute_start", "minute_end", "player_id", "group", "position"]
    base_cols = [c for c in players_df.columns if c not in key_cols]

    rows = []
    for (start, end), window in players_df.groupby(["minute_start", "minute_end"], sort=True):
        per_player = (window.groupby(["player_id", "group"], as_index=False)[base_cols]
                             .sum(numeric_only=True))
        out = {"minute_start": start, "minute_end": end}

        # counts per group
        for g in ALL_GROUPS:
            out[f"count_{g}"] = int((per_player["group"] == g).sum())

        # means per base feature per group
        for g in ALL_GROUPS:
            sub = per_player[per_player["group"] == g]
            if len(sub) > 0:
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
    parts = []
    for f in game_folder.glob("merged_features_*.csv"):
        df = pd.read_csv(f)
        stem = f.stem  # merged_features_<POS>_<id>
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


def load_games_subset_grouped(root: Path, include: Optional[set]) -> pd.DataFrame:
    """
    Load only game folders whose names are in `include` (or all if include is None).
    """
    dfs = []
    for game_folder in root.iterdir():
        if not game_folder.is_dir():
            continue
        if (include is not None) and (game_folder.name not in include):
            continue
        gdf = load_game_df_grouped(game_folder)
        if gdf.empty:
            continue
        gdf["game"] = game_folder.name
        dfs.append(gdf)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def list_game_names(root: Path) -> List[str]:
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


# -----------------------------
# Modeling helpers
# -----------------------------
def get_feature_columns(df: pd.DataFrame) -> List[str]:
    drop_cols = ["minute_start", "minute_end", "maccabi_haifa_possession_percent", "game"]
    return [c for c in df.columns if c not in drop_cols]


def make_model(seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=seed,
        early_stopping_rounds=50,
        verbose=False
    )


def sample_weights(y: np.ndarray, weight_scale: float = 0.0) -> np.ndarray:
    # set weight_scale > 0 to emphasize low-possession windows
    return np.where(y < 0.5, 1.0 + weight_scale * (0.5 - y) / 0.5, 1.0)


def train_eval(train_df: pd.DataFrame,
               test_df: pd.DataFrame,
               seed: int,
               weight_scale: float = 0.0) -> Tuple[Dict[str, float], pd.DataFrame]:
    if train_df.empty or test_df.empty:
        return {}, pd.DataFrame()

    # Align columns (union)
    feature_cols = sorted(set(get_feature_columns(train_df)) | set(get_feature_columns(test_df)))
    X_train = train_df.reindex(columns=feature_cols, fill_value=0.0)
    X_test  = test_df.reindex(columns=feature_cols, fill_value=0.0)
    y_train = train_df["maccabi_haifa_possession_percent"].values
    y_test  = test_df["maccabi_haifa_possession_percent"].values

    w_train = sample_weights(y_train, weight_scale)
    model = make_model(seed)
    model.fit(X_train, y_train, sample_weight=w_train, eval_set=(X_test, y_test))

    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae  = float(mean_absolute_error(y_test, y_pred))
    r2   = float(r2_score(y_test, y_pred))

    preds = test_df[["game", "minute_start", "minute_end"]].copy()
    preds["true"] = y_test
    preds["pred"] = y_pred

    return {"rmse": rmse, "mae": mae, "r2": r2, "n_windows": int(len(preds))}, preds


# -----------------------------
# Metrics for plots
# -----------------------------
def compute_midrange_mae(preds: pd.DataFrame) -> float:
    df = preds[(preds["true"] >= 0.25) & (preds["true"] <= 0.75)]
    return float(np.mean(np.abs(df["true"] - df["pred"]))) if len(df) else float("nan")


def compute_mae_60_90(preds: pd.DataFrame) -> float:
    df = preds[(preds["minute_start"] >= 60) & (preds["minute_start"] < 90)]
    return float(np.mean(np.abs(df["true"] - df["pred"]))) if len(df) else float("nan")


def compute_flips(preds: pd.DataFrame) -> Tuple[float, float]:
    """
    Returns (avg_over_pred_under_true, avg_under_pred_over_true) averaged per game.
    """
    if preds.empty:
        return float("nan"), float("nan")

    df = preds.copy()
    df["over_pred_under_true"] = ((df["pred"] > 0.5) & (df["true"] < 0.5)).astype(int)
    df["under_pred_over_true"] = ((df["pred"] < 0.5) & (df["true"] > 0.5)).astype(int)

    per_game = (df.groupby("game")
                  .agg(over_pred_under_true=("over_pred_under_true", "sum"),
                       under_pred_over_true=("under_pred_over_true", "sum"))
                  .reset_index())

    return (float(per_game["over_pred_under_true"].mean()) if len(per_game) else float("nan"),
            float(per_game["under_pred_over_true"].mean()) if len(per_game) else float("nan"))


# -----------------------------
# Plotting
# -----------------------------
def bar_with_values(xlabels: List[str], values: List[float], ylabel: str, title: str, out_path: Path):
    plt.figure(figsize=(7, 4))
    xs = np.arange(len(xlabels))
    plt.bar(xs, values)
    plt.xticks(xs, xlabels)
    plt.ylabel(ylabel)
    plt.title(title)
    for i, v in enumerate(values):
        if not (v != v):  # not NaN
            plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"🖼️ Saved plot → {out_path}")


def grouped_two_bar(xlabels: List[str],
                    left_vals: List[float], right_vals: List[float],
                    left_label: str, right_label: str,
                    ylabel: str, title: str, out_path: Path):
    plt.figure(figsize=(8, 4))
    xs = np.arange(len(xlabels))
    width = 0.38
    plt.bar(xs - width/2, left_vals, width, label=left_label)
    plt.bar(xs + width/2, right_vals, width, label=right_label)
    plt.xticks(xs, xlabels)
    plt.ylabel(ylabel)
    plt.title(title)
    for i, v in enumerate(left_vals):
        if not (v != v):
            plt.text(xs[i] - width/2, v, f"{v:.2f}", ha="center", va="bottom")
    for i, v in enumerate(right_vals):
        if not (v != v):
            plt.text(xs[i] + width/2, v, f"{v:.2f}", ha="center", va="bottom")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"🖼️ Saved plot → {out_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_1min", required=True, help="Root with ALL games at 1-minute windows")
    ap.add_argument("--root_3min", required=True, help="Root with ALL games at 3-minute windows")
    ap.add_argument("--train_5min", required=True, help="Root with TRAIN games at 5-minute windows")
    ap.add_argument("--eval_5min",  required=True, help="Root with EVAL games at 5-minute windows")
    ap.add_argument("--out_dir", default="experiment_window_comparison_grouped",
                    help="Output folder for metrics and plots")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weight_scale", type=float, default=0.0, help=">0 to emphasize low-possession windows")
    ap.add_argument("--save_preds", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    root_1m = Path(args.root_1min)
    root_3m = Path(args.root_3min)
    tr_5m   = Path(args.train_5min)
    ev_5m   = Path(args.eval_5min)

    # --- Eval games list (from 5m eval root) ---
    eval_5m_names = set(list_game_names(ev_5m))
    if not eval_5m_names:
        print("❌ No eval games found in --eval_5min.", file=sys.stderr)
        sys.exit(1)
    train_5m_names = set(list_game_names(tr_5m))

    # --- Load 5m train/eval directly from their roots ---
    print("📥 Loading 5-min TRAIN...")
    train_5m_df = load_games_subset_grouped(tr_5m, include=None)  # all in train root
    print("📥 Loading 5-min EVAL...")
    eval_5m_df  = load_games_subset_grouped(ev_5m, include=None)  # all in eval root

    # --- Build splits for 1m & 3m by reusing eval_5m_names ---
    # For each root: eval = intersection; train = others
    def split_for(root: Path, eval_names_ref: set) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
        all_names = set(list_game_names(root))
        eval_names  = sorted(all_names & eval_names_ref)
        train_names = sorted(all_names - set(eval_names))
        if not eval_names:
            print(f"⚠️ No overlapping eval games in {root}.")
        train_df = load_games_subset_grouped(root, include=set(train_names)) if train_names else pd.DataFrame()
        eval_df  = load_games_subset_grouped(root, include=set(eval_names)) if eval_names else pd.DataFrame()
        return train_df, eval_df, train_names, eval_names

    print("📥 Loading 1-min split from all-games root...")
    train_1m_df, eval_1m_df, train_1m_names, eval_1m_names = split_for(root_1m, eval_5m_names)

    print("📥 Loading 3-min split from all-games root...")
    train_3m_df, eval_3m_df, train_3m_names, eval_3m_names = split_for(root_3m, eval_5m_names)

    # --- Train & evaluate for each window size ---
    results = []  # accumulate rows for summary.csv
    preds_paths = {}

    def run_one(label: str, train_df: pd.DataFrame, eval_df: pd.DataFrame):
        if train_df.empty or eval_df.empty:
            print(f"⚠️ Skipping {label}: empty train or eval.")
            return
        metrics, preds = train_eval(train_df, eval_df, seed=args.seed, weight_scale=args.weight_scale)

        # Extra metrics for plots
        mae_mid   = compute_midrange_mae(preds)
        mae_60_90 = compute_mae_60_90(preds)
        flips_up, flips_down = compute_flips(preds)

        row = {
            "window": label,
            **metrics,
            "mae_mid_025_075": mae_mid,
            "mae_60_90": mae_60_90,
            "avg_over_pred_under_true": flips_up,
            "avg_under_pred_over_true": flips_down,
            "n_games_eval": preds["game"].nunique()
        }
        results.append(row)

        if args.save_preds:
            p = out_dir / "predictions" / f"{label}_preds.csv"
            preds.to_csv(p, index=False)
            preds_paths[label] = str(p)
            print(f"🧾 Saved predictions → {p}")

    run_one("1min", train_1m_df, eval_1m_df)
    run_one("3min", train_3m_df, eval_3m_df)
    run_one("5min", train_5m_df, eval_5m_df)

    if not results:
        print("❌ No window had valid data; exiting.")
        sys.exit(1)

    summary = pd.DataFrame(results).sort_values("window").reset_index(drop=True)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"📊 Summary → {summary_path}")

    # --- Plots across windows ---
    # Ensure consistent ordering 1min, 3min, 5min if present
    order = [w for w in ["1min", "3min", "5min"] if w in summary["window"].tolist()]
    s = summary.set_index("window").loc[order].reset_index()

    # 1) Overall MAE
    bar_with_values(
        order,
        [float(x) for x in s["mae"]],
        ylabel="MAE (all minutes)",
        title="MAE by Window Size (grouped features)",
        out_path=out_dir / "mae_by_window.png"
    )

    # 2) Flips (two bars per window)
    grouped_two_bar(
        order,
        [float(x) for x in s["avg_over_pred_under_true"]],
        [float(x) for x in s["avg_under_pred_over_true"]],
        left_label="pred>0.5 & true<0.5",
        right_label="pred<0.5 & true>0.5",
        ylabel="Avg flips per game",
        title="Average threshold flips per game by window size",
        out_path=out_dir / "flips_by_window.png"
    )

    # 3) MAE on 0.25 ≤ true ≤ 0.75
    bar_with_values(
        order,
        [float(x) if x == x else np.nan for x in s["mae_mid_025_075"]],
        ylabel="MAE (true ∈ [0.25, 0.75])",
        title="Mid-range MAE by Window Size",
        out_path=out_dir / "mae_midrange_by_window.png"
    )

    # 4) MAE on minutes 60–90
    bar_with_values(
        order,
        [float(x) if x == x else np.nan for x in s["mae_60_90"]],
        ylabel="MAE (minutes 60–90)",
        title="Late-window MAE (60–90) by Window Size",
        out_path=out_dir / "mae_60_90_by_window.png"
    )

    print("\nDone ✅")


if __name__ == "__main__":
    main()
