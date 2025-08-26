#!/usr/bin/env python3
"""
Compare CatBoost (grouped vs flat) on possession regression.

Usage:
  python compare_grouped_vs_flat.py \
    --train_root /path/to/train_games \
    --eval_root /path/to/eval_games \
    --out_dir experiment_grouped_vs_flat \
    [--seed 42] [--save_preds]

Outputs under out_dir/:
  - metrics_overall.csv                 # overall MAE/RMSE/R2 per model
  - metrics_per_game.csv                # per-game metrics per model
  - mae_midrange.csv                    # MAE on 0.25 ≤ true ≤ 0.75 per model
  - flips_per_game.csv                  # per-game flip counts per model
  - mean_flips_per_game.csv             # mean±std flips per game per model
  - plots:
      overall_mae.png                   # bar (mean±std) MAE per model
      midrange_mae.png                  # bar MAE on 0.25–0.75 per model
      avg_flips_per_game.png            # grouped bars of flips per model
  - predictions/model_flat_eval.csv     # per-interval predictions (flat)
  - predictions/model_grouped_eval.csv  # per-interval predictions (grouped)
"""

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor

# ----------------------- Group mapping (UNK -> G3) -----------------------
POS2GROUP = {
    "CB": "G1",
    "CM": "G2", "DM": "G2",
    "RB": "G3", "RW": "G3", "RM": "G3", "LB": "G3", "LW": "G3", "LM": "G3",
    "AM": "G4", "CF": "G4",
    "UNK": "G3",
}
ALL_GROUPS = ["G1", "G2", "G3", "G4"]

def parse_pos_from_stem(stem: str) -> str:
    # "merged_features_AM_21" -> "AM"
    try:
        return stem.replace("merged_features_", "").split("_")[0]
    except Exception:
        return "UNK"

# ----------------------- Loaders: FLAT -----------------------
def build_features_flat(players_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in players_df.columns
                    if c not in ["minute_start","minute_end","player_id"]]
    rows = []
    for (start, end), window in players_df.groupby(["minute_start","minute_end"]):
        window = window.sort_values(["minute_start","minute_end"])
        slots, used = [], set()
        for idx, row in window.iterrows():
            if idx in used:
                continue
            acc = row[feature_cols].copy()
            end_t = row["minute_end"]; used.add(idx)
            merge_flag = True
            while merge_flag:
                merge_flag = False
                for j, r2 in window.iterrows():
                    if j in used: continue
                    if r2["minute_start"] == end_t:
                        acc += r2[feature_cols]
                        end_t = r2["minute_end"]
                        used.add(j)
                        merge_flag = True
                        break
            slots.append(acc)
        if slots:
            avg = sum(slots) / len(slots)
            d = {"minute_start": start, "minute_end": end}
            d.update({c: avg[c] for c in feature_cols})
            rows.append(d)
    return pd.DataFrame(rows)

def load_all_games_flat(root_folder: str) -> pd.DataFrame:
    root = Path(root_folder)
    dfs = []
    for game in root.iterdir():
        if not game.is_dir(): continue
        parts = []
        for f in game.glob("merged_features_*.csv"):
            df = pd.read_csv(f)
            df["player_id"] = f.stem.replace("merged_features_","")
            parts.append(df)
        if not parts: continue
        feats = build_features_flat(pd.concat(parts, ignore_index=True))
        poss = pd.read_csv(game/"poss.csv")
        poss["minute_start"] = poss["time_start_sec"] // 60
        poss["minute_end"]   = poss["time_end_sec"]   // 60
        merged = feats.merge(
            poss[["minute_start","minute_end","maccabi_haifa_possession_percent"]],
            on=["minute_start","minute_end"], how="inner"
        )
        merged["game"] = game.name
        dfs.append(merged)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ----------------------- Loaders: GROUPED -----------------------
def build_features_grouped(players_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["minute_start","minute_end","player_id","position","group"]
    feature_cols = [c for c in players_df.columns if c not in key_cols]
    rows = []
    for (start, end), window in players_df.groupby(["minute_start","minute_end"], sort=True):
        row = {"minute_start": start, "minute_end": end}
        # counts
        for g in ALL_GROUPS:
            row[f"count_{g}"] = int((window["group"] == g).sum())
        # per-group MEANS
        for g in ALL_GROUPS:
            sub = window[window["group"] == g]
            means = sub[feature_cols].mean(numeric_only=True) if len(sub) else pd.Series(dtype=float)
            for c in feature_cols:
                row[f"{c}__{g}"] = float(means.get(c, 0.0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["minute_start","minute_end"]).reset_index(drop=True)

def load_all_games_grouped(root_folder: str) -> pd.DataFrame:
    root = Path(root_folder)
    dfs = []
    for game in root.iterdir():
        if not game.is_dir(): continue
        parts = []
        for f in game.glob("merged_features_*.csv"):
            df = pd.read_csv(f)
            stem = f.stem
            pos  = parse_pos_from_stem(stem) or "UNK"
            pos  = pos if pos in POS2GROUP else "UNK"
            grp  = POS2GROUP[pos]  # UNK -> G3
            df["player_id"] = stem.replace("merged_features_","")
            df["position"]  = pos
            df["group"]     = grp
            parts.append(df)
        if not parts: continue
        feats = build_features_grouped(pd.concat(parts, ignore_index=True))
        poss = pd.read_csv(game/"poss.csv")
        poss["minute_start"] = poss["time_start_sec"] // 60
        poss["minute_end"]   = poss["time_end_sec"]   // 60
        merged = feats.merge(
            poss[["minute_start","minute_end","maccabi_haifa_possession_percent"]],
            on=["minute_start","minute_end"], how="inner"
        )
        merged["game"] = game.name
        dfs.append(merged)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# ----------------------- Model helpers -----------------------
def get_feature_columns(df: pd.DataFrame) -> List[str]:
    drop_cols = ["minute_start","minute_end","maccabi_haifa_possession_percent","game"]
    return [c for c in df.columns if c not in drop_cols]

def make_model(seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=1200, learning_rate=0.05, depth=8,
        loss_function="RMSE", eval_metric="RMSE",
        random_seed=seed, early_stopping_rounds=100, verbose=False
    )

def train_and_predict(train_df: pd.DataFrame, eval_df: pd.DataFrame, seed: int):
    feat_cols = sorted(get_feature_columns(train_df))
    X_tr = train_df.reindex(columns=feat_cols, fill_value=0.0)
    y_tr = train_df["maccabi_haifa_possession_percent"].values
    X_te = eval_df.reindex(columns=feat_cols, fill_value=0.0)
    y_te = eval_df["maccabi_haifa_possession_percent"].values

    model = make_model(seed)
    model.fit(X_tr, y_tr, eval_set=(X_te, y_te))
    pred = model.predict(X_te)

    # overall metrics
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    mae  = float(mean_absolute_error(y_te, pred))
    r2   = float(r2_score(y_te, pred))
    metrics_overall = {"rmse": rmse, "mae": mae, "r2": r2, "n_windows": int(len(eval_df))}

    # per-interval prediction frame
    preds_df = eval_df[["game","minute_start","minute_end"]].copy()
    preds_df["true"] = y_te
    preds_df["pred"] = pred
    return metrics_overall, preds_df

# ----------------------- Analysis/plots -----------------------
def metrics_per_game(preds: pd.DataFrame) -> pd.DataFrame:
    # per game MAE/RMSE/R2
    rows = []
    for g, d in preds.groupby("game"):
        y = d["true"].values; p = d["pred"].values
        rows.append({
            "game": g,
            "mae": float(mean_absolute_error(y, p)),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": float(r2_score(y, p)) if len(np.unique(y)) > 1 else np.nan
        })
    return pd.DataFrame(rows)

def midrange_mae(preds: pd.DataFrame) -> float:
    d = preds[(preds["true"] >= 0.25) & (preds["true"] <= 0.75)]
    return float(mean_absolute_error(d["true"], d["pred"])) if not d.empty else np.nan

def flips_per_game(preds: pd.DataFrame) -> pd.DataFrame:
    df = preds.copy()
    df["over_pred_under_true"] = ((df["pred"] > 0.5) & (df["true"] < 0.5)).astype(int)
    df["under_pred_over_true"] = ((df["pred"] < 0.5) & (df["true"] > 0.5)).astype(int)
    return (df.groupby("game")
              .agg(over_pred_under_true=("over_pred_under_true","sum"),
                   under_pred_over_true=("under_pred_over_true","sum"))
              .reset_index())

def bar_with_error(values: pd.DataFrame, value_col: str, label: str,
                   ax: plt.Axes, x_pos: float, width: float, color=None):
    mean = values[value_col].mean()
    std  = values[value_col].std(ddof=0)
    ax.bar([x_pos], [mean], yerr=[0.0 if np.isnan(std) else std], width=width,
           label=label, capsize=6, color=color)
    ax.text(x_pos, mean, f"{mean:.3f}", ha="center", va="bottom")

# ----------------------- Main -----------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", required=True, help="Folder with train game subfolders")
    ap.add_argument("--eval_root",  required=True, help="Folder with eval game subfolders")
    ap.add_argument("--out_dir",    default="experiment_grouped_vs_flat")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--save_preds", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    preds_dir = out / "predictions"; preds_dir.mkdir(parents=True, exist_ok=True)

    # --------- Load data (flat & grouped) ---------
    print("📥 Loading FLAT train/eval …")
    train_flat = load_all_games_flat(args.train_root)
    eval_flat  = load_all_games_flat(args.eval_root)

    print("📥 Loading GROUPED train/eval …")
    train_grp = load_all_games_grouped(args.train_root)
    eval_grp  = load_all_games_grouped(args.eval_root)

    if train_flat.empty or eval_flat.empty or train_grp.empty or eval_grp.empty:
        print("❌ One of the datasets is empty. Check folders."); return

    # --------- Train & predict ---------
    print("🚂 Training FLAT …")
    m_flat, pred_flat = train_and_predict(train_flat, eval_flat, args.seed)
    pred_flat["model"] = "flat"

    print("🚂 Training GROUPED …")
    m_grp, pred_grp = train_and_predict(train_grp, eval_grp, args.seed)
    pred_grp["model"] = "grouped"

    # Save predictions
    if args.save_preds:
        pred_flat.to_csv(preds_dir/"model_flat_eval.csv", index=False)
        pred_grp.to_csv(preds_dir/"model_grouped_eval.csv", index=False)

    # --------- Metrics tables ---------
    overall = pd.DataFrame([
        {"model": "flat", **m_flat},
        {"model": "grouped", **m_grp},
    ])
    overall.to_csv(out/"metrics_overall.csv", index=False)

    # Per-game metrics
    pg_flat = metrics_per_game(pred_flat); pg_flat["model"] = "flat"
    pg_grp  = metrics_per_game(pred_grp);  pg_grp["model"]  = "grouped"
    per_game = pd.concat([pg_flat, pg_grp], ignore_index=True)
    per_game.to_csv(out/"metrics_per_game.csv", index=False)

    # Mid-range MAE (0.25–0.75)
    mr = pd.DataFrame([
        {"model":"flat",    "mae_midrange": midrange_mae(pred_flat)},
        {"model":"grouped", "mae_midrange": midrange_mae(pred_grp)},
    ])
    mr.to_csv(out/"mae_midrange.csv", index=False)

    # Flips per game and averages
    flips_f = flips_per_game(pred_flat); flips_f["model"] = "flat"
    flips_g = flips_per_game(pred_grp);  flips_g["model"]  = "grouped"
    flips_all = pd.concat([flips_f, flips_g], ignore_index=True)
    flips_all.to_csv(out/"flips_per_game.csv", index=False)

    mean_flips = (flips_all
                  .groupby("model")[["over_pred_under_true","under_pred_over_true"]]
                  .agg(["mean","std"])
                  .reset_index())
    mean_flips.columns = ["model",
                          "over_mean","over_std",
                          "under_mean","under_std"]
    mean_flips.to_csv(out/"mean_flips_per_game.csv", index=False)

    # --------- Plots ---------
    # 1) Overall MAE bar (mean±std across games)
    plt.figure(figsize=(6,4))
    ax = plt.gca()
    # compute per-game MAE distributions for error bars
    for i, model in enumerate(["flat","grouped"]):
        dist = per_game[per_game["model"]==model]
        bar_with_error(dist, "mae", model, ax=ax, x_pos=i, width=0.6)
    ax.set_xticks([0,1]); ax.set_xticklabels(["flat","grouped"])
    ax.set_ylabel("MAE (mean ± std across games)")
    ax.set_title("Overall MAE on eval set")
    plt.tight_layout(); plt.savefig(out/"overall_mae.png", dpi=150); plt.close()

    # 2) Mid-range MAE bar (no error bars—single number per model)
    plt.figure(figsize=(6,4))
    plt.bar(mr["model"], mr["mae_midrange"])
    for x, y in enumerate(mr["mae_midrange"]):
        plt.text(x, y, f"{y:.3f}", ha="center", va="bottom")
    plt.ylabel("MAE (0.25 ≤ true ≤ 0.75)")
    plt.title("Mid-range MAE on eval set")
    plt.tight_layout(); plt.savefig(out/"midrange_mae.png", dpi=150); plt.close()

    # 3) Average flips per game (grouped bars with std)
    plot_df = mean_flips.set_index("model").loc[["flat","grouped"]].reset_index()
    x = np.arange(len(plot_df)); width = 0.35
    plt.figure(figsize=(7,4))
    plt.bar(x - width/2, plot_df["over_mean"], width,
            yerr=plot_df["over_std"].fillna(0.0), capsize=6, label="pred>0.5 & true<0.5")
    plt.bar(x + width/2, plot_df["under_mean"], width,
            yerr=plot_df["under_std"].fillna(0.0), capsize=6, label="pred<0.5 & true>0.5")
    plt.xticks(x, plot_df["model"])
    for i in range(len(x)):
        plt.text(x[i]-width/2, plot_df["over_mean"][i], f"{plot_df['over_mean'][i]:.2f}", ha="center", va="bottom")
        plt.text(x[i]+width/2, plot_df["under_mean"][i], f"{plot_df['under_mean'][i]:.2f}", ha="center", va="bottom")
    plt.ylabel("Avg flips per game")
    plt.title("Average threshold flips per game (eval set)")
    plt.legend()
    plt.tight_layout(); plt.savefig(out/"avg_flips_per_game.png", dpi=150); plt.close()

    print("✅ Done. See:", out.resolve())

if __name__ == "__main__":
    main()
