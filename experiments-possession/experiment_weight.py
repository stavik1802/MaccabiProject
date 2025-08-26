#!/usr/bin/env python3
"""
Compare GROUPED CatBoost (with vs without sample weights) for possession regression.

Usage:
  python compare_grouped_weights.py \
    --train_root /path/to/train_games \
    --eval_root  /path/to/eval_games \
    --out_dir experiment_grouped_weighting \
    [--seed 42] [--weight_scale 0.8] [--save_preds]

Outputs in out_dir/:
  metrics_overall.csv
  metrics_per_game.csv
  mae_midrange.csv
  flips_per_game.csv
  mean_flips_per_game.csv
  plots/overall_mae.png
  plots/avg_flips_per_game.png
  plots/midrange_mae.png
  predictions/weighted_eval.csv (if --save_preds)
  predictions/unweighted_eval.csv (if --save_preds)
"""

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

# ----------------------- GROUPED loaders -----------------------
def build_features_grouped(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each [minute_start, minute_end]:
      - add count_Gk for k in 1..4
      - for every numeric feature f, add f__Gk = mean over players in group k
    """
    key_cols = ["minute_start","minute_end","player_id","position","group"]
    feature_cols = [c for c in players_df.columns if c not in key_cols]
    rows = []
    for (start, end), window in players_df.groupby(["minute_start","minute_end"], sort=True):
        row = {"minute_start": start, "minute_end": end}
        for g in ALL_GROUPS:
            row[f"count_{g}"] = int((window["group"] == g).sum())
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

# ----------------------- Model & training -----------------------
def get_feature_columns(df: pd.DataFrame) -> List[str]:
    drop_cols = ["minute_start","minute_end","maccabi_haifa_possession_percent","game"]
    return [c for c in df.columns if c not in drop_cols]

def make_model(seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=1200, learning_rate=0.05, depth=8,
        loss_function="RMSE", eval_metric="RMSE",
        random_seed=seed, early_stopping_rounds=100, verbose=False
    )

def make_weights(y: np.ndarray, weight_scale: float) -> np.ndarray:
    """
    Emphasize low-possession windows (<0.5).
    w = 1 + weight_scale * (0.5 - y)/0.5  if y<0.5 else 1
    Set weight_scale=0 to effectively disable weighting.
    """
    return np.where(y < 0.5, 1.0 + weight_scale * (0.5 - y) / 0.5, 1.0)

def train_and_predict(train_df: pd.DataFrame, eval_df: pd.DataFrame,
                      seed: int, use_weights: bool, weight_scale: float):
    feat_cols = sorted(get_feature_columns(train_df))
    X_tr = train_df.reindex(columns=feat_cols, fill_value=0.0)
    y_tr = train_df["maccabi_haifa_possession_percent"].values
    X_te = eval_df.reindex(columns=feat_cols, fill_value=0.0)
    y_te = eval_df["maccabi_haifa_possession_percent"].values

    model = make_model(seed)
    if use_weights and weight_scale > 0:
        w_tr = make_weights(y_tr, weight_scale)
        model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_te, y_te))
    else:
        model.fit(X_tr, y_tr, eval_set=(X_te, y_te))

    pred = model.predict(X_te)

    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    mae  = float(mean_absolute_error(y_te, pred))
    r2   = float(r2_score(y_te, pred))
    metrics_overall = {"rmse": rmse, "mae": mae, "r2": r2, "n_windows": int(len(eval_df))}

    preds_df = eval_df[["game","minute_start","minute_end"]].copy()
    preds_df["true"] = y_te
    preds_df["pred"] = pred
    return metrics_overall, preds_df

# ----------------------- Analysis & plots -----------------------
def per_game_metrics(preds: pd.DataFrame) -> pd.DataFrame:
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
                   ax: plt.Axes, x_pos: float, width: float):
    mean = values[value_col].mean()
    std  = values[value_col].std(ddof=0)
    ax.bar([x_pos], [mean], yerr=[0.0 if np.isnan(std) else std],
           width=width, capsize=6, label=label)
    ax.text(x_pos, mean, f"{mean:.3f}", ha="center", va="bottom")

# ----------------------- Main -----------------------
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--train_root", required=True, help="Folder with TRAIN game subfolders")
    p.add_argument("--eval_root",  required=True, help="Folder with EVAL game subfolders")
    p.add_argument("--out_dir",    default="experiment_grouped_weighting")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--weight_scale", type=float, default=0.8,
                   help="Strength of weighting for y<0.5 (0 disables effect)")
    p.add_argument("--save_preds", action="store_true")
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    plots = out / "plots"; plots.mkdir(parents=True, exist_ok=True)
    preds_dir = out / "predictions"; preds_dir.mkdir(parents=True, exist_ok=True)

    # Load grouped data
    print("📥 Loading grouped TRAIN/EVAL …")
    train_df = load_all_games_grouped(args.train_root)
    eval_df  = load_all_games_grouped(args.eval_root)
    if train_df.empty or eval_df.empty:
        print("❌ One of the datasets is empty. Check folders.")
        return

    # Train & predict: unweighted vs weighted
    print("🚂 Training UNWEIGHTED …")
    m_unw, pred_unw = train_and_predict(train_df, eval_df, args.seed, use_weights=False, weight_scale=0.0)
    pred_unw["model"] = "unweighted"

    print("🚂 Training WEIGHTED …")
    m_w, pred_w = train_and_predict(train_df, eval_df, args.seed, use_weights=True, weight_scale=args.weight_scale)
    pred_w["model"] = "weighted"

    if args.save_preds:
        pred_unw.to_csv(preds_dir/"unweighted_eval.csv", index=False)
        pred_w.to_csv(preds_dir/"weighted_eval.csv", index=False)

    # Tables
    overall = pd.DataFrame([
        {"model":"unweighted", **m_unw},
        {"model":"weighted",   **m_w}
    ])
    overall.to_csv(out/"metrics_overall.csv", index=False)

    pg_unw = per_game_metrics(pred_unw); pg_unw["model"] = "unweighted"
    pg_w   = per_game_metrics(pred_w);   pg_w["model"]   = "weighted"
    per_game = pd.concat([pg_unw, pg_w], ignore_index=True)
    per_game.to_csv(out/"metrics_per_game.csv", index=False)

    mr = pd.DataFrame([
        {"model":"unweighted", "mae_midrange": midrange_mae(pred_unw)},
        {"model":"weighted",   "mae_midrange": midrange_mae(pred_w)},
    ])
    mr.to_csv(out/"mae_midrange.csv", index=False)

    flips_unw = flips_per_game(pred_unw); flips_unw["model"] = "unweighted"
    flips_w   = flips_per_game(pred_w);   flips_w["model"]   = "weighted"
    flips_all = pd.concat([flips_unw, flips_w], ignore_index=True)
    flips_all.to_csv(out/"flips_per_game.csv", index=False)

    mean_flips = (flips_all
                  .groupby("model")[["over_pred_under_true","under_pred_over_true"]]
                  .agg(["mean","std"])
                  .reset_index())
    mean_flips.columns = ["model","over_mean","over_std","under_mean","under_std"]
    mean_flips.to_csv(out/"mean_flips_per_game.csv", index=False)

    # ---------- Plots (the 3 you asked) ----------
    # 1) Overall MAE (mean ± std across games)
    plt.figure(figsize=(6,4))
    ax = plt.gca()
    for i, model in enumerate(["unweighted","weighted"]):
        dist = per_game[per_game["model"]==model]
        bar_with_error(dist, "mae", model, ax=ax, x_pos=i, width=0.6)
    ax.set_xticks([0,1]); ax.set_xticklabels(["unweighted","weighted"])
    ax.set_ylabel("MAE (mean ± std across games)")
    ax.set_title("Overall MAE on eval set (grouped features)")
    plt.tight_layout(); plt.savefig(plots/"overall_mae.png", dpi=150); plt.close()

    # 2) Average flips per game (two bars per model)
    plot_df = mean_flips.set_index("model").loc[["unweighted","weighted"]].reset_index()
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
    plt.title("Average threshold flips per game (grouped features)")
    plt.legend()
    plt.tight_layout(); plt.savefig(plots/"avg_flips_per_game.png", dpi=150); plt.close()

    # 3) Mid-range MAE (0.25–0.75)
    plt.figure(figsize=(6,4))
    plt.bar(mr["model"], mr["mae_midrange"])
    for x_i, y in enumerate(mr["mae_midrange"]):
        plt.text(x_i, y, f"{y:.3f}", ha="center", va="bottom")
    plt.ylabel("MAE (0.25 ≤ true ≤ 0.75)")
    plt.title("Mid-range MAE on eval set (grouped features)")
    plt.tight_layout(); plt.savefig(plots/"midrange_mae.png", dpi=150); plt.close()

    print("✅ Done. See:", out.resolve())

if __name__ == "__main__":
    main()
