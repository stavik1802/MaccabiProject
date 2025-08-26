#!/usr/bin/env python3
"""
Per-feature evaluation with multiple rankings @70':
- Uses a trained model (window=130) and eval set.
- Autoregressive rollout from chunk 240 for 41 steps (to include minute 70).
Outputs in OUT_DIR:
  feature_mae_at_70_metrics.csv                             # MAE, NMAE(global/local), sMAPE + ranks
  feature_mae_at_70_bar_abs.png                             # bars by absolute MAE (lower=better)
  feature_nmae_at_70_bar_global.png                         # bars by NMAE with global P95–P5 scale
  feature_nmae_at_70_bar_local.png                          # bars by NMAE with local-at-70 P95–P5 scale
  feature_smape_at_70_bar.png                               # bars by sMAPE
  curves_true_pred/<feature>.png                            # mean true vs pred
  curves_mae/<feature>_mae.png                              # MAE per minute
  curves_nmae/<feature>_nmae.png                            # NMAE (global scale) per minute
  per_feature_per_minute_mae.csv                            # per-minute MAE & NMAE (global) for each feature
"""

import os, re, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model
from tcn import TCN

# ========= Paths (edit if needed) =========
MODEL_PATH = "/home/stav.karasik/MaccabiProject/scripts/experiments_window/models/player_tcn_model_w130.keras"
EVAL_ROOT  = "games_eval_new"
OUT_DIR    = "experiments_feature_mae"
os.makedirs(OUT_DIR, exist_ok=True)

# ========= Eval settings =========
INPUT_WINDOW     = 130
PREDICT_AHEAD    = 1
PREDICT_CHUNKS   = 41      # include step 40 (minute 70)
START_CHUNK      = 240     # minute 60
TARGET_MINUTE    = 70.0
MAX_PLAYERS_EVAL = 50
EPS              = 1e-9

target_cols = [
    "inst_dist_m_sum", "Speed (m/s)_sum",  "hsr_m_sum",
    "vha_count_1s_sum", "avg_jerk_1s_sum", "turns_per_sec_sum", "playerload_1s_sum",
    "walk_time_sum", "jog_time_sum", "run_time_sum", "sprint_time_sum",
    "total_sprints_sum", "sprint_attack_sum", "sprint_defense_sum",
    "dist_attack_sum", "dist_defense_sum", "time_attack_sum", "time_defense_sum",
    "attacking_third_time_sum", "middle_third_time_sum", "defending_third_time_sum"
]

def safe_name(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', s)

def rank_and_plot(df, metric, png_name, xlabel, title):
    df = df.sort_values(metric, ascending=True).reset_index(drop=True)
    df[f"rank_{metric}"] = df[metric].rank(method="dense", ascending=True).astype(int)
    plt.figure(figsize=(12, 7))
    plt.barh(df["feature"], df[metric])
    plt.gca().invert_yaxis()
    plt.xlabel(xlabel)
    plt.title(title)
    for i, val in enumerate(df[metric].values):
        r = df.loc[i, f"rank_{metric}"]
        plt.text(val, i, f"  #{r}  ({val:.3f})", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, png_name), dpi=150)
    plt.close()
    return df[["feature", metric, f"rank_{metric}"]]

def main():
    model = load_model(MODEL_PATH, custom_objects={"TCN": TCN})

    # Per-step, per-player containers
    y_pred_all = defaultdict(lambda: [[] for _ in range(PREDICT_CHUNKS)])  # feature -> [step] -> list of players' values
    y_true_all = defaultdict(lambda: [[] for _ in range(PREDICT_CHUNKS)])
    player_count = 0
    valid_chunks = []

    # For global normalization scale (P95-P5) per feature, collect true values across players/steps
    global_true_vals = {col: [] for col in target_cols}

    for root, _, files in os.walk(EVAL_ROOT):
        for file in files:
            if not file.endswith(".csv") or player_count >= MAX_PLAYERS_EVAL:
                continue
            fp = os.path.join(root, file)
            try:
                df = pd.read_csv(fp).sort_values("chunk").reset_index(drop=True)
                if not all(c in df.columns for c in target_cols):
                    continue

                idx_list = df.index[df["chunk"] == START_CHUNK].tolist()
                if not idx_list:
                    continue
                start_idx = idx_list[0]

                # allow indexing up to step 40
                if start_idx < INPUT_WINDOW or (start_idx + (PREDICT_CHUNKS - 1)) >= len(df):
                    continue

                # Per-player standardization on full sequence
                scaler = StandardScaler()
                df[target_cols] = scaler.fit_transform(df[target_cols])

                # Seed input with W history before 240
                input_seq = df[target_cols].iloc[start_idx - INPUT_WINDOW:start_idx].values
                input_buffer = input_seq.copy()

                if not valid_chunks:
                    valid_chunks = [START_CHUNK + s for s in range(PREDICT_CHUNKS)]

                for step in range(PREDICT_CHUNKS):
                    model_input = input_buffer[-INPUT_WINDOW:].reshape(1, INPUT_WINDOW, -1)
                    y_pred_norm = model.predict(model_input, verbose=0)[0]

                    # closed-loop rollout
                    next_row = input_buffer[-1].copy()
                    next_row[:len(target_cols)] = y_pred_norm
                    input_buffer = np.vstack([input_buffer, next_row])[1:]

                    # true (still normalized)
                    true_row = df[target_cols].iloc[start_idx + step].values

                    # de-standardize to original units
                    for j, col in enumerate(target_cols):
                        mean = scaler.mean_[j]
                        std  = np.sqrt(scaler.var_[j])
                        pred_val = y_pred_norm[j] * std + mean
                        true_val = true_row[j] * std + mean
                        y_pred_all[col][step].append(pred_val)
                        y_true_all[col][step].append(true_val)
                        global_true_vals[col].append(true_val)

                player_count += 1
                print(f"✅ {player_count} processed: {fp}")

            except Exception as e:
                print(f"❌ Eval error in {fp}: {e}")

    if player_count == 0:
        raise RuntimeError("No valid evaluation files found.")

    # Time axis
    minutes = [c / 4.0 for c in valid_chunks]
    idx70 = int(np.argmin([abs(m - TARGET_MINUTE) for m in minutes]))
    print(f"\nUsing minute index {idx70} ~ {minutes[idx70]:.2f} min.\n")

    # Compute global robust ranges (P95 - P5) per feature
    scales_global = {}
    for col in target_cols:
        arr = np.asarray(global_true_vals[col], dtype=float)
        if arr.size == 0:
            scales_global[col] = 1.0
            continue
        p5, p95 = np.percentile(arr, [5, 95])
        r = max(p95 - p5, EPS)
        if r < 1e-6:
            r = max(np.std(arr), EPS)
        scales_global[col] = r

    # Mean curves across players (True vs Pred), and per-minute MAE/NMAE
    curves_dir      = os.path.join(OUT_DIR, "curves_true_pred")
    mae_curves_dir  = os.path.join(OUT_DIR, "curves_mae")
    nmae_curves_dir = os.path.join(OUT_DIR, "curves_nmae")
    os.makedirs(curves_dir, exist_ok=True)
    os.makedirs(mae_curves_dir, exist_ok=True)
    os.makedirs(nmae_curves_dir, exist_ok=True)

    per_minute_rows = []  # per-feature, per-minute MAE & NMAE (global)

    # Also compute mean curves across players for plotting
    avg_pred_all = {col: [np.mean(step_vals) for step_vals in steps] for col, steps in y_pred_all.items()}
    avg_true_all = {col: [np.mean(step_vals) for step_vals in steps] for col, steps in y_true_all.items()}

    # Per-feature plots
    for col in target_cols:
        safe_col = safe_name(col)

        # 1) Mean True vs Pred curves
        plt.figure(figsize=(12, 5))
        plt.plot(minutes, avg_true_all[col], marker="o", label="True (mean)")
        plt.plot(minutes, avg_pred_all[col], marker="x", label="Predicted (mean)")
        plt.title(f"Autoregressive Prediction (Mean over {player_count} Players) — {col}")
        plt.xlabel("Minute")
        plt.ylabel(col)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(curves_dir, f"{safe_col}.png"))
        plt.close()

        # 2) MAE per minute for this feature (across players)
        mae_curve = []
        for i in range(len(valid_chunks)):
            mae_i = mean_absolute_error(y_true_all[col][i], y_pred_all[col][i])
            mae_curve.append(mae_i)
            per_minute_rows.append({
                "feature": col,
                "chunk": valid_chunks[i],
                "minute": minutes[i],
                "mae": mae_i,                           # raw MAE
                "nmae_global": mae_i / max(scales_global[col], EPS)  # normalized by global robust range
            })

        plt.figure(figsize=(12, 5))
        plt.plot(minutes, mae_curve, marker="o")
        plt.title(f"MAE per Minute — {col}")
        plt.xlabel("Minute")
        plt.ylabel("MAE")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(mae_curves_dir, f"{safe_col}_mae.png"))
        plt.close()

        # 3) NMAE per minute (global)
        nmae_curve = [m / max(scales_global[col], EPS) for m in mae_curve]
        plt.figure(figsize=(12, 5))
        plt.plot(minutes, nmae_curve, marker="o")
        plt.title(f"Normalized MAE (by P95–P5, global) per Minute — {col}")
        plt.xlabel("Minute")
        plt.ylabel("NMAE")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(nmae_curves_dir, f"{safe_col}_nmae.png"))
        plt.close()

    # Save per-feature, per-minute MAE/NMAE (global)
    pd.DataFrame(per_minute_rows).to_csv(os.path.join(OUT_DIR, "per_feature_per_minute_mae.csv"), index=False)

    # ====== Metrics at 70' (four flavors) ======
    rows70 = []
    for col in target_cols:
        true70 = np.asarray(y_true_all[col][idx70], dtype=float)
        pred70 = np.asarray(y_pred_all[col][idx70], dtype=float)
        err70  = np.abs(true70 - pred70)

        # 1) Absolute MAE
        mae70  = float(np.mean(err70))

        # 2) NMAE with global scale (P95–P5 over all players/steps)
        nmae70_global = mae70 / max(scales_global[col], EPS)

        # 3) NMAE with local-at-70 scale (P95–P5 across players at minute 70)
        p5_70, p95_70 = np.percentile(true70, [5, 95])
        scale70_local = max(p95_70 - p5_70, EPS)
        nmae70_local  = mae70 / scale70_local

        # 4) sMAPE (symmetric MAPE)
        smape70 = float(np.mean(2.0 * err70 / (np.abs(true70) + np.abs(pred70) + EPS)))

        rows70.append({
            "feature": col,
            "mae_at_70": mae70,
            "nmae_at_70_global": nmae70_global,
            "nmae_at_70_local": nmae70_local,
            "smape_at_70": smape70,
            "scale_global_P95_P5": scales_global[col],
            "scale_local_P95_P5": scale70_local
        })

    df_metrics = pd.DataFrame(rows70)

    # Ranks (lower=better)
    for m in ["mae_at_70", "nmae_at_70_global", "nmae_at_70_local", "smape_at_70"]:
        df_metrics[f"rank_{m}"] = df_metrics[m].rank(method="dense", ascending=True).astype(int)

    # Save full metrics + ranks
    csv_metrics = os.path.join(OUT_DIR, "feature_mae_at_70_metrics.csv")
    df_metrics.sort_values("nmae_at_70_global", ascending=True).to_csv(csv_metrics, index=False)
    print(f"💾 Saved metrics & ranks: {csv_metrics}")

    # ====== Bar charts for each metric ======
    rank_and_plot(
        df_metrics[["feature","mae_at_70"]].copy(),
        metric="mae_at_70",
        png_name="feature_mae_at_70_bar_abs.png",
        xlabel="MAE at 70:00 (original units) — lower is better",
        title="Feature Ranking by Absolute MAE at minute 70"
    )

    rank_and_plot(
        df_metrics[["feature","nmae_at_70_global"]].copy(),
        metric="nmae_at_70_global",
        png_name="feature_nmae_at_70_bar_global.png",
        xlabel="Normalized MAE at 70:00 (global P95–P5) — lower is better",
        title="Feature Ranking by Normalized MAE (Global Range) at 70 minute"
    )

    rank_and_plot(
        df_metrics[["feature","nmae_at_70_local"]].copy(),
        metric="nmae_at_70_local",
        png_name="feature_nmae_at_70_bar_local.png",
        xlabel="Normalized MAE at 70:00 (local P95–P5 at minute 70) — lower is better",
        title="Feature Ranking by Normalized MAE "
    )

    rank_and_plot(
        df_metrics[["feature","smape_at_70"]].copy(),
        metric="smape_at_70",
        png_name="feature_smape_at_70_bar.png",
        xlabel="sMAPE at 70:00 — lower is better",
        title="Feature Ranking by sMAPE at minute 70"
    )

    gc.collect()

if __name__ == "__main__":
    main()
