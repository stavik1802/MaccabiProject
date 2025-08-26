#!/usr/bin/env python3
"""
Evaluate an autoregressive TCN (trained on per-player chunk CSVs with per-file StandardScaler).

- Iterates CSVs under EVAL_ROOT (recursively), selects up to MAX_FILES that have
  all target_cols and enough length (>= input_window + N_STEPS).
- For each file: scales columns with StandardScaler (fit on that file), runs
  autoregressive prediction for N_STEPS starting right after the first input_window,
  inverse-transforms both y_pred and y_true to original units, and logs results.
- Aggregates MAE/RMSE/R² per feature and overall.

Outputs:
  eval_out/
    ├─ autoreg_master.csv
    ├─ summary_feature_metrics.csv
    ├─ summary_overall.csv
    ├─ eval_files.txt
    └─ plots/feature_<sanitized>.png
"""

import os, re, random, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
from tcn import TCN

# ===================== Params (match your train) =====================
INPUT_WINDOW = 130
N_STEPS      = 40
RANDOM_SEED  = 42
MAX_FILES    = 50
MIN_LEN      = INPUT_WINDOW + N_STEPS
MODEL_PATH   = "saved_model/player_tcn_model.keras"
EVAL_ROOT    = "/home/stav.karasik/MaccabiProject/scripts/games_eval_new"
OUT_DIR      = Path("eval_out")

TARGET_COLS = [
    "inst_dist_m_sum", "Speed (m/s)_sum",  "hsr_m_sum",
    "vha_count_1s_sum", "avg_jerk_1s_sum", "turns_per_sec_sum", "playerload_1s_sum",
    "walk_time_sum", "jog_time_sum", "run_time_sum", "sprint_time_sum",
    "total_sprints_sum", "sprint_attack_sum", "sprint_defense_sum",
    "dist_attack_sum", "dist_defense_sum", "time_attack_sum", "time_defense_sum",
    "attacking_third_time_sum", "middle_third_time_sum", "defending_third_time_sum"
]

# ===================== Helpers =====================
def find_csvs(root: str) -> list[str]:
    rootp = Path(root)
    return [str(p) for p in rootp.rglob("*.csv")]

def valid_for_eval(fp: str) -> bool:
    try:
        df = pd.read_csv(fp)
        if not all(c in df.columns for c in TARGET_COLS): return False
        if len(df) < MIN_LEN: return False
        return True
    except Exception:
        return False

def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)

# ===================== Load model =====================
print(f"📥 Loading model: {MODEL_PATH}")
model = load_model(MODEL_PATH, custom_objects={"TCN": TCN})

# ===================== Pick a fixed eval roster =====================
all_csvs = find_csvs(EVAL_ROOT)
candidates = [fp for fp in all_csvs if valid_for_eval(fp)]
if not candidates:
    raise RuntimeError("No valid evaluation CSVs found (missing columns or too short).")

candidates = sorted(candidates)  # determinism
if len(candidates) > MAX_FILES:
    rng = random.Random(RANDOM_SEED)
    candidates = rng.sample(candidates, MAX_FILES)

OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_DIR / "eval_files.txt", "w") as f:
    for fp in candidates:
        f.write(fp + "\n")
print(f"🗂  Using {len(candidates)} eval files. List saved → {OUT_DIR/'eval_files.txt'}")

# ===================== Evaluate files =====================
rows = []  # master rows
y_true_all = defaultdict(list)  # feature -> list of true values (all files/steps)
y_pred_all = defaultdict(list)  # feature -> list of pred values (all files/steps)

for idx, fp in enumerate(candidates, 1):
    try:
        df = pd.read_csv(fp).sort_values("chunk").reset_index(drop=True)
        # fit scaler per file on the target columns (like training)
        scaler = StandardScaler()
        df[TARGET_COLS] = scaler.fit_transform(df[TARGET_COLS])

        data = df[TARGET_COLS].to_numpy(dtype=np.float32)
        window = data[:INPUT_WINDOW].copy()

        # cache for inverse transform
        means = scaler.mean_
        stds  = np.sqrt(scaler.var_)  # vector

        for step in range(N_STEPS):
            model_input = window[-INPUT_WINDOW:].reshape(1, INPUT_WINDOW, -1)
            y_pred_norm = model.predict(model_input, verbose=0)[0]  # shape=(F,)

            # advance the rolling buffer
            next_row = window[-1].copy()
            next_row[:len(TARGET_COLS)] = y_pred_norm
            window = np.vstack([window, next_row])[1:]

            # guard true index
            true_idx = INPUT_WINDOW + step
            if true_idx >= len(data):
                break

            true_norm = data[true_idx]  # normalized vector

            # inverse transform to original units
            pred_real = y_pred_norm * stds + means
            true_real = true_norm    * stds + means

            for j, col in enumerate(TARGET_COLS):
                val_t = float(true_real[j])
                val_p = float(pred_real[j])
                rows.append({
                    "file": fp,
                    "step": step,                 # 0..N_STEPS-1 (15s per step)
                    "feature": col,
                    "true_value": val_t,
                    "predicted_value": val_p
                })
                y_true_all[col].append(val_t)
                y_pred_all[col].append(val_p)

        print(f"✅ [{idx}/{len(candidates)}] {fp}")
    except Exception as e:
        print(f"❌ [{idx}/{len(candidates)}] {fp} -> {e}")

# ===================== Save master CSV =====================
master = pd.DataFrame(rows)
if master.empty:
    raise RuntimeError("Evaluation produced no rows. Check inputs and target columns.")

master_path = OUT_DIR / "autoreg_master.csv"
master.to_csv(master_path, index=False)
print(f"💾 Master CSV → {master_path}")

# ===================== Metrics: per-feature and overall =====================
summary_rows = []
flat_true, flat_pred = [], []

for col in TARGET_COLS:
    yt = np.array(y_true_all[col], dtype=float)
    yp = np.array(y_pred_all[col], dtype=float)
    if yt.size == 0:
        continue
    mae = mean_absolute_error(yt, yp)
    rmse = math.sqrt(mean_squared_error(yt, yp))
    # r2 can be nan if variance=0; handle safely
    try:
        r2 = r2_score(yt, yp)
    except Exception:
        r2 = float("nan")
    summary_rows.append({"feature": col, "MAE": mae, "RMSE": rmse, "R2": r2})

    flat_true.append(yt)
    flat_pred.append(yp)

summary_df = pd.DataFrame(summary_rows).sort_values("feature")
summary_path = OUT_DIR / "summary_feature_metrics.csv"
summary_df.to_csv(summary_path, index=False)
print(f"📊 Per-feature metrics → {summary_path}")

# overall
yt_all = np.concatenate(flat_true) if flat_true else np.array([])
yp_all = np.concatenate(flat_pred) if flat_pred else np.array([])
overall = {
    "MAE": mean_absolute_error(yt_all, yp_all) if yt_all.size else float("nan"),
    "RMSE": math.sqrt(mean_squared_error(yt_all, yp_all)) if yt_all.size else float("nan"),
    "R2": r2_score(yt_all, yp_all) if yt_all.size else float("nan"),
    "files": len(candidates),
    "steps_per_file": N_STEPS,
}
overall_df = pd.DataFrame([overall])
overall_path = OUT_DIR / "summary_overall.csv"
overall_df.to_csv(overall_path, index=False)
print(f"📈 Overall metrics → {overall_path}")

# ===================== Optional: aggregated plots =====================
plots_dir = OUT_DIR / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# 1) Pred vs True curves aggregated by mean across files at each step, per feature
for col in TARGET_COLS:
    feat_df = master[master["feature"] == col]
    if feat_df.empty: 
        continue
    agg = feat_df.groupby("step")[["true_value","predicted_value"]].mean().reset_index()
    plt.figure(figsize=(10,4))
    plt.plot(agg["step"], agg["true_value"], marker="o", label="True (mean)")
    plt.plot(agg["step"], agg["predicted_value"], marker="x", linestyle="--", label="Pred (mean)")
    plt.title(f"Autoregressive Forecast – {col}")
    plt.xlabel("Step (15s)")
    plt.ylabel(col)
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(plots_dir / f"feature_{sanitize(col)}.png")
    plt.close()

print(f"🖼  Plots → {plots_dir}")
print("✅ Done.")
