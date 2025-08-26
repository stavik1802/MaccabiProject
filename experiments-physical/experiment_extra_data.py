#!/usr/bin/env python3
"""
Compare MAE@70 for window=130 WITHOUT retraining baseline:
- Baseline: read MAE@70 from experiments_window/window_mae_at_70.csv (window=INPUT_WINDOW)
- Augmented: train on originals + perturbed, evaluate MAE@70
Saves:
  experiments_more_data/mae_at_70_comparison.csv
  experiments_more_data/mae_at_70_bar.png
  models under experiments_more_data/models/
"""

import os, re, gc, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tcn import TCN

# ======= Paths (edit if needed) ============================================
TRAIN_ROOT_AUGMENTED = "games_train_mixed"  # originals + *_perturbed.csv side-by-side
BASELINE_CSV         = "experiments_window/window_mae_at_70.csv"  # has columns: window, mae_at_70
EVAL_ROOT            = "/home/stav.karasik/MaccabiProject/scripts/games_eval_new"

OUT_DIR   = "experiments_more_data"; os.makedirs(OUT_DIR, exist_ok=True)
MODEL_DIR = os.path.join(OUT_DIR, "models"); os.makedirs(MODEL_DIR, exist_ok=True)

# ======= Fixed experiment settings =========================================
INPUT_WINDOW     = 130   # <-- window to compare; must exist in BASELINE_CSV
PREDICT_AHEAD    = 1
PREDICT_CHUNKS   = 41    # include step 40 (= minute 70)
START_CHUNK      = 240   # minute 60
TARGET_MINUTE    = 70.0
MAX_PLAYERS_EVAL = 50

target_cols = [
    "inst_dist_m_sum", "Speed (m/s)_sum",  "hsr_m_sum",
    "vha_count_1s_sum", "avg_jerk_1s_sum", "turns_per_sec_sum", "playerload_1s_sum",
    "walk_time_sum", "jog_time_sum", "run_time_sum", "sprint_time_sum",
    "total_sprints_sum", "sprint_attack_sum", "sprint_defense_sum",
    "dist_attack_sum", "dist_defense_sum", "time_attack_sum", "time_defense_sum",
    "attacking_third_time_sum", "middle_third_time_sum", "defending_third_time_sum"
]

# ======= Repro & TF setup ===================================================
def set_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
set_seeds(42)

# ======= Data loading =======================================================
def create_sequences_with_scaler(df, input_window, predict_ahead, columns):
    df = df.sort_values("chunk")
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    F = df[columns].to_numpy()
    X, y = [], []
    for i in range(input_window, len(df) - predict_ahead):
        X.append(F[i - input_window:i, :])
        y.append(F[i + predict_ahead, :])
    if not X:
        return None, None
    return np.asarray(X, np.float32), np.asarray(y, np.float32)

def load_training_data(train_root, input_window, predict_ahead, columns):
    X_train, y_train = [], []
    for game_folder in os.listdir(train_root):
        game_path = os.path.join(train_root, game_folder)
        if not os.path.isdir(game_path): 
            continue
        for file in os.listdir(game_path):
            if not file.endswith(".csv"): 
                continue
            fp = os.path.join(game_path, file)
            try:
                df = pd.read_csv(fp)
                if not all(c in df.columns for c in columns): 
                    continue
                if len(df) < input_window + predict_ahead + 1: 
                    continue
                X, y = create_sequences_with_scaler(df, input_window, predict_ahead, columns)
                if X is not None:
                    X_train.append(X); y_train.append(y)
            except Exception as e:
                print(f"⚠️ Skipping {fp}: {e}")
    if not X_train:
        raise RuntimeError(f"No training sequences found under {train_root}.")
    X = np.concatenate(X_train); Y = np.concatenate(y_train)
    print(f"✅ augmented | sequences: {len(X)}  (features={X.shape[2]})")
    return X, Y

# ======= Model ==============================================================
def build_tcn_model(input_window, num_features, output_size):
    model = Sequential([
        TCN(input_shape=(input_window, num_features), nb_filters=64, kernel_size=3,
            dilations=[1, 2, 4], return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(output_size)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_augmented_and_save(epochs=40, batch_size=32, patience=5):
    X_train, Y_train = load_training_data(TRAIN_ROOT_AUGMENTED, INPUT_WINDOW, PREDICT_AHEAD, target_cols)
    model = build_tcn_model(INPUT_WINDOW, X_train.shape[2], Y_train.shape[1])
    es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    model.fit(X_train, Y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size,
              callbacks=[es], verbose=2)
    path = os.path.join(MODEL_DIR, f"player_tcn_augmented_w{INPUT_WINDOW}.keras")
    model.save(path)
    print(f"💾 Saved augmented model: {path}")
    # cleanup
    del X_train, Y_train
    tf.keras.backend.clear_session(); gc.collect()
    return path

# ======= Evaluation: MAE at 70:00 ==========================================
def mae_at_70(model_path) -> float:
    model = load_model(model_path, custom_objects={"TCN": TCN})
    player_count = 0
    y_pred_all = defaultdict(lambda: [[] for _ in range(PREDICT_CHUNKS)])
    y_true_all = defaultdict(lambda: [[] for _ in range(PREDICT_CHUNKS)])
    valid_chunks = []

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
                if start_idx < INPUT_WINDOW or (start_idx + (PREDICT_CHUNKS - 1)) >= len(df):
                    continue

                scaler = StandardScaler()
                df[target_cols] = scaler.fit_transform(df[target_cols])

                input_seq = df[target_cols].iloc[start_idx - INPUT_WINDOW:start_idx].values
                input_buffer = input_seq.copy()

                if not valid_chunks:
                    valid_chunks = [START_CHUNK + step for step in range(PREDICT_CHUNKS)]

                for step in range(PREDICT_CHUNKS):
                    model_input = input_buffer[-INPUT_WINDOW:].reshape(1, INPUT_WINDOW, -1)
                    y_pred_norm = model.predict(model_input, verbose=0)[0]

                    # closed-loop roll
                    next_row = input_buffer[-1].copy()
                    next_row[:len(target_cols)] = y_pred_norm
                    input_buffer = np.vstack([input_buffer, next_row])[1:]

                    true_row = df[target_cols].iloc[start_idx + step].values

                    # de-standardize
                    for j, col in enumerate(target_cols):
                        mean = scaler.mean_[j]; std = np.sqrt(scaler.var_[j])
                        pred_val = y_pred_norm[j] * std + mean
                        true_val = true_row[j] * std + mean
                        y_pred_all[col][step].append(pred_val)
                        y_true_all[col][step].append(true_val)

                player_count += 1
            except Exception as e:
                print(f"❌ Eval error in {fp}: {e}")

    if player_count == 0:
        raise RuntimeError("No valid evaluation files found.")

    avg_pred_all = {col: [np.mean(vals) for vals in steps] for col, steps in y_pred_all.items()}
    avg_true_all = {col: [np.mean(vals) for vals in steps] for col, steps in y_true_all.items()}

    minutes = [c / 4.0 for c in valid_chunks]
    idx70 = int(np.argmin([abs(m - TARGET_MINUTE) for m in minutes]))

    true_vec = [avg_true_all[col][idx70] for col in target_cols]
    pred_vec = [avg_pred_all[col][idx70] for col in target_cols]
    mae_70 = float(mean_absolute_error(true_vec, pred_vec))
    print(f"🏁 Augmented MAE@{minutes[idx70]:.2f} min = {mae_70:.4f}  (players={player_count})")
    tf.keras.backend.clear_session(); gc.collect()
    return mae_70

# ======= Baseline loader (from CSV) =========================================
def load_baseline_mae_from_csv(csv_path: str, window: int) -> float:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Baseline CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {"window", "mae_at_70"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required_cols}, got {list(df.columns)}")
    row = df[df["window"] == window]
    if row.empty:
        # try int-cast fallback in case column is float
        row = df[(df["window"].astype(int)) == int(window)]
    if row.empty:
        raise ValueError(f"No MAE entry found in {csv_path} for window={window}.")
    return float(row.iloc[0]["mae_at_70"])

# ======= Main ===============================================================
def main():
    set_seeds(42)

    # 1) Load baseline MAE@70 for the desired window from CSV
    mae_base = load_baseline_mae_from_csv(BASELINE_CSV, INPUT_WINDOW)
    print(f"📄 Baseline (orig only) MAE@70 from CSV (window={INPUT_WINDOW}): {mae_base:.4f}")

    # 2) Train augmented (orig + perturbed) and evaluate MAE@70
    model_path_aug = os.path.join(MODEL_DIR, f"player_tcn_augmented_w{INPUT_WINDOW}.keras")
    if not os.path.isfile(model_path_aug):
        model_path_aug = train_augmented_and_save(epochs=40, batch_size=32, patience=5)
    mae_aug = mae_at_70(model_path_aug)

    # 3) Save CSV + bar plot
    df = pd.DataFrame([
        {"setup": "Baseline (orig only)", "mae_at_70": mae_base},
        {"setup": "Augmented (+perturbed)", "mae_at_70": mae_aug},
    ])
    csv_path = os.path.join(OUT_DIR, "mae_at_70_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"💾 Saved: {csv_path}")

    plt.figure(figsize=(7,5))
    bars = plt.bar(df["setup"], df["mae_at_70"])
    plt.ylabel("MAE at 70:00")
    plt.title(f"Effect of Extra (Perturbed) Data on MAE@70 (window={INPUT_WINDOW})")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, h, f"{h:.4f}", ha="center", va="bottom")
    png_path = os.path.join(OUT_DIR, "mae_at_70_bar.png")
    plt.tight_layout(); plt.savefig(png_path); plt.close()
    print(f"📈 Saved: {png_path}")

if __name__ == "__main__":
    main()
