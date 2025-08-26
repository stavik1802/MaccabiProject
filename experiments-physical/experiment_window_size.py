#!/usr/bin/env python3
"""
Experiment: MAE at the 70th minute vs input_window
- Windows: 80..150 step 5
- Eval horizon set to 41 steps so minute 70.00 is included
- Returns one scalar per window: MAE at ~70th minute (closest chunk)
- Saves CSV + plot under experiments/
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

# ----------------- Paths -----------------
TRAIN_ROOT = "/home/stav.karasik/MaccabiProject/scripts/games_train_new"
EVAL_ROOT = "/home/stav.karasik/MaccabiProject/scripts/games_eval_new"  
OUT_DIR = "experiments_window"; os.makedirs(OUT_DIR, exist_ok=True)
MODEL_DIR = os.path.join(OUT_DIR, "models"); os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------- Fixed params -----------------
PREDICT_AHEAD = 1
PREDICT_CHUNKS = 41           # ensure we hit minute 70.00
START_CHUNK = 240             # minute 60
MAX_PLAYERS_EVAL = 50
TARGET_MINUTE = 70.0          # what we measure MAE at

target_cols = [
    "inst_dist_m_sum", "Speed (m/s)_sum",  "hsr_m_sum",
    "vha_count_1s_sum", "avg_jerk_1s_sum", "turns_per_sec_sum", "playerload_1s_sum",
    "walk_time_sum", "jog_time_sum", "run_time_sum", "sprint_time_sum",
    "total_sprints_sum", "sprint_attack_sum", "sprint_defense_sum",
    "dist_attack_sum", "dist_defense_sum", "time_attack_sum", "time_defense_sum",
    "attacking_third_time_sum", "middle_third_time_sum", "defending_third_time_sum"
]

# ----------------- Repro & TF setup -----------------
def set_seeds(seed=42):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
set_seeds(42)

# ----------------- Data utils -----------------
def create_sequences_with_scaler(df, input_window, predict_ahead, columns):
    df = df.sort_values("chunk")
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    F = df[columns].to_numpy()
    X, y = [], []
    for i in range(input_window, len(df) - predict_ahead):
        X.append(F[i - input_window:i, :])
        y.append(F[i + predict_ahead, :])
    if len(X) == 0:
        return None, None
    return np.asarray(X, np.float32), np.asarray(y, np.float32)

def load_training_data(train_root, input_window, predict_ahead, columns):
    X_train, y_train = [], []
    for game_folder in os.listdir(train_root):
        game_path = os.path.join(train_root, game_folder)
        if not os.path.isdir(game_path): continue
        for file in os.listdir(game_path):
            if not file.endswith(".csv"): continue
            fp = os.path.join(game_path, file)
            try:
                df = pd.read_csv(fp)
                if not all(c in df.columns for c in columns): continue
                if len(df) < input_window + predict_ahead + 1: continue
                X, y = create_sequences_with_scaler(df, input_window, predict_ahead, columns)
                if X is not None:
                    X_train.append(X); y_train.append(y)
            except Exception as e:
                print(f"⚠️ Skipping {fp}: {e}")
    if not X_train:
        raise RuntimeError("No training sequences found.")
    X = np.concatenate(X_train); Y = np.concatenate(y_train)
    print(f"✅ window={input_window} | training sequences: {len(X)}  (features={X.shape[2]})")
    return X, Y

# ----------------- Model -----------------
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

def train_one_window(input_window, epochs=40, batch_size=32, patience=5):
    X_train, Y_train = load_training_data(TRAIN_ROOT, input_window, PREDICT_AHEAD, target_cols)
    model = build_tcn_model(input_window, X_train.shape[2], Y_train.shape[1])
    es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    model.fit(X_train, Y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size,
              callbacks=[es], verbose=2)
    path = os.path.join(MODEL_DIR, f"player_tcn_model_w{input_window}.keras")
    model.save(path)
    print(f"💾 Saved model: {path}")
    del X_train, Y_train
    return path

# ----------------- Evaluation (MAE at ~70:00) -----------------
def evaluate_window_mae_at_70(model_path, input_window) -> float:
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
                if not all(c in df.columns for c in target_cols): continue

                idx_list = df.index[df["chunk"] == START_CHUNK].tolist()
                if not idx_list: continue
                start_idx = idx_list[0]

                # precise boundary so we can index up to start_idx + (PREDICT_CHUNKS-1)
                if start_idx < input_window or (start_idx + (PREDICT_CHUNKS - 1)) >= len(df):
                    continue

                scaler = StandardScaler()
                df[target_cols] = scaler.fit_transform(df[target_cols])

                input_seq = df[target_cols].iloc[start_idx - input_window:start_idx].values
                input_buffer = input_seq.copy()

                if not valid_chunks:
                    valid_chunks = [START_CHUNK + step for step in range(PREDICT_CHUNKS)]

                for step in range(PREDICT_CHUNKS):
                    model_input = input_buffer[-input_window:].reshape(1, input_window, -1)
                    y_pred_norm = model.predict(model_input, verbose=0)[0]

                    # roll forward
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

    # Aggregate across players
    avg_pred_all = {col: [np.mean(vals) for vals in steps] for col, steps in y_pred_all.items()}
    avg_true_all = {col: [np.mean(vals) for vals in steps] for col, steps in y_true_all.items()}

    # Find index closest to 70:00
    minutes = [c / 4.0 for c in valid_chunks]
    idx70 = int(np.argmin([abs(m - TARGET_MINUTE) for m in minutes]))

    # MAE at that minute across all features
    true_vec = [avg_true_all[col][idx70] for col in target_cols]
    pred_vec = [avg_pred_all[col][idx70] for col in target_cols]
    mae_70 = float(mean_absolute_error(true_vec, pred_vec))
    print(f"🏁 window={input_window} | MAE@{minutes[idx70]:.2f} min ≈ {mae_70:.4f}  (players={player_count})")
    return mae_70

# ----------------- Main -----------------
def main():
    results = []
    windows = list(range(80, 151, 5))
    EPOCHS, BATCH, PATIENCE = 40, 32, 5

    for W in windows:
        set_seeds(42)
        tf.keras.backend.clear_session()

        model_path = os.path.join(MODEL_DIR, f"player_tcn_model_w{W}.keras")
        if not os.path.isfile(model_path):
            model_path = train_one_window(W, epochs=EPOCHS, batch_size=BATCH, patience=PATIENCE)

        mae_70 = evaluate_window_mae_at_70(model_path, W)
        results.append({"window": W, "mae_at_70": mae_70})

        tf.keras.backend.clear_session()
        gc.collect()

    df = pd.DataFrame(results).sort_values("window")
    csv_path = os.path.join(OUT_DIR, "window_mae_at_70.csv")
    df.to_csv(csv_path, index=False)
    print(f"💾 Saved: {csv_path}")

    plt.figure(figsize=(10,5))
    plt.plot(df["window"], df["mae_at_70"], marker="o")
    plt.xlabel("Input window size (chunks)")
    plt.ylabel("MAE at 70:00")
    plt.title("MAE at 70th Minute vs. Input Window Size")
    plt.grid(True)
    png_path = os.path.join(OUT_DIR, "window_mae_at_70.png")
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
    print(f"📈 Saved: {png_path}")

if __name__ == "__main__":
    main()
