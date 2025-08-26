"""
Train a TCN to forecast per-player 15s physical features, then demo an
autoregressive rollout on one eval file.

• Data: per-player CSVs with a 'chunk' column and all target_cols.
• Preprocess: per-file StandardScaler on target_cols; build sliding windows
  (input_window=130) to predict the next chunk (predict_ahead=1).
• Model: Keras TCN → Dropout → Dense → Dropout → Dense (multi-target, MSE).
• Training: concat sequences from all training CSVs, val_split=0.2, early stopping.
• Save: model → saved_model/player_tcn_model.keras.

• Eval demo: pick one CSV (len≥200), fit its own scaler, roll out N_steps=40
  autoregressively, inverse-transform, and plot inst_dist_m_sum
  to inst_dist_m_sum_autoregressive.png.

Edit paths: player_data_root / eval_data_root.
"""

import os
import re
import numpy as np
import pandas as pd
import pickle
import random
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tcn import TCN
import matplotlib.pyplot as plt

# === Parameters ===
input_window = 130
predict_ahead = 1
N_steps = 40
player_data_root = "/home/stav.karasik/MaccabiProject/scripts/games_subs_train_phy/games_train_new"
eval_data_root = "/home/stav.karasik/MaccabiProject/scripts/games_eval_sub_phy"
target_cols = [
    "inst_dist_m_sum", "Speed (m/s)_sum", "hsr_m_sum",
    "vha_count_1s_sum", "avg_jerk_1s_sum", "turns_per_sec_sum", "playerload_1s_sum",
    "walk_time_sum", "jog_time_sum", "run_time_sum", "sprint_time_sum",
    "total_sprints_sum", "sprint_attack_sum", "sprint_defense_sum",
    "dist_attack_sum", "dist_defense_sum", "time_attack_sum", "time_defense_sum",
    "attacking_third_time_sum", "middle_third_time_sum", "defending_third_time_sum"
]

# === Function: Sequence creation with per-player normalization ===
def create_sequences_with_scaler(df, input_window, predict_ahead, target_cols):
    df = df.sort_values("chunk")
    scaler = StandardScaler()
    df[target_cols] = scaler.fit_transform(df[target_cols])
    features = df[target_cols].copy()
    X, y = [], []
    for i in range(input_window, len(df) - predict_ahead):
        X.append(features.iloc[i - input_window:i].values)
        y.append(features.iloc[i + predict_ahead].values)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# === Load training data ===
X_train, y_train = [], []
for game_folder in os.listdir(player_data_root):
    game_path = os.path.join(player_data_root, game_folder)
    if not os.path.isdir(game_path):
        continue
    for file in os.listdir(game_path):
        if file.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(game_path, file))
                if not all(col in df.columns for col in target_cols):
                    continue
                if len(df) < input_window + predict_ahead:
                    continue
                X, y = create_sequences_with_scaler(df, input_window, predict_ahead, target_cols)
                if len(X) > 0:
                    X_train.append(X)
                    y_train.append(y)
            except Exception as e:
                print(f"⚠️ Skipping {file}: {e}")

X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)
print(f"✅ Total training sequences: {len(X_train)}")

# === Build TCN model ===
def build_tcn_model(input_window, num_features, output_size):
    model = Sequential([
        TCN(input_shape=(input_window, num_features), nb_filters=64, kernel_size=3, dilations=[1, 2, 4], return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(output_size)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_tcn_model(input_window, X_train.shape[2], y_train.shape[1])
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, validation_split=0.2, epochs=40, batch_size=32, callbacks=[early_stop])

# === Save model ===
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/player_tcn_model.keras")
print("✅ Model saved.")

# === Autoregressive prediction for evaluation ===
inst_idx = target_cols.index("inst_dist_m_sum")
valid_eval_files = []

for game_folder in os.listdir(eval_data_root):
    game_path = os.path.join(eval_data_root, game_folder)
    if not os.path.isdir(game_path):
        continue
    for f in os.listdir(game_path):
        if f.endswith(".csv"):
            file_path = os.path.join(game_path, f)
            try:
                df = pd.read_csv(file_path)
                if len(df) >= 200:
                    valid_eval_files.append(file_path)
            except Exception as e:
                print(f"⚠️ Skipping {file_path}: {e}")

if not valid_eval_files:
    raise ValueError("No evaluation CSV files with at least 200 chunks found.")

# === Pick one at random
chosen_file = random.choice(valid_eval_files)
print(f"📂 Evaluating file: {chosen_file}")
eval_df = pd.read_csv(chosen_file).sort_values("chunk").reset_index(drop=True)

# === Normalize eval file (per-player)
scaler = StandardScaler()
eval_df[target_cols] = scaler.fit_transform(eval_df[target_cols])

data = eval_df[target_cols].values
window = data[:input_window].copy()
preds_scaled = []
true_scaled = []

# === Autoregressive prediction loop
for i in range(N_steps):
    input_seq = np.expand_dims(window[-input_window:], axis=0)
    pred = model.predict(input_seq, verbose=0)[0]
    preds_scaled.append(pred)
    if input_window + i < len(data):
        true_scaled.append(data[input_window + i])
    window = np.vstack([window, pred])

# === Inverse transform results
preds_scaled = np.array(preds_scaled)
true_scaled = np.array(true_scaled)

# === Manually unscale just the inst_dist_m_sum column
mean = scaler.mean_[inst_idx]
std = np.sqrt(scaler.var_[inst_idx])

y_pred_inst = preds_scaled[:, inst_idx] * std + mean

if true_scaled.size > 0:
    y_true_inst = true_scaled[:, inst_idx] * std + mean
else:
    print("⚠️ No ground truth available — only predictions will be plotted.")
    y_true_inst = np.full_like(y_pred_inst, np.nan)

# === Plotting
plt.figure(figsize=(14, 5))
plt.plot(y_true_inst, label="True (inst_dist_m_sum)", marker='o')
plt.plot(y_pred_inst, label="Predicted (inst_dist_m_sum)", marker='x')
plt.xlabel("Time Step (15s each)")
plt.ylabel("inst_dist_m_sum")
plt.title("Autoregressive Prediction — inst_dist_m_sum (20 minutes)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("inst_dist_m_sum_autoregressive.png", dpi=300)
plt.close()
print("📊 Saved plot to inst_dist_m_sum_autoregressive.png")
