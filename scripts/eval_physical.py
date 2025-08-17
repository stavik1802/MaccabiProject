# this is eval for the physical features prediction model

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model
from tcn import TCN

# === Parameters ===
input_window = 115
predict_chunks = 40
start_chunk = 240
player_data_root = "games_eval_new"
target_cols = [
    "inst_dist_m_sum", "Speed (m/s)_sum",  "hsr_m_sum",
    "vha_count_1s_sum", "avg_jerk_1s_sum", "turns_per_sec_sum", "playerload_1s_sum",
    "walk_time_sum", "jog_time_sum", "run_time_sum", "sprint_time_sum",
    "total_sprints_sum", "sprint_attack_sum", "sprint_defense_sum",
    "dist_attack_sum", "dist_defense_sum", "time_attack_sum", "time_defense_sum",
    "attacking_third_time_sum", "middle_third_time_sum", "defending_third_time_sum"
]

# === Load trained model ===
model = load_model("saved_model/player_tcn_model.keras", custom_objects={"TCN": TCN})

# === Evaluation containers ===
max_players = 50
player_count = 0
y_pred_all = defaultdict(lambda: [[] for _ in range(predict_chunks)])
y_true_all = defaultdict(lambda: [[] for _ in range(predict_chunks)])
valid_chunks = []

# === Loop through evaluation files ===
for root, dirs, files in os.walk(player_data_root):
    for file in files:
        if not file.endswith(".csv") or player_count >= max_players:
            continue

        file_path = os.path.join(root, file)
        try:
            df = pd.read_csv(file_path).sort_values("chunk").reset_index(drop=True)
            if not all(col in df.columns for col in target_cols):
                continue

            idx_list = df.index[df["chunk"] == start_chunk].tolist()
            if not idx_list:
                continue

            start_idx = idx_list[0]
            if start_idx < input_window or start_idx + predict_chunks >= len(df):
                continue

            # Per-player normalization
            scaler = StandardScaler()
            df[target_cols] = scaler.fit_transform(df[target_cols])

            input_seq = df[target_cols].iloc[start_idx - input_window:start_idx].values
            input_buffer = input_seq.copy()

            if not valid_chunks:
                valid_chunks = [start_chunk + step for step in range(predict_chunks)]

            for step in range(predict_chunks):
                model_input = input_buffer[-input_window:].reshape(1, input_window, -1)
                y_pred_norm = model.predict(model_input, verbose=0)[0]

                # Roll forward prediction
                next_row = input_buffer[-1].copy()
                for j, col in enumerate(target_cols):
                    next_row[j] = y_pred_norm[j]
                input_buffer = np.vstack([input_buffer, next_row])[1:]

                true_row = df[target_cols].iloc[start_idx + step].values
                for j, col in enumerate(target_cols):
                    mean = scaler.mean_[j]
                    std = np.sqrt(scaler.var_[j])
                    pred_val = y_pred_norm[j] * std + mean
                    true_val = true_row[j] * std + mean
                    y_pred_all[col][step].append(pred_val)
                    y_true_all[col][step].append(true_val)

            player_count += 1
            print(f"✅ Processed {player_count}: {file_path}")

        except Exception as e:
            print(f"❌ Error in {file_path}: {e}")

if player_count == 0:
    raise RuntimeError("❌ No valid evaluation files found.")

print(f"\n📊 Averaging results from {player_count} players...\n")

# === Aggregate and save plots ===
avg_pred_all = {col: [np.mean(step_vals) for step_vals in steps] for col, steps in y_pred_all.items()}
avg_true_all = {col: [np.mean(step_vals) for step_vals in steps] for col, steps in y_true_all.items()}

os.makedirs("autoregressive_plots", exist_ok=True)
minutes = [chunk / 4 for chunk in valid_chunks]

for col in target_cols:
    plt.figure(figsize=(14, 5))
    plt.plot(minutes, avg_true_all[col], label="True (mean)", marker="o")
    plt.plot(minutes, avg_pred_all[col], label="Predicted (mean)", marker="x")
    plt.title(f"Autoregressive Prediction (Mean over {player_count} Players) — {col}")
    plt.xlabel("Minute")
    plt.ylabel(col)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    safe_col = re.sub(r'[^a-zA-Z0-9_\-]', "_", col)
    plt.savefig(f"autoregressive_plots/autoreg_{safe_col}_minute.png")
    plt.close()

# === Save to CSV ===
rows = []
for i, chunk in enumerate(valid_chunks):
    for col in target_cols:
        rows.append({
            "chunk": chunk,
            "minute": chunk / 4,
            "feature": col,
            "true_value": avg_true_all[col][i],
            "predicted_value": avg_pred_all[col][i]
        })
df_results = pd.DataFrame(rows)
df_results.to_csv("autoregressive_plots/autoregressive_predictions_mean_minute.csv", index=False)

# === MAE Plot ===
mae_per_chunk = []
for i in range(len(valid_chunks)):
    true_vals = [avg_true_all[col][i] for col in target_cols]
    pred_vals = [avg_pred_all[col][i] for col in target_cols]
    mae = mean_absolute_error(true_vals, pred_vals)
    mae_per_chunk.append(mae)

plt.figure(figsize=(12, 5))
plt.plot(minutes, mae_per_chunk, marker='o')
plt.title("Mean Absolute Error per Minute (Averaged Across Players)")
plt.xlabel("Minute")
plt.ylabel("MAE")
plt.grid(True)
plt.tight_layout()
plt.savefig("autoregressive_plots/mae_per_minute_mean.png")
plt.close()

print("✅ Evaluation complete. Plots and CSV saved.")
