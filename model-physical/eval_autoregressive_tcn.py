# import os
# import re
# import random
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from collections import defaultdict
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import load_model
# from tcn import TCN

# # === Parameters ===
# input_window = 10
# predict_minutes = 10
# min_required_chunks = 50  # 50 halves, each is a prediction chunk
# player_data_root = "player_data"
# target_cols = [
#     "inst_dist_m_sum", "Speed (m/s)_mean", "Speed (m/s)_max", "hsr_m_sum", "vha_count_1s_sum",
#     "avg_jerk_1s_mean", "turns_per_sec_sum", "playerload_1s_sum", "walk_time_sum", "jog_time_sum",
#     "run_time_sum", "sprint_time_sum", "total_sprints_sum", "sprint_attack_sum", "sprint_defense_sum",
#     "dist_attack_sum", "dist_defense_sum", "time_attack_sum", "time_defense_sum",
#     "attacking_third_time_sum", "middle_third_time_sum", "defending_third_time_sum"
# ]

# # === Load model
# model = load_model("saved_model/player_tcn_model.keras", custom_objects={"TCN": TCN})

# # === Step 1: Collect all player files by date and half
# player_game_files = defaultdict(lambda: defaultdict(dict))  # player -> date -> {"first": path, "second": path}

# for player_folder in os.listdir(player_data_root):
#     player_path = os.path.join(player_data_root, player_folder)
#     if not os.path.isdir(player_path):
#         continue
#     for file in os.listdir(player_path):
#         if file.endswith(".csv"):
#             parts = file.split("_")
#             if len(parts) < 2:
#                 continue
#             date = parts[0]
#             half = "first" if "first" in file else "second" if "second" in file else None
#             if half:
#                 player_game_files[player_folder][date][half] = os.path.join(player_path, file)

# # === Step 2: Collect halves from players who have both halves for a game
# chunks = []
# for player, games in player_game_files.items():
#     for date, halves in games.items():
#         if "first" in halves and "second" in halves:
#             chunks.append((player, date, "first", halves["first"]))
#             chunks.append((player, date, "second", halves["second"]))

# random.seed(42)
# random.shuffle(chunks)

# # === Step 3: Predict on chunks until we have 50
# rows = []
# used_chunks = 0

# for player, date, half_label, file_path in chunks:
#     try:
#         df = pd.read_csv(file_path)
#     except:
#         continue

#     if df.shape[0] <= input_window + predict_minutes:
#         continue

#     df = df.sort_values("minute").reset_index(drop=True)
#     df["source_file"] = os.path.basename(file_path)
#     df["source_folder"] = player

#     feature_cols = df.columns.drop(["minute", "source_file", "source_folder"])
#     scalers = {col: StandardScaler().fit(df[[col]]) for col in feature_cols}
#     for col in feature_cols:
#         df[col] = scalers[col].transform(df[[col]])

#     input_sequence = df.loc[:input_window-1, feature_cols].values
#     input_buffer = input_sequence.copy()
#     start_minute = df.loc[input_window-1, "minute"]

#     for step in range(predict_minutes):
#         model_input = input_buffer[-input_window:].reshape(1, input_window, -1)
#         y_pred_norm = model.predict(model_input, verbose=0)[0]

#         next_row = input_buffer[-1].copy()
#         for j, col in enumerate(target_cols):
#             next_row[j] = y_pred_norm[j]
#         input_buffer = np.vstack([input_buffer, next_row])[1:]

#         current_index = input_window + step
#         if current_index >= len(df):
#             break

#         true_row = df.loc[current_index, target_cols]
#         minute = start_minute + step + 1

#         for j, col in enumerate(target_cols):
#             scaler = scalers[col]
#             pred_val_real = y_pred_norm[j] * scaler.scale_[0] + scaler.mean_[0]
#             true_val_real = true_row[col] * scaler.scale_[0] + scaler.mean_[0]
#             rows.append({
#                 "minute": minute,
#                 "feature": col,
#                 "true_value": true_val_real,
#                 "predicted_value": pred_val_real
#             })

#     used_chunks += 1
#     if used_chunks >= min_required_chunks:
#         break

# if used_chunks == 0:
#     print("❌ No valid chunks found.")
#     exit()

# print(f"✅ Used {used_chunks} player halves with both halves available.")

# # === Step 4: Aggregate and plot
# df_results = pd.DataFrame(rows)
# df_summary = df_results.groupby(["minute", "feature"]).agg({
#     "true_value": "mean",
#     "predicted_value": "mean"
# }).reset_index()

# os.makedirs("autoregressive_plots", exist_ok=True)
# df_summary.to_csv("autoregressive_plots/autoregressive_predictions_50_chunks.csv", index=False)

# for col in target_cols:
#     df_feat = df_summary[df_summary["feature"] == col]
#     plt.figure(figsize=(14, 5))
#     plt.plot(df_feat["minute"], df_feat["true_value"], label="True", marker="o")
#     plt.plot(df_feat["minute"], df_feat["predicted_value"], label="Predicted", marker="x")
#     plt.title(f"Autoregressive Prediction — {col}")
#     plt.xlabel("Minute")
#     plt.ylabel(col)
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     safe_col = re.sub(r'[^a-zA-Z0-9_\-]', "_", col)
#     plt.savefig(f"autoregressive_plots/autoreg_50chunks_{safe_col}.png")
#     plt.close()

# print("📁 Results saved in: autoregressive_plots/")
