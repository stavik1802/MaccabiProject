
"""
Football GPS‑IMU‑HR feature engineering (extended v2).

Adds:
• Local X‑Y position            • Heading / bearing
• Instantaneous jerk            • Angular velocity magnitude
• 1‑s High‑speed‑running metres • Very‑high acceleration count
• Avg jerk (1‑s)                • Turn count / sec (1‑s)
• PlayerLoad™ (1‑s)             • Cumulative / 5‑min distance
• Work‑rate ratio               • Accel–decel balance
• Sprint totals (cum + 5‑min)   • Max rolling 1‑min speed

Outputs Parquet (if pyarrow/fastparquet available) or CSV fallback.
"""

from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from geopy.distance import geodesic

# CONFIG
SAMPLE_RATE_HZ = 10
HSR_THRESHOLD_MS = 5.5
SPRINT_THRESHOLD = 7.0
VHA_THRESHOLD_MS2 = 3.0
TURN_THRESHOLD_RS = 2.5

def process_half(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["Lat"].abs() > 1) & (df["Lon"].abs() > 1)].reset_index(drop=True)
    if df["Lat"].abs().max() > 360:
        df[["Lat", "Lon"]] *= 1e-6

    df["Time"] = pd.to_datetime(df["Time"], format="mixed")
    df["minute"] = df["Time"].dt.floor("min")

    coords = list(zip(df["Lat"], df["Lon"]))
    inst_dist = [0.0]
    duration_sec = (df["Time"].iloc[-1] - df["Time"].iloc[0]).total_seconds()
    sample_rate = len(df) / duration_sec
    frame_duration = 1 / sample_rate
    for i in range(1, len(coords)):
        dist = geodesic(coords[i - 1], coords[i]).meters
        inst_dist.append(dist)
    df["inst_dist_m"] = inst_dist

    speed = df["Speed (m/s)"]
    df["acc_long"] = speed.diff() * SAMPLE_RATE_HZ
    df["acc_mag"] = np.sqrt(df["Accl X"]**2 + df["Accl Y"]**2 + df["Accl Z"]**2)
    df["jerk"] = df["acc_mag"].diff() * SAMPLE_RATE_HZ
    df["ang_vel_mag"] = np.sqrt(
        np.radians(df["Gyro Yro X"])**2 +
        np.radians(df["Gyro Y"])**2 +
        np.radians(df["Gyro Z"])**2
    )

    df["hsr_flag"] = (speed > HSR_THRESHOLD_MS)
    df["hsr_m_1s"] = (df["hsr_flag"] * df["inst_dist_m"]).rolling(SAMPLE_RATE_HZ, min_periods=1).sum()
    df["vha_count_1s"] = (df["acc_mag"].abs() > VHA_THRESHOLD_MS2).rolling(SAMPLE_RATE_HZ, min_periods=1).sum()
    df["avg_jerk_1s"] = df["jerk"].rolling(SAMPLE_RATE_HZ, min_periods=1).mean()

    turn_flag = (df["ang_vel_mag"] > TURN_THRESHOLD_RS).astype(int)
    turn_start = (turn_flag.diff() == 1).fillna(0).astype(int)
    df["turns_per_sec"] = turn_start.rolling(SAMPLE_RATE_HZ, min_periods=1).sum()

    pl = np.sqrt(df["Accl X"].diff()**2 + df["Accl Y"].diff()**2 + df["Accl Z"].diff()**2)
    df["playerload_1s"] = pl.rolling(SAMPLE_RATE_HZ, min_periods=1).sum()

    df["pos_acc_events"] = (df["acc_long"] > VHA_THRESHOLD_MS2).cumsum()
    df["neg_acc_events"] = (df["acc_long"] < -VHA_THRESHOLD_MS2).cumsum()
    df["accel_decel_balance"] = df["pos_acc_events"] - df["neg_acc_events"]

    sprint_flag = (speed > SPRINT_THRESHOLD).astype(int)
    sprint_start = (sprint_flag.diff() == 1).fillna(0).astype(int)
    df["total_sprints"] = sprint_start.cumsum()

    df["walk_time"]   = ((speed >= 0) & (speed < 2)).astype(float) * frame_duration
    df["jog_time"]    = ((speed >= 2) & (speed < 4)).astype(float) * frame_duration
    df["run_time"]    = ((speed >= 4) & (speed < 7)).astype(float) * frame_duration
    df["sprint_time"] = (speed >= 7).astype(float) * frame_duration

    minute_stats = df.groupby("minute").agg({
        "inst_dist_m": "sum",
        "Speed (m/s)": ["mean", "max"],
        "hsr_m_1s": "sum",
        "vha_count_1s": "sum",
        "avg_jerk_1s": "mean",
        "turns_per_sec": "mean",
        "playerload_1s": "sum",
        "walk_time": "sum",
        "jog_time": "sum",
        "run_time": "sum",
        "sprint_time": "sum",
        "accel_decel_balance": "last",
        "total_sprints": lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0,
    }).reset_index()

    minute_stats.columns = ["minute"] + ["_".join(col) for col in minute_stats.columns[1:]]

    for col in ["walk_time_sum", "jog_time_sum", "run_time_sum", "sprint_time_sum"]:
        if col in minute_stats.columns:
            minute_stats[col] = minute_stats[col].round(1)

    minute_stats["minute"] = np.arange(1, len(minute_stats) + 1)
    return minute_stats

def process_player_folder(player_folder: Path):
    first_file = player_folder / "first_half.csv"
    second_file = player_folder / "second_half.csv"

    if not first_file.exists():
        print(f"❌ Missing first_half.csv in {player_folder.name}")
        return

    print(f"📂 Processing {player_folder.name}")

    df_first = pd.read_csv(first_file)
    df_second = pd.read_csv(second_file) if second_file.exists() else pd.DataFrame()

    features_first = process_half(df_first) if not df_first.empty else pd.DataFrame()
    features_second = process_half(df_second) if not df_second.empty else pd.DataFrame()

    # Save available half feature files
    if not features_first.empty:
        features_first.to_csv(player_folder / "first_half_features.csv", index=False)

    if not features_second.empty:
        if not features_first.empty:
            last_minute_first = features_first["minute"].iloc[-1]
            features_second["minute"] += last_minute_first
        features_second.to_csv(player_folder / "second_half_features.csv", index=False)

    # Save merged only if both halves exist
    if not features_first.empty and not features_second.empty:
        features_merged = pd.concat([features_first, features_second], ignore_index=True)
        features_merged.to_csv(player_folder / "merged_features.csv", index=False)
        print(f"✅ Saved merged_features.csv in {player_folder.name}/")
    else:
        print(f"⚠️ Skipped merged_features.csv – one half is empty in {player_folder.name}")



def main(parent_folder: str):
    parent_path = Path(parent_folder)
    player_folders = [f for f in parent_path.iterdir() if f.is_dir()]

    if not player_folders:
        print("❌ No player folders found.")
        return

    for player_folder in player_folders:
        try:
            process_player_folder(player_folder)
        except Exception as e:
            print(f"❌ Error in {player_folder.name}: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extract_features_from_halves.py <parent_folder>")
    else:
        main(sys.argv[1])

# from __future__ import annotations
# import math
# from pathlib import Path
# import numpy as np
# import pandas as pd
# from geopy.distance import geodesic
#
# # CONFIG
# CSV_FILE = Path("2024-08-17-mWaW-Entire-Session_second_half.csv")
# SAMPLE_RATE_HZ = 10
# HSR_THRESHOLD_MS = 5.5
# SPRINT_THRESHOLD = 7.0
# VHA_THRESHOLD_MS2 = 3.0
# TURN_THRESHOLD_RS = 2.5
#
# def main():
#     df = pd.read_csv(CSV_FILE)
#     df = df[(df["Lat"].abs() > 1) & (df["Lon"].abs() > 1)].reset_index(drop=True)
#
#     if df["Lat"].abs().max() > 360:
#         df[["Lat", "Lon"]] *= 1e-6
#
#     df["Time"] = pd.to_datetime(df["Time"], format="mixed")
#     df["minute"] = df["Time"].dt.floor("min")
#
#     coords = list(zip(df["Lat"], df["Lon"]))
#     inst_dist = [0.0]
#     duration_sec = (df["Time"].iloc[-1] - df["Time"].iloc[0]).total_seconds()
#     sample_rate = len(df) / duration_sec
#     FRAME_DURATION = 1 / sample_rate
#     for i in range(1, len(coords)):
#         dist = geodesic(coords[i - 1], coords[i]).meters
#         inst_dist.append(dist)
#     df["inst_dist_m"] = inst_dist
#
#     speed = df["Speed (m/s)"]
#     df["acc_long"] = speed.diff() * SAMPLE_RATE_HZ
#     df["acc_mag"] = np.sqrt(df["Accl X"]**2 + df["Accl Y"]**2 + df["Accl Z"]**2)
#     df["jerk"] = df["acc_mag"].diff() * SAMPLE_RATE_HZ
#     df["ang_vel_mag"] = np.sqrt(
#         np.radians(df["Gyro Yro X"])**2 +
#         np.radians(df["Gyro Y"])**2 +
#         np.radians(df["Gyro Z"])**2
#     )
#
#     # High-speed running
#     df["hsr_flag"] = (speed > HSR_THRESHOLD_MS)
#     df["hsr_m_1s"] = (df["hsr_flag"] * df["inst_dist_m"]).rolling(SAMPLE_RATE_HZ, min_periods=1).sum()
#     df["vha_count_1s"] = (df["acc_mag"].abs() > VHA_THRESHOLD_MS2).rolling(SAMPLE_RATE_HZ, min_periods=1).sum()
#     df["avg_jerk_1s"] = df["jerk"].rolling(SAMPLE_RATE_HZ, min_periods=1).mean()
#
#     # Turns
#     turn_flag = (df["ang_vel_mag"] > TURN_THRESHOLD_RS).astype(int)
#     turn_start = (turn_flag.diff() == 1).fillna(0).astype(int)
#     df["turns_per_sec"] = turn_start.rolling(SAMPLE_RATE_HZ, min_periods=1).sum()
#
#     # PlayerLoad
#     pl = np.sqrt(df["Accl X"].diff()**2 + df["Accl Y"].diff()**2 + df["Accl Z"].diff()**2)
#     df["playerload_1s"] = pl.rolling(SAMPLE_RATE_HZ, min_periods=1).sum()
#
#     # Accel/decel balance
#     df["pos_acc_events"] = (df["acc_long"] > VHA_THRESHOLD_MS2).cumsum()
#     df["neg_acc_events"] = (df["acc_long"] < -VHA_THRESHOLD_MS2).cumsum()
#     df["accel_decel_balance"] = df["pos_acc_events"] - df["neg_acc_events"]
#
#     # Sprints
#     sprint_flag = (speed > SPRINT_THRESHOLD).astype(int)
#     sprint_start = (sprint_flag.diff() == 1).fillna(0).astype(int)
#     df["total_sprints"] = sprint_start.cumsum()
#
#     # Movement categories
#     df["walk_time"]   = ((speed >= 0) & (speed < 2)).astype(float) * FRAME_DURATION
#     df["jog_time"]    = ((speed >= 2) & (speed < 4)).astype(float) * FRAME_DURATION
#     df["run_time"]    = ((speed >= 4) & (speed < 7)).astype(float) * FRAME_DURATION
#     df["sprint_time"] = (speed >= 7).astype(float) * FRAME_DURATION
#
#     # Per-minute aggregation
#     minute_stats = df.groupby("minute").agg({
#         "inst_dist_m": "sum",
#         "Speed (m/s)": ["mean", "max"],
#         "hsr_m_1s": "sum",
#         "vha_count_1s": "sum",
#         "avg_jerk_1s": "mean",
#         "turns_per_sec": "mean",
#         "playerload_1s": "sum",
#         "walk_time": "sum",
#         "jog_time": "sum",
#         "run_time": "sum",
#         "sprint_time": "sum",
#         "accel_decel_balance": "last",
#         "total_sprints": lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0,
#     }).reset_index()
#
#     # Flatten column names
#     minute_stats.columns = ["minute"] + ["_".join(col) for col in minute_stats.columns[1:]]
#
#     # Round durations
#     for col in ["walk_time_sum", "jog_time_sum", "run_time_sum", "sprint_time_sum"]:
#         if col in minute_stats.columns:
#             minute_stats[col] = minute_stats[col].round(1)
#
#     # Replace datetime with integer minute index
#     minute_stats["minute"] = np.arange(1, len(minute_stats) + 1)
#
#     # Save
#     out_path = f"per_minute_summary_all_features.csv"
#     minute_stats.to_csv(out_path, index=False)
#     print(f"✅ Saved: {out_path}")
#
# if __name__ == "__main__":
#     main()





### old version
# from __future__ import annotations
#
# import math
# from pathlib import Path
#
# import numpy as np
# import pandas as pd
# from geopy.distance import geodesic
#
# # ────────────────────── CONFIG ────────────────────────────────────────────
# CSV_FILE           = Path("2024-08-17-mWaW-Entire-Session_second_half.csv")
# SAMPLE_RATE_HZ     = 10          # 10 Hz data
# HSR_THRESHOLD_MS   = 5.5         # high‑speed ≥ 5.5 m/s
# SPRINT_THRESHOLD   = 7.0         # sprint ≥ 7 m/s
# VHA_THRESHOLD_MS2  = 3.0         # |acc| ≥ 3 m/s²
# TURN_THRESHOLD_RS  = 2.5         # |ω| ≥ 2.5 rad/s
# ROLL_1S            = 1  * SAMPLE_RATE_HZ
# ROLL_5M            = 5*60 * SAMPLE_RATE_HZ
# ROLL_1M            = 60  * SAMPLE_RATE_HZ
# # ──────────────────────────────────────────────────────────────────────────
#
#
# def latlon_to_xy(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
#     """Simple equirectangular projection around (lat0, lon0) → metres."""
#     R = 6_371_000
#     x = math.radians(lon - lon0) * R * math.cos(math.radians(lat0))
#     y = math.radians(lat - lat0) * R
#     return x, y
#
#
# def main() -> None:
#     if not CSV_FILE.exists():
#         raise FileNotFoundError(f"{CSV_FILE} not found – adjust CSV_FILE.")
#
#     df = pd.read_csv(CSV_FILE)
#
#
#     # ── 1. clean bad GPS rows ────────────────────────────────────────────
#     df = df[(df["Lat"].abs() > 1) & (df["Lon"].abs() > 1)].reset_index(drop=True)
#
#     # micro‑degree → degree conversion
#     if df["Lat"].abs().max() > 360:
#         df[["Lat", "Lon"]] *= 1e-6
#
#     # ── 2. time + local projection ──────────────────────────────────────
#     df["t_sec"] = np.arange(len(df)) / SAMPLE_RATE_HZ
#
#     coords = list(zip(df["Lat"], df["Lon"]))
#     inst_dist = [0.0]
#     for i in range(1, len(coords)):
#         dist = geodesic(coords[i - 1], coords[i]).meters
#         inst_dist.append(dist)
#     df["inst_dist_m"] = inst_dist
#
#     speed = df["Speed (m/s)"]
#     df["acc_long"] = speed.diff() * SAMPLE_RATE_HZ
#
#     # ── 3. instantaneous kinematics ─────────────────────────────────────
#     df["acc_mag"] = np.sqrt(df["Accl X"] ** 2 + df["Accl Y"] ** 2 + df["Accl Z"] ** 2)
#     df["jerk"] = df["acc_mag"].diff() * SAMPLE_RATE_HZ
#     df["ang_vel_mag"] = np.sqrt(
#         df["Gyro Yro X"] ** 2 + df["Gyro Y"] ** 2 + df["Gyro Z"] ** 2
#     )
#
#     # ── 4. rolling 1‑s features ─────────────────────────────────────────
#     roll1s = df.rolling(ROLL_1S, min_periods=1)
#
#     df["hsr_m_1s"] = ((df["Speed (m/s)"] > HSR_THRESHOLD_MS) * df["inst_dist_m"]).rolling(
#         ROLL_1S, min_periods=1
#     ).sum()
#
#     df["vha_count_1s"] = (
#         (df["acc_mag"].abs() > VHA_THRESHOLD_MS2).rolling(ROLL_1S).sum()
#     )
#
#     df["avg_jerk_1s"] = roll1s["jerk"].mean()
#
#     df["ang_vel_mag"] = np.sqrt(
#         np.radians(df["Gyro Yro X"]) ** 2 +  # convert on the fly
#         np.radians(df["Gyro Y"]) ** 2 +
#         np.radians(df["Gyro Z"]) ** 2
#     )
#
#     TURN_THRESHOLD_RS = 2.5  # rad/s  (≈ 143 deg/s)
#     turn_flag = (df["ang_vel_mag"] > TURN_THRESHOLD_RS).astype(int)
#     turn_start = (turn_flag.diff() == 1).fillna(0).astype(int)
#
#     df["turn_count"] = turn_start.cumsum()
#     df["turns_per_sec"] = turn_start.rolling(ROLL_1S).sum()
#
#     pl = np.sqrt(
#         df["Accl X"].diff() ** 2 + df["Accl Y"].diff() ** 2 + df["Accl Z"].diff() ** 2
#     )
#     df["playerload_1s"] = pl.rolling(ROLL_1S, min_periods=1).sum()
#
#     # ── 5. cumulative + 5‑min metrics ───────────────────────────────────
#     df["cum_dist_m"] = df["inst_dist_m"].cumsum()
#     df["cum_hsr_m"] = (
#         (df["Speed (m/s)"] > HSR_THRESHOLD_MS) * df["inst_dist_m"]
#     ).cumsum()
#     df["work_rate_ratio"] = df["cum_hsr_m"] / df["cum_dist_m"].replace(0, np.nan)
#
#     ACC_THR = VHA_THRESHOLD_MS2
#
#     df["pos_acc_events"] = (df["acc_long"] > ACC_THR).cumsum()
#     df["neg_acc_events"] = (df["acc_long"] < -ACC_THR).cumsum()
#     df["accel_decel_balance"] = df["pos_acc_events"] - df["neg_acc_events"]
#
#     sprint_flag = (df["Speed (m/s)"] > SPRINT_THRESHOLD).astype(int)
#     sprint_start = (sprint_flag.diff() == 1).fillna(0).astype(int)
#     df["total_sprints"] = sprint_start.cumsum()
#
#     roll5m = df.rolling(ROLL_5M, min_periods=1)
#     df["dist_m_5min"] = roll5m["inst_dist_m"].sum()
#     df["sprints_5min"] = roll5m["total_sprints"].apply(
#         lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else x.iloc[-1]
#     )
#
#     # max rolling 1‑min speed
#     df["max_speed_1min"] = df.rolling(ROLL_1M, min_periods=1)["Speed (m/s)"].max()
#
#     # ── 6. save result ──────────────────────────────────────────────────
#     out_base = CSV_FILE.with_name(f"features_{CSV_FILE.stem}")
#     try:
#         import pyarrow  # noqa: F401  (just to check availability)
#
#         out_file = out_base.with_suffix(".parquet")
#         df.to_parquet(out_file, index=False, compression="zstd")
#         fmt = "Parquet"
#     except ImportError:
#         out_file = out_base.with_suffix(".csv")
#         df.to_csv(out_file, index=False)
#         fmt = "CSV (pyarrow missing)"
#
#     print(f"✅  Feature file saved as {out_file.name}  [{fmt}]")
#     new_cols = (
#         set(df.columns)
#         - {
#             "Lat",
#             "Lon",
#             "Accl X",
#             "Accl Y",
#             "Accl Z",
#             "Gyro Yro X",
#             "Gyro Y",
#             "Gyro Z",
#         }
#     )
#     print(df[["t_sec", "Speed (m/s)", "inst_dist_m", "dist_m_5min"]].head(3100).tail())
#
#
# if __name__ == "__main__":
#     main()
