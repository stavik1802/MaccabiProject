
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

# ────────────────────── CONFIG ────────────────────────────────────────────
CSV_FILE           = Path("2024-08-17-xPlkAkl/2024-08-17-xPIkAkI-Entire-Session_full.csv")
SAMPLE_RATE_HZ     = 10          # 10 Hz data
HSR_THRESHOLD_MS   = 5.5         # high‑speed ≥ 5.5 m/s
SPRINT_THRESHOLD   = 7.0         # sprint ≥ 7 m/s
VHA_THRESHOLD_MS2  = 3.0         # |acc| ≥ 3 m/s²
TURN_THRESHOLD_RS  = 2.5         # |ω| ≥ 2.5 rad/s
ROLL_1S            = 1  * SAMPLE_RATE_HZ
ROLL_5M            = 5*60 * SAMPLE_RATE_HZ
ROLL_1M            = 60  * SAMPLE_RATE_HZ
# ──────────────────────────────────────────────────────────────────────────


def latlon_to_xy(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Simple equirectangular projection around (lat0, lon0) → metres."""
    R = 6_371_000
    x = math.radians(lon - lon0) * R * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * R
    return x, y


def main() -> None:
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"{CSV_FILE} not found – adjust CSV_FILE.")

    df = pd.read_csv(CSV_FILE)


    # ── 1. clean bad GPS rows ────────────────────────────────────────────
    df = df[(df["Lat"].abs() > 1) & (df["Lon"].abs() > 1)].reset_index(drop=True)

    # micro‑degree → degree conversion
    if df["Lat"].abs().max() > 360:
        df[["Lat", "Lon"]] *= 1e-6

    # ── 2. time + local projection ──────────────────────────────────────
    df["t_sec"] = np.arange(len(df)) / SAMPLE_RATE_HZ

    coords = list(zip(df["Lat"], df["Lon"]))
    inst_dist = [0.0]
    for i in range(1, len(coords)):
        dist = geodesic(coords[i - 1], coords[i]).meters
        inst_dist.append(dist)
    df["inst_dist_m"] = inst_dist

    speed = df["Speed (m/s)"]
    df["acc_long"] = speed.diff() * SAMPLE_RATE_HZ

    # ── 3. instantaneous kinematics ─────────────────────────────────────
    df["acc_mag"] = np.sqrt(df["Accl X"] ** 2 + df["Accl Y"] ** 2 + df["Accl Z"] ** 2)
    df["jerk"] = df["acc_mag"].diff() * SAMPLE_RATE_HZ
    df["ang_vel_mag"] = np.sqrt(
        df["Gyro Yro X"] ** 2 + df["Gyro Y"] ** 2 + df["Gyro Z"] ** 2
    )

    # ── 4. rolling 1‑s features ─────────────────────────────────────────
    roll1s = df.rolling(ROLL_1S, min_periods=1)

    df["hsr_m_1s"] = ((df["Speed (m/s)"] > HSR_THRESHOLD_MS) * df["inst_dist_m"]).rolling(
        ROLL_1S, min_periods=1
    ).sum()

    df["vha_count_1s"] = (
        (df["acc_mag"].abs() > VHA_THRESHOLD_MS2).rolling(ROLL_1S).sum()
    )

    df["avg_jerk_1s"] = roll1s["jerk"].mean()

    df["ang_vel_mag"] = np.sqrt(
        np.radians(df["Gyro Yro X"]) ** 2 +  # convert on the fly
        np.radians(df["Gyro Y"]) ** 2 +
        np.radians(df["Gyro Z"]) ** 2
    )

    TURN_THRESHOLD_RS = 2.5  # rad/s  (≈ 143 deg/s)
    turn_flag = (df["ang_vel_mag"] > TURN_THRESHOLD_RS).astype(int)
    turn_start = (turn_flag.diff() == 1).fillna(0).astype(int)

    df["turn_count"] = turn_start.cumsum()
    df["turns_per_sec"] = turn_start.rolling(ROLL_1S).sum()

    pl = np.sqrt(
        df["Accl X"].diff() ** 2 + df["Accl Y"].diff() ** 2 + df["Accl Z"].diff() ** 2
    )
    df["playerload_1s"] = pl.rolling(ROLL_1S, min_periods=1).sum()

    # ── 5. cumulative + 5‑min metrics ───────────────────────────────────
    df["cum_dist_m"] = df["inst_dist_m"].cumsum()
    df["cum_hsr_m"] = (
        (df["Speed (m/s)"] > HSR_THRESHOLD_MS) * df["inst_dist_m"]
    ).cumsum()
    df["work_rate_ratio"] = df["cum_hsr_m"] / df["cum_dist_m"].replace(0, np.nan)

    ACC_THR = VHA_THRESHOLD_MS2

    df["pos_acc_events"] = (df["acc_long"] > ACC_THR).cumsum()
    df["neg_acc_events"] = (df["acc_long"] < -ACC_THR).cumsum()
    df["accel_decel_balance"] = df["pos_acc_events"] - df["neg_acc_events"]

    sprint_flag = (df["Speed (m/s)"] > SPRINT_THRESHOLD).astype(int)
    sprint_start = (sprint_flag.diff() == 1).fillna(0).astype(int)
    df["total_sprints"] = sprint_start.cumsum()

    roll5m = df.rolling(ROLL_5M, min_periods=1)
    df["dist_m_5min"] = roll5m["inst_dist_m"].sum()
    df["sprints_5min"] = roll5m["total_sprints"].apply(
        lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else x.iloc[-1]
    )

    # max rolling 1‑min speed
    df["max_speed_1min"] = df.rolling(ROLL_1M, min_periods=1)["Speed (m/s)"].max()

    # ── 6. save result ──────────────────────────────────────────────────
    out_base = CSV_FILE.with_name(f"features_{CSV_FILE.stem}")
    try:
        import pyarrow  # noqa: F401  (just to check availability)

        out_file = out_base.with_suffix(".parquet")
        df.to_parquet(out_file, index=False, compression="zstd")
        fmt = "Parquet"
    except ImportError:
        out_file = out_base.with_suffix(".csv")
        df.to_csv(out_file, index=False)
        fmt = "CSV (pyarrow missing)"

    print(f"✅  Feature file saved as {out_file.name}  [{fmt}]")
    new_cols = (
        set(df.columns)
        - {
            "Lat",
            "Lon",
            "Accl X",
            "Accl Y",
            "Accl Z",
            "Gyro Yro X",
            "Gyro Y",
            "Gyro Z",
        }
    )
    print(df[["t_sec", "Speed (m/s)", "inst_dist_m", "dist_m_5min"]].head(3100).tail())


if __name__ == "__main__":
    main()
