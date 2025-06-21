"""
Script: main.py

Description:
    This is the main processing script for football player tracking analysis. It processes GPS and IMU data
    to generate comprehensive performance metrics for each player. The script handles both halves of the game,
    accounts for substitutions, and provides detailed movement and performance analysis.
    There ways of visualizing the thirds on the field spread for player and the heatmap of the player.(The use of this is commented) 

Key Features:
    - Processes GPS and IMU data at 100Hz sampling rate
    - Calculates local X-Y positions relative to field orientation
    - Computes advanced metrics including:
        • High-speed running (HSR) and sprint analysis
        • Acceleration and deceleration patterns
        • PlayerLoad™ and work rate calculations
        • Turn detection and angular velocity analysis
        • Field position analysis (thirds of the field)
        • Attack/defense movement patterns
        • Speed zones (walk, jog, run, sprint)
        • Instantaneous distance and movement vectors
        • Jerk and angular velocity magnitude
        • Very-high acceleration counts
        • Average jerk per second
        • Turn count per second
        • Cumulative and 5-minute distance metrics
        • Acceleration-deceleration balance
        • Sprint totals (cumulative and 5-minute)
        • Maximum rolling 1-minute speed
        • Time spent in each third of the field
        • Attack/defense time distribution
        • Positive and negative acceleration counts
        • Movement direction projection
    - Handles both halves of the game with proper time synchronization
    - Accounts for player substitutions and partial game participation
    - Generates per-minute statistics and aggregated metrics

Input:
    - Parent folder containing player data folders
    - Center forward (CF) and center back (CB) CSV files for attack direction
    - First and second half start times
    - Field frame data (from field_frame.json)

Output:
    For each player folder:
    - first_half_features.csv: First half metrics
    - second_half_features.csv: Second half metrics
    - merged_features.csv: Combined game metrics
    Features include:
    - Speed zones (walk, jog, run, sprint)
    - Distance covered (total, attack, defense)
    - High-speed running metrics
    - Acceleration/deceleration counts
    - Turn analysis
    - Field position time distribution
    - PlayerLoad™ and work rate metrics
    - Per-minute statistics:
        • Total distance
        • Average and maximum speed
        • HSR distance
        • VHA count
        • Average jerk
        • Turn count
        • PlayerLoad™
        • Time in each speed zone
        • Attack/defense balance
        • Sprint counts
        • Distance in attack/defense
        • Time in attack/defense
        • Time in each third of the field

Usage:
    python main.py <parent_folder> <cf_csv> <cb_csv> [first_half_start] [second_half_start]
    Example: python main.py ./match_data cf_tracking.csv cb_tracking.csv "18:30:00.000" "19:15:00.000"

Football GPS‑IMU‑HR feature engineering (extended v2).

Adds:
• Local X‑Y position            • Heading / bearing
• Instantaneous jerk            • Angular velocity magnitude
• 1‑s High‑speed‑running metres • Very‑high acceleration count
• Avg jerk (1‑s)                • Turn count / sec (1‑s)
• PlayerLoad™ (1‑s)             • Cumulative / 5‑min distance
• Work‑rate ratio               • Accel–decel balance
• Sprint totals (cum + 5‑min)   • Max rolling 1‑min speed

Outputs Parquet (if pyarrow/fastparquet available) or CSV fallback.
"""

from __future__ import annotations
import math
import json
from pathlib import Path
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.decomposition import PCA
from attacking_direction import infer_attack_vector_only, infer_attack_vector_cf  # ✅ NEW

# CONFIG
SAMPLE_RATE_HZ = 100
HSR_THRESHOLD_MS = 5.5
SPRINT_THRESHOLD = 7.0
VHA_THRESHOLD_MS2 = 4.0
TURN_THRESHOLD_RS = 2.5 # 2.5
JITTER_METRES = 0.07

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def draw_pitch(ax, field_length=105, field_width=68):
    """Draw a simple football pitch on the given axes."""
    # Pitch outline
    ax.plot([-field_length/2, field_length/2], [-field_width/2, -field_width/2], color="white")
    ax.plot([-field_length/2, field_length/2], [field_width/2, field_width/2], color="white")
    ax.plot([-field_length/2, -field_length/2], [-field_width/2, field_width/2], color="white")
    ax.plot([field_length/2, field_length/2], [-field_width/2, field_width/2], color="white")

    # Halfway line
    ax.plot([0, 0], [-field_width/2, field_width/2], color="white")

    # Center circle
    center_circle = plt.Circle((0, 0), 9.15, color="white", fill=False)
    ax.add_patch(center_circle)
    ax.plot(0, 0, 'wo')  # Center point

    # Penalty areas
    for x in [-field_length/2, field_length/2]:
        sign = 1 if x > 0 else -1
        ax.plot([x, x - sign*16.5], [-16.5, -16.5], color="white")
        ax.plot([x, x - sign*16.5], [16.5, 16.5], color="white")
        ax.plot([x - sign*16.5, x - sign*16.5], [-16.5, 16.5], color="white")

    ax.set_facecolor("green")
    ax.set_xlim(-field_length/2, field_length/2)
    ax.set_ylim(-field_width/2, field_width/2)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_one_half_heatmap_from_xy(df: pd.DataFrame, save_path=None, cmap="YlOrRd",flip =False):
    if flip:
        df["X"] = -df["X"]
        df["Y"] = -df["Y"]
    fig, ax = plt.subplots(figsize=(12, 8))
    draw_pitch(ax)

    sns.kdeplot(
        x=df["X"], y=df["Y"],
        fill=True, cmap=cmap,
        bw_adjust=0.4,
        levels=100,
        thresh=0.02,
        alpha=0.8
    )

    ax.set_title("Heatmap - One Half")
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Heatmap saved to {save_path}")
        plt.close()
    else:
        plt.show()



def plot_field_thirds(
    df: pd.DataFrame,
    thirds_bounds: tuple[float, float, float],
    title: str = "Field Thirds Visualization",
    save_path: str = None,
    attacking_positive_x: bool = True
):
    """
    Plot player positions color-coded by field third, with third boundaries.
    
    Args:
        df: DataFrame with 'X', 'Y', and 'third' columns
        thirds_bounds: (x_min, x_mid1, x_mid2) used to draw third boundaries
        title: Plot title
        save_path: If provided, saves the figure to this path
        attacking_positive_x: If False, will flip X axis to visualize right-to-left attack
    """
    if df.empty or not all(col in df.columns for col in ["X", "Y", "third"]):
        print("⚠️ Missing required columns in DataFrame.")
        return

    # Optionally flip the X values for visualization
    plot_x = df["X"] if attacking_positive_x else -df["X"]
    plot_y = df["Y"]

    colors = {
        "attacking": "red",
        "middle": "yellow",
        "defending": "blue"
    }

    plt.figure(figsize=(12, 7))
    for third in ["defending", "middle", "attacking"]:
        sub_df = df[df["third"] == third]
        plt.scatter(
            plot_x[sub_df.index],
            plot_y[sub_df.index],
            s=5,
            alpha=0.6,
            label=third,
            color=colors[third]
        )

    # Add vertical lines for third boundaries
    x_min, x_mid1, x_mid2 = thirds_bounds
    if not attacking_positive_x:
        x_min, x_mid1, x_mid2 = -x_min, -x_mid1, -x_mid2

    plt.axvline(x=x_mid1, color='black', linestyle='--', label="Third boundary 1")
    plt.axvline(x=x_mid2, color='black', linestyle='--', label="Third boundary 2")

    plt.xlabel("Local X (meters)")
    plt.ylabel("Local Y (meters)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()

    plt.savefig("filename.png")
    print("saved")
    plt.close()


def load_field_frame(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)
    center = (data["center"]["lat"], data["center"]["lon"])
    corners = [(c["lat"], c["lon"]) for c in sorted(data["corners_latlon"], key=lambda d: d["corner"])]
    return center, corners, data["field_bounds_m"]


def gps_to_local_xy(df, center, corners, pca=None):
    """
    Convert GPS coordinates to local X-Y coordinates.
    If pca is provided, use it for transformation. Otherwise, calculate it from corners.
    """
    if pca is None:
        # Calculate PCA from field corners
        corner_coords = np.c_[
            [(c[0] - center[0]) * 111_132 for c in corners],  # Latitude in meters
            [(c[1] - center[1]) * 111_320 * np.cos(np.radians(center[0])) for c in corners]  # Longitude in meters
        ]
        pca = PCA(n_components=2)
        pca.fit(corner_coords)

    # If df is empty, just return it with the PCA
    if df.empty:
        return df, pca

    coords = np.c_[
        (df["Lat"].to_numpy() - center[0]) * 111_132,
        (df["Lon"].to_numpy() - center[1]) * 111_320 * np.cos(np.radians(center[0]))
    ]
    xy = pca.transform(coords)
    df["X"] = xy[:, 0]
    df["Y"] = xy[:, 1]
    return df, pca



def compute_third_flags(df: pd.DataFrame, attacking_positive_x: bool, thirds_bounds: tuple[float, float, float], x_max: float) -> pd.DataFrame:
    """
    Compute third flags for the field position.
    
    Args:
        df: DataFrame with X coordinates
        attacking_positive_x: Whether attacking direction is positive X
        thirds_bounds: (x_min, x_mid1, x_mid2) - boundaries for thirds
        x_max: Maximum X coordinate of the field
    """
    x_min, x_mid1, x_mid2 = thirds_bounds
    
    # Make a copy of X for thirds calculation
    df["X_for_thirds"] = df["X"].copy()
    
    if not attacking_positive_x:
        df["X_for_thirds"] = -df["X_for_thirds"]
    
    # Use x_mid2 for the second boundary to create three even sections
    df["third"] = pd.cut(
    df["X_for_thirds"],
    bins=[float('-inf'), x_mid1, x_mid2, float('inf')],
    labels=["defending", "middle", "attacking"]
    )
    
    # Drop the temporary column
    df = df.drop("X_for_thirds", axis=1)
    return df


def third_time_by_minute(df: pd.DataFrame, frame_duration: float) -> pd.DataFrame:
    """
    Calculate the time spent in each third of the field per minute.
    
    Args:
        df: DataFrame with minute and third columns
        frame_duration: Duration of each frame in seconds
        
    Returns:
        DataFrame with minutes and time spent in each third (in seconds)
    """
    
    if "minute" not in df.columns or "third" not in df.columns:
        return pd.DataFrame()
    
    # Count frames per minute and third
    counts = df.groupby(["minute", "third"], observed=True).size().unstack(fill_value=0)
    
    # Calculate total frames per minute for normalization
    total_frames_per_minute = counts.sum(axis=1)
    
    # Convert to proportions and then to seconds (60 seconds per minute)
    proportions = counts.div(total_frames_per_minute, axis=0)
    time_sec = proportions * 60
    
    time_sec.columns = [f"time_{col}_sec" for col in time_sec.columns]
    
    # Verify sums
    sums = time_sec.sum(axis=1)
    
    return time_sec.reset_index()


def process_half(df: pd.DataFrame, center, corners, attacking_positive_x: bool, 
                half_start_time: str,
                attack_vector: np.ndarray = None,
                field_bounds: dict[str, float] | None = None,
                pca = None,heatmap_path=None) -> pd.DataFrame:
    """
    Process a half of the game data.
    
    Args:
        df: DataFrame containing the player tracking data
        center: Field center coordinates
        corners: Field corner coordinates
        attacking_positive_x: Whether attacking direction is positive X
        half_start_time: Start time of the half
        attack_vector: Vector indicating attacking direction
        field_bounds: Dictionary containing field boundaries with keys: x_min, x_max, y_min, y_max
        pca: Pre-calculated PCA transformation for coordinate conversion
    """
    
    df = df[(df["Lat"].abs() > 1) & (df["Lon"].abs() > 1)].reset_index(drop=True)
    if df["Lat"].abs().max() > 360:
        df[["Lat", "Lon"]] *= 1e-6
    
    # Parse times with flexible format handling
    try:
        # First check if Time is already datetime
        if not pd.api.types.is_datetime64_any_dtype(df["Time"]):
            # If it's just a time string (HH:MM:SS.S), add today's date
            if ":" in str(df["Time"].iloc[0]) and len(str(df["Time"].iloc[0])) <= 12:
                today = pd.Timestamp.today().normalize()
                df["Time"] = pd.to_datetime(today.strftime("%Y-%m-%d ") + df["Time"].astype(str), format="mixed")
            else:
                # If it's a full datetime string
                df["Time"] = pd.to_datetime(df["Time"], format="mixed")
    except Exception as e:
        print(f"⚠️ Error processing times: {e}")
        return pd.DataFrame()

    # Parse half start time
    try:
        if ":" in half_start_time and len(half_start_time) <= 12:
            # If it's just a time string (HH:MM:SS.S), add today's date
            today = pd.Timestamp.today().normalize()
            half_start = pd.to_datetime(today.strftime("%Y-%m-%d ") + half_start_time, format="mixed")
        else:
            # If it's a full datetime string
            half_start = pd.to_datetime(half_start_time, format="mixed")
    except Exception as e:
        print(f"⚠️ Error processing half start time: {e}")
        return pd.DataFrame()
    
    if df.empty:
        print("❌ Empty DataFrame after initial processing")
        return pd.DataFrame()

    # Calculate minutes since half start, ensuring we only compare times within the same day
    df["seconds"] = (df["Time"] - df["Time"].dt.normalize()) / pd.Timedelta(seconds=1)
    half_start_seconds = (half_start - half_start.normalize()) / pd.Timedelta(seconds=1)
    
    minutes_offset = (df["seconds"].iloc[0] - half_start_seconds) / 60
    
    df["minute"] = ((df["seconds"] - half_start_seconds) / 60).astype(int) + 1
    
    coords = list(zip(df["Lon"], df["Lat"]))
    inst_dist = [0.0]
    movement_vectors = [(0.0, 0.0)]
    duration_sec = (df["Time"].iloc[-1] - df["Time"].iloc[0]).total_seconds()
    sample_rate = len(df) / duration_sec
    frame_duration = 1 / sample_rate
    
    for i in range(1, len(coords)):
        dist = geodesic(coords[i - 1], coords[i]).meters
        dist = dist if dist >= JITTER_METRES else 0.0
        inst_dist.append(dist)
        dx = df["Lon"].iloc[i] - df["Lon"].iloc[i - 1]   # Longitude = X
        dy = df["Lat"].iloc[i] - df["Lat"].iloc[i - 1]   # Latitude = Y

        movement_vectors.append((dx, dy) if dist >= 0.0 else (0.0, 0.0))

    df["inst_dist_m"] = inst_dist
    df["movement_dx"] = [v[0] for v in movement_vectors]
    df["movement_dy"] = [v[1] for v in movement_vectors]

    # Local coordinates conversion
    df, pca = gps_to_local_xy(df, center, corners, pca)
    
    
    # Calculate thirds bounds from field_bounds
    if field_bounds is not None and isinstance(field_bounds, dict):
        x_min = field_bounds.get("x_min")
        x_max = field_bounds.get("x_max")
        if x_min is not None and x_max is not None:
            x_range = x_max - x_min
            x_mid1 = x_min + x_range / 3
            x_mid2 = x_min + 2 * x_range / 3
            thirds_bounds = (x_min, x_mid1, x_mid2)
        else:
            # Fallback to calculating from data if field_bounds missing required values
            x_min = df["X"].min()
            x_max = df["X"].max()
            x_range = x_max - x_min
            x_mid1 = x_min + x_range / 3
            x_mid2 = x_min + 2 * x_range / 3
            thirds_bounds = (x_min, x_mid1, x_mid2)
    else:
        # Fallback to calculating from data if field_bounds not provided
        x_min = df["X"].min()
        x_max = df["X"].max()
        x_range = x_max - x_min
        x_mid1 = x_min + x_range / 3
        x_mid2 = x_min + 2 * x_range / 3
        thirds_bounds = (x_min, x_mid1, x_mid2)
    
    # Compute thirds
    df = compute_third_flags(df, attacking_positive_x, thirds_bounds, x_max)
    
    # Project normalized movement onto attack vector
    if attack_vector is not None:
        # Create a 1-second timestamp floor
        df["ts_floor"] = df["Time"].dt.floor("1s")

        # Aggregate dx, dy per second (ignoring zeros)
        grouped = df.groupby("ts_floor")[["movement_dx", "movement_dy"]].mean()
        movement_matrix = grouped.to_numpy()

        # (Optional) If your dx/dy are in degrees, convert to radians (or skip if already local meters)
        # movement_matrix = np.radians(movement_matrix)

        norms = np.linalg.norm(movement_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        movement_unit = movement_matrix / norms

        # Project to attack vector
        dir_proj = (movement_unit @ attack_vector).flatten()
        is_attack_sec = dir_proj >= 0

        # Map back to full 10Hz sample set
        attack_map = pd.Series(is_attack_sec, index=grouped.index)
        df["is_attack"] = df["ts_floor"].map(attack_map).fillna(False)
        df["is_defense"] = ~df["is_attack"]

        df.drop(columns="ts_floor", inplace=True)

    else:
        # If no attack vector provided, use X coordinate direction
        df["is_attack"] = df["X"] >= 0 if attacking_positive_x else df["X"] < 0
        df["is_defense"] = ~df["is_attack"]

    speed = df["Speed (m/s)"]
    df["acc_long"] = speed.diff() * SAMPLE_RATE_HZ
    df["acc_mag"] = np.sqrt(df["Accl X"]**2 + df["Accl Y"]**2 + df["Accl Z"]**2)
    df["jerk"] = df["acc_mag"].diff() * SAMPLE_RATE_HZ
    df["ang_vel_mag"] = np.sqrt(
        np.radians(df["Gyro Yro X"])**2 +
        np.radians(df["Gyro Y"])**2 +
        np.radians(df["Gyro Z"])**2
    )
    from scipy.signal import medfilt
    df["ang_vel_mag"] = medfilt(df["ang_vel_mag"], kernel_size=5)

    

    df["hsr_flag"] = (speed > HSR_THRESHOLD_MS)
    df["hsr_m"] = df["hsr_flag"] * df["inst_dist_m"]
    df["vha_count_1s"] = (df["acc_mag"] > VHA_THRESHOLD_MS2).rolling(SAMPLE_RATE_HZ, min_periods=1).mean()
    df["avg_jerk_1s"] = df["jerk"].rolling(SAMPLE_RATE_HZ, min_periods=1).mean()

    from scipy.ndimage import label

    # Label contiguous regions where ang_vel_mag > threshold
    turn_flag = df["ang_vel_mag"] > TURN_THRESHOLD_RS
    labels, num_labels = label(turn_flag)

    # Filter out short bursts
    valid_turn_start = np.zeros_like(turn_flag, dtype=int)
    for lbl in range(1, num_labels + 1):
        idx = np.where(labels == lbl)[0]
        if len(idx) >= 30:  # 0.3 sec at 100Hz
            valid_turn_start[idx[0]] = 1  # count only once

    df["turns_per_sec"] = pd.Series(valid_turn_start).rolling(SAMPLE_RATE_HZ, min_periods=1).mean()


    pl = np.sqrt(df["Accl X"].diff()**2 + df["Accl Y"].diff()**2 + df["Accl Z"].diff()**2)
    df["playerload_1s"] = pl.rolling(SAMPLE_RATE_HZ, min_periods=1).mean()


    # --- Sprint detection ---
    df["sprint_flag"] = speed >= SPRINT_THRESHOLD
    df["sprint_start"] = (df["sprint_flag"] & (~df["sprint_flag"].shift(1, fill_value=False)))
    df["sprint_id"] = df["sprint_start"].cumsum()

    sprint_lengths = df.groupby("sprint_id")["sprint_flag"].sum()
    valid_sprint_ids = sprint_lengths[sprint_lengths >= 30].index
    df["sprint_label"] = df["sprint_id"].isin(valid_sprint_ids).astype(int)

    valid_sprint_df = df[df["sprint_label"] == 1]

    def classify_direction(group):
        return "attack" if group["is_attack"].mean() >= 0.5 else "defense"

    sprint_directions = valid_sprint_df.groupby("sprint_id", observed=True).apply(classify_direction)
    attack_ids = sprint_directions[sprint_directions == "attack"].index
    defense_ids = sprint_directions[sprint_directions == "defense"].index

    df["total_sprints"] = 0
    df["sprint_attack"] = 0
    df["sprint_defense"] = 0

    start_rows = valid_sprint_df.groupby("sprint_id").head(1).index
    df.loc[start_rows, "total_sprints"] = 1
    attack_starts = valid_sprint_df[valid_sprint_df["sprint_id"].isin(attack_ids)].groupby("sprint_id").head(1).index
    defense_starts = valid_sprint_df[valid_sprint_df["sprint_id"].isin(defense_ids)].groupby("sprint_id").head(1).index
    df.loc[attack_starts, "sprint_attack"] = 1
    df.loc[defense_starts, "sprint_defense"] = 1

    # Calculate time spent in each speed zone - normalize to sum to 60 seconds per minute
    frame_duration = 1 / SAMPLE_RATE_HZ
    
    # First create the speed zone flags
    df["is_walking"] = ((speed >= 0) & (speed < 2)).astype(float)
    df["is_jogging"] = ((speed >= 2) & (speed < 4)).astype(float)
    df["is_running"] = ((speed >= 4) & (speed < 7)).astype(float)
    df["is_sprinting"] = (speed >= SPRINT_THRESHOLD).astype(float)
    
    # Calculate frames per minute for normalization
    df["frames_in_minute"] = df.groupby("minute")["is_walking"].transform("count")
    df["time_per_frame"] = 60 / df["frames_in_minute"]

    # Calculate normalized times
    df["walk_time"] = df["is_walking"] * df["time_per_frame"]
    df["jog_time"] = df["is_jogging"] * df["time_per_frame"]
    df["run_time"] = df["is_running"] * df["time_per_frame"]
    df["sprint_time"] = df["is_sprinting"] * df["time_per_frame"]

    # Calculate attack/defense metrics based on position
    df["dist_attack"] = df["inst_dist_m"] * df["is_attack"]
    df["dist_defense"] = df["inst_dist_m"] * df["is_defense"]
    
    # Calculate attack/defense times using the same normalization
    df["time_attack"] = df["is_attack"] * df["time_per_frame"]
    df["time_defense"] = df["is_defense"] * df["time_per_frame"] 


    #positive and negative acc
    df["pos_acc_attack"] = ((df["acc_long"] > VHA_THRESHOLD_MS2) & df["is_attack"]).cumsum()
    df["neg_acc_attack"] = ((df["acc_long"] < -VHA_THRESHOLD_MS2) & df["is_attack"]).cumsum()
    df["accel_decel_balance_attack"] = df["pos_acc_attack"] - df["neg_acc_attack"]

    df["pos_acc_defense"] = ((df["acc_long"] > VHA_THRESHOLD_MS2) & df["is_defense"]).cumsum()
    df["neg_acc_defense"] = ((df["acc_long"] < -VHA_THRESHOLD_MS2) & df["is_defense"]).cumsum()
    df["accel_decel_balance_defense"] = df["pos_acc_defense"] - df["neg_acc_defense"]

   
    # Create flags for each third
    df["is_attacking_third"] = (df["third"] == "attacking").astype(float)
    df["is_middle_third"] = (df["third"] == "middle").astype(float)
    df["is_defending_third"] = (df["third"] == "defending").astype(float)

    # Use the same normalization as for speed zones
    df["attacking_third_time"] = df["is_attacking_third"] * df["time_per_frame"]
    df["middle_third_time"] = df["is_middle_third"] * df["time_per_frame"]
    df["defending_third_time"] = df["is_defending_third"] * df["time_per_frame"]
    
    #if want to see the field thirds visualization of a player uncomment the line below
    # plot_field_thirds(df, title="First Half: Field Thirds Visualization", thirds_bounds=thirds_bounds, attacking_positive_x=attacking_positive_x)
    
    #if want to see the heatmap of a player uncomment the lines below
    # if heatmap_path:
    #     if attacking_positive_x:
    #         plot_one_half_heatmap_from_xy(df, save_path=heatmap_path)
    #     else:
    #         plot_one_half_heatmap_from_xy(df, save_path=heatmap_path,flip=True)
    minute_stats = df.groupby("minute").agg({
        "inst_dist_m": "sum",
        "Speed (m/s)": ["mean", "max"],
        "hsr_m": "sum",
        "vha_count_1s": "sum",
        "avg_jerk_1s": "mean",
        "turns_per_sec": "sum",
        "playerload_1s": "sum",
        "walk_time": "sum",
        "jog_time": "sum",
        "run_time": "sum",
        "sprint_time": "sum",
        "accel_decel_balance_attack": "last",
        "accel_decel_balance_defense": "last",
        "total_sprints": "sum",
        "sprint_attack": "sum",
        "sprint_defense": "sum",
        "dist_attack": "sum",
        "dist_defense": "sum",
        "time_attack": "sum",
        "time_defense": "sum",
        "attacking_third_time": "sum",
        "middle_third_time": "sum",
        "defending_third_time": "sum",
    }).reset_index()

    # Fix column names - handle both single and multi-level columns
    if isinstance(minute_stats.columns, pd.MultiIndex):
        # For multi-level columns (from agg with lists)
        minute_stats.columns = ["minute" if col[0] == "minute" else f"{col[0]}_{col[1]}" for col in minute_stats.columns]
    else:
        # For single-level columns
        minute_stats.columns = [col if col == "minute" else f"{col}_sum" for col in minute_stats.columns]

    # Verify that times sum to 60 for each minute
    minute_stats["speed_time_sum"] = minute_stats[["walk_time_sum", "jog_time_sum", "run_time_sum", "sprint_time_sum"]].sum(axis=1)
    minute_stats["attack_defense_sum"] = minute_stats["time_attack_sum"] + minute_stats["time_defense_sum"]
    
    # Clean up verification columns
    minute_stats = minute_stats.drop(columns=["speed_time_sum", "attack_defense_sum"])

    # Set minute numbers
    minute_stats["minute"] = np.arange(1, len(minute_stats) + 1)

    return minute_stats



def process_player_folder(player_folder: Path, attack_vector: np.ndarray,
                         first_half_start: str,  # e.g., "00:00:00.000"
                         second_half_start: str,
                         center, corners,
                         field_bounds: dict,
                         pca):
    """
    Process player folder handling missing half files gracefully.
    Will process whatever data is available (first half, second half, or both).
    """
    first_file = player_folder / "first_half.csv"
    second_file = player_folder / "second_half.csv"

    print(f"📂 Processing {player_folder.name}")

    features_first = pd.DataFrame()
    features_second = pd.DataFrame()
    positive_x = True if attack_vector[0] >= 0 else False
    # Process first half if exists
    if first_file.exists():
        try:
            df_first = pd.read_csv(first_file)
            if not df_first.empty:
                features_first = process_half(df_first, center, corners, positive_x, 
                                           half_start_time=first_half_start,
                                           attack_vector=attack_vector,
                                           field_bounds=field_bounds, pca=pca,heatmap_path=player_folder / "heatmap.png")
        except Exception as e:
            print(f"⚠️ Error processing first half for {player_folder.name}: {e}")

    # Process second half if exists
    positive_x = not positive_x
    if second_file.exists():
        try:
            df_second = pd.read_csv(second_file)
            if not df_second.empty:
                features_second = process_half(df_second, center, corners, positive_x,
                                            half_start_time=second_half_start,
                                            attack_vector=-attack_vector,
                                            field_bounds=field_bounds, pca=pca)
        except Exception as e:
            print(f"⚠️ Error processing second half for {player_folder.name}: {e}")

    # Save whatever features we have
    if not features_first.empty:
        features_first.to_csv(player_folder / "first_half_features.csv", index=False)
        print(f"✅ Saved first_half_features.csv for {player_folder.name}")

    if not features_second.empty:
        # Adjust second half minutes if we have first half data
        if not features_first.empty:
            last_minute_first = features_first["minute"].iloc[-1]
            features_second["minute"] += last_minute_first
        features_second.to_csv(player_folder / "second_half_features.csv", index=False)
        print(f"✅ Saved second_half_features.csv for {player_folder.name}")
    # Merge whatever features we have
    features_to_merge = []
    if not features_first.empty:
        features_to_merge.append(features_first)
    if not features_second.empty:
        features_to_merge.append(features_second)

    if features_to_merge:
        features_merged = pd.concat(features_to_merge, ignore_index=True)
        features_merged.to_csv(player_folder / "merged_features.csv", index=False)
        print(f"✅ Saved merged_features.csv in {player_folder.name}/")
        if len(features_to_merge) == 1:
            print(f"ℹ️ Note: {player_folder.name} only has {'first' if not features_first.empty else 'second'} half data")
    else:
        print(f"⚠️ No valid data found for {player_folder.name}")


def main(parent_folder: str, cf_csv: str, cb_csv: str, 
         first_half_start: str = "00:00:00.000",
         second_half_start: str = "00:45:00.000"):
    """
    Main processing function with half start times.
    The data for each player will automatically start from when they entered the game.
    """
    # attack_vector = infer_attack_vector_cf(Path(cf_csv))
    attack_vector = infer_attack_vector_only(Path(cf_csv), Path(cb_csv))
    print(attack_vector)
    if attack_vector is None:
        print("❌ Could not determine attacking direction.")
        return
    center, corners, field_bounds = load_field_frame(Path("field_frame.json"))  # Adjust path as needed
    # Calculate PCA transformation once using field corners
    _, pca = gps_to_local_xy(pd.DataFrame(), center, corners)
    # check if the attack vector is reasonable
    pos_x = True if attack_vector[0] >= 0 else False
    df_cf = pd.read_csv(cf_csv)
    df_cb = pd.read_csv(cb_csv)
    stats_first = process_half(df_cf.copy(), center, corners, pos_x,
                                half_start_time=first_half_start,
                                attack_vector=attack_vector,
                                field_bounds=field_bounds, pca=pca)

    total_attack_time = stats_first["attacking_third_time_sum"].sum()
    total_defense_time =stats_first["defending_third_time_sum"].sum()

    print(f"➡️  CF total time: attack={total_attack_time:.1f}s, defense={total_defense_time:.1f}s")

    if total_defense_time > total_attack_time:
        print("🔄 Flipping attack vector (CF spent more time defending)")
        attack_vector *= -1
    # if vector is flipped you can see the striker and the defender thirds spread to see reasonable results
    # print(attack_vector)
    # pos_x = not pos_x
    # process_half(df_cf.copy(), center, corners, pos_x,
    #                             half_start_time=first_half_start,
    #                             attack_vector=attack_vector,
    #                             field_bounds=field_bounds, pca=pca)
    # process_half(df_cb.copy(), center, corners, pos_x,
    #                             half_start_time=first_half_start,
    #                             attack_vector=attack_vector,
    #                             field_bounds=field_bounds, pca=pca)
    

    parent_path = Path(parent_folder)
    player_folders = [f for f in parent_path.iterdir() if f.is_dir()]

    if not player_folders:
        print("❌ No player folders found.")
        return

    for player_folder in player_folders:
        try:
            process_player_folder(player_folder, attack_vector, 
                                first_half_start, second_half_start,
                                center, corners, field_bounds, pca)
        except Exception as e:
            print(f"❌ Error in {player_folder.name}: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python main.py <parent_folder> <cf_csv> <cb_csv> [first_half_start] [second_half_start]")
    else:
        first_start = sys.argv[4] if len(sys.argv) > 4 else "00:00:00.000"
        second_start = sys.argv[5] if len(sys.argv) > 5 else "00:45:00.000"
        main(sys.argv[1], sys.argv[2], sys.argv[3], first_start, second_start)



