from pathlib import Path
"""
Script: find_ten_runners.py

Description:
    This script analyzes GPS tracking data to identify the top runners (players who covered
    the most distance) during a specified time window. It processes player movement data
    with noise reduction and distance calculations using the Haversine formula.
    We use it to identify the starting ten players of the game (without GK).

Key Features:
    - Processes GPS data sampled at 10Hz (10 samples per second)
    - Applies median filtering to reduce GPS jitter (0.5s window)
    - Calculates actual distance traveled using the Haversine formula
    - Filters out small movements below 0.2 meters to reduce noise
    - Ranks players by total distance covered in the specified time window

Parameters:
    - SAMPLE_RATE_HZ: 100 (data collection frequency)
    - MEDFILT_SEC: 0.5 (smoothing window duration)
    - JITTER_METRES: 0.2 (minimum movement threshold)
    - EARTH_R: 6,371,008.8 (Earth's radius in meters)

Input:
    - Folder containing CSV files with player GPS data
    - Each CSV must contain: Time, Lat, Lon, Speed (m/s)
    - Start time for the analysis window
    - Optional window duration (default: 20 minutes)

Output:
    - List of top runners sorted by distance covered
    - Each player's total distance in meters

Usage:
    python find_ten_runners.py <folder_path> <start_time>
    Example: python find_ten_runners.py ./match_data "18:30:00.000"
"""

import numpy as np
import pandas as pd
from scipy.signal import medfilt
from datetime import datetime

# Parameters
SAMPLE_RATE_HZ = 100
MEDFILT_SEC = 0.5
MEDFILT_K = int(round(MEDFILT_SEC * SAMPLE_RATE_HZ)) | 1
JITTER_METRES = 0.1
EARTH_R = 6_371_008.8  # Earth radius in meters

def to_numeric(series):
    return pd.to_numeric(series.astype(str)
                         .str.replace(',', '.')
                         .str.strip()
                         .replace({'': np.nan}),
                         errors='coerce')

def load_window(csv_file: Path, start_time_str: str, window_min: int = 20) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_file, usecols=['Time', 'Lat', 'Lon', 'Speed (m/s)'],
                     dtype=str, keep_default_na=False)

    ts = pd.to_datetime(df['Time'], format='%H:%M:%S.%f', errors='coerce')
    df = df.assign(ts=ts)
    df['Lat'] = to_numeric(df['Lat'])
    df['Lon'] = to_numeric(df['Lon'])
    df = df.dropna(subset=['ts', 'Lat', 'Lon']).sort_values('ts')
    if df.empty:
        return np.array([]), np.array([])

    start_time = pd.to_datetime(start_time_str, format='%H:%M:%S.%f')
    df = df[(df['ts'] >= start_time) & (df['ts'] <= start_time + pd.Timedelta(minutes=window_min))]

    lat = medfilt(df['Lat'].to_numpy(), MEDFILT_K)
    lon = medfilt(df['Lon'].to_numpy(), MEDFILT_K)
    return lat, lon

def haversine_distance(lat, lon):
    if len(lat) < 2:
        return 0.0
    phi = np.radians(lat)
    lam = np.radians(lon)
    dphi = np.diff(phi)
    dlam = np.diff(lam)
    a = np.sin(dphi/2)**2 + np.cos(phi[:-1]) * np.cos(phi[1:]) * np.sin(dlam/2)**2
    steps = 2 * EARTH_R * np.arcsin(np.sqrt(a))
    return steps[steps >= JITTER_METRES].sum()

def top_runners(folder: Path, start_time_str: str, top_k: int = 10) -> list[str]:
    records = []
    for csv_file in folder.glob("*.csv"):
        lat, lon = load_window(csv_file, start_time_str)
        dist = haversine_distance(lat, lon)
        player_name = csv_file.stem
        records.append({"player": player_name, "distance_m": dist})

    if not records:
        print("❌ No valid player data found.")
        return []

    df = (pd.DataFrame(records)
          .sort_values('distance_m', ascending=False)
          .reset_index(drop=True))
    top_players = df.head(top_k)['player'].tolist()
    return top_players

# Example usage
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Get top runners from GPS data.")
    parser.add_argument("folder", type=str, help="Path to folder containing one CSV per player.")
    parser.add_argument("start_time", type=str, help="Start time (e.g., '00:00:00.000')")
    args = parser.parse_args()

    result = top_runners(Path(args.folder), args.start_time)
    print("🏃 Top runners:", result)

