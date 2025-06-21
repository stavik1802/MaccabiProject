# ==================== attacking_direction.py ====================
"""
This module analyzes player movement data to determine the attacking direction in a game.
It processes GPS coordinates from two players (typically a center forward and center back)
to infer the direction of attack using Principal Component Analysis (PCA).

Key functions:
- load_player_coords: Loads and smooths player GPS coordinates from CSV files
- infer_attack_vector_only: Determines the attacking direction vector using PCA on player movements

The module expects CSV files containing timestamped GPS coordinates (Latitude and Longitude)
and uses median filtering to smooth the movement data. The attacking direction is inferred
by analyzing the relative movement patterns of the two players.

Usage:
    python attacking_direction.py <cf_csv_path> <cb_csv_path>
"""

#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from scipy.signal import medfilt

def load_player_coords(csv_file: Path, smooth_k: int = 5):
    df = pd.read_csv(csv_file)
    df['Timestamp'] = pd.to_datetime(df['Time'], format='%H:%M:%S.%f', errors='coerce')
    df = df.dropna(subset=['Timestamp', 'Lat', 'Lon']).sort_values('Timestamp')

    if len(df) < 20:
        return None, None

    df['Lat'] = medfilt(df['Lat'], kernel_size=smooth_k)
    df['Lon'] = medfilt(df['Lon'], kernel_size=smooth_k)

    coords = np.column_stack([df['Lon'].to_numpy(), df['Lat'].to_numpy()])
    return coords, df['Timestamp'].to_numpy()

def infer_attack_vector_only(cf_file: Path, cb_file: Path, sample_rate=100):
    coords_cf, _ = load_player_coords(cf_file)
    coords_cb, _ = load_player_coords(cb_file)

    if coords_cf is None or coords_cb is None:
        print("⚠️ Insufficient data for one or both players.")
        return None

    min_len = min(len(coords_cf), len(coords_cb))
    coords = np.concatenate([coords_cf[:min_len], coords_cb[:min_len]])
    pca = PCA(n_components=1)
    projected = pca.fit_transform(coords).flatten()
    direction_vector = pca.components_[0]

    n = 5 * sample_rate
    start_mean = projected[:n].mean()
    end_mean = projected[-n:].mean()
    #check if the direction vector is in the right direction
    direction_vector_x = direction_vector[0]
    direction_vector_y = direction_vector[1]
    direction_vector[0] = direction_vector_y
    direction_vector[1] = direction_vector_x
    # if (end_mean <= start_mean):
    #     print("⚠️ Attacking direction vector is in the wrong direction.")
    return direction_vector * (1.0 if end_mean > start_mean else -1.0)

def infer_attack_vector_cf(cf_file: Path, sample_rate=100):
    coords_cf, _ = load_player_coords(cf_file)
    if coords_cf is None or len(coords_cf) < 10:
        print("⚠️ Insufficient data for CF.")
        return None

    # Displacement vectors (velocity direction)
    displacements = coords_cf[1:] - coords_cf[:-1]
    pca = PCA(n_components=1).fit(displacements)
    direction_vector = pca.components_[0]

    # Align direction to actual movement
    net_movement = coords_cf[-1] - coords_cf[0]
    if np.dot(net_movement, direction_vector) < 0:
        direction_vector *= -1

    return direction_vector

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Infer attacking direction vector only.")
    parser.add_argument("cf_csv", type=str, help="Path to CF CSV file")
    parser.add_argument("cb_csv", type=str, help="Path to CB/defender CSV file")
    args = parser.parse_args()

    cf_file = Path(args.cf_csv)
    cb_file = Path(args.cb_csv)

    direction_vector = infer_attack_vector_only(cf_file, cb_file)
    if direction_vector is not None:
        print("[INFO] Attacking direction vector:", direction_vector)
