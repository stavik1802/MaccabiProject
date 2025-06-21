"""
Script: update_data.py

Description:
    This script handles the updating and maintenance of the player tracking database.
    It processes new data files, updates existing records, and ensures data consistency
    across the system.

Input:
    - New data files to be integrated
    - Existing database or data files
    - Configuration parameters for data update process

Output:
    - Updated database/files with new information integrated
    - Log of changes and updates made
    - Verification report of successful updates

Usage:
    Run this script when new tracking data needs to be integrated into the system
"""

import math
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from pathlib import Path
import argparse


SAMPLE_RATE_HZ = 100
ROLL_1S = SAMPLE_RATE_HZ
ROLL_1M = 60 * SAMPLE_RATE_HZ
ROLL_5M = 5 * 60 * SAMPLE_RATE_HZ


def process_basic_metrics(csv_file: Path, out_dir: Path):
    df = pd.read_csv(csv_file)
    df = df[(df["Lat"].abs() > 1) & (df["Lon"].abs() > 1)].reset_index(drop=True)

    if df["Lat"].abs().max() > 360:
        df[["Lat", "Lon"]] *= 1e-6

    df["t_sec"] = np.arange(len(df)) / SAMPLE_RATE_HZ

    coords = list(zip(df["Lat"], df["Lon"]))
    inst_dist = [0.0]
    for i in range(1, len(coords)):
        dist = geodesic(coords[i - 1], coords[i]).meters
        inst_dist.append(dist)
    df["inst_dist_m"] = inst_dist

    df["cum_dist_m"] = df["inst_dist_m"].cumsum()

    df["mean_speed_1s"] = df["Speed (m/s)"].rolling(ROLL_1S, min_periods=1).mean()
    df["max_speed_1min"] = df["Speed (m/s)"].rolling(ROLL_1M, min_periods=1).max()
    df["dist_m_5min"] = df["inst_dist_m"].rolling(ROLL_5M, min_periods=1).sum()

    out_base = out_dir / f"basic_metrics_{csv_file.stem}"
    try:
        import pyarrow  # noqa
        out_file = out_base.with_suffix(".parquet")
        df.to_parquet(out_file, index=False, compression="zstd")
        print(f"✅ {csv_file.name} → {out_file.name} [Parquet]")
    except ImportError:
        out_file = out_base.with_suffix(".csv")
        df.to_csv(out_file, index=False)
        print(f"✅ {csv_file.name} → {out_file.name} [CSV]")

def process_match_basic_metrics(game_folder: Path):
    processed_dir = game_folder / "basic_metrics"
    processed_dir.mkdir(exist_ok=True)
    
    # Process each player's data in the game folder
    for player_file in game_folder.glob("*.csv"):
        if player_file.name != "filtered_subs.csv":  # Skip the filtered subs file
            try:
                process_basic_metrics(player_file, processed_dir)
            except Exception as e:
                print(f"❌ Error processing {player_file.name}: {e}")
    return processed_dir


def main():
    parser = argparse.ArgumentParser(description="Extract speed-only features from GPS data.")
    parser.add_argument("input_folder", type=str, help="Folder with input CSV files")
    parser.add_argument("output_folder", type=str, help="Folder to save output feature files")
    args = parser.parse_args()

    in_dir = Path(args.input_folder)
    out_dir = Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    for file in in_dir.glob("*.csv"):
        try:
            process_basic_metrics(file, out_dir)
        except Exception as e:
            print(f"❌ Failed to process {file.name}: {e}")


if __name__ == "__main__":
    main()

