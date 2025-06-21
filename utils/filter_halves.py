"""
Script: filter_halves.py

Description:
    This script processes match data by splitting and filtering it into first and second halves.
    It handles the separation of game data into distinct periods and applies necessary
    filtering operations to clean and prepare the data for analysis.

Input:
    - Raw match data files (CSV format)
    - Expected to contain:
        * Time-series data for the entire match
        * Player position and movement data
        * Match event information

Output:
    - Two separate CSV files:
        * first_half.csv: Contains filtered data for the first half
        * second_half.csv: Contains filtered data for the second half
    - Each output file contains cleaned and processed data specific to that half

Usage:
    Run this script to split and filter match data into separate half periods
"""

import pandas as pd
from pathlib import Path
from datetime import timedelta
import argparse

def detect_half_gap(folder_path, gap_threshold_sec=60, alignment_margin_min=2):
    befores, afters = [], []

    for file in Path(folder_path).glob("*.csv"):
        try:
            df = pd.read_csv(file)
            df['Timestamp'] = pd.to_datetime(df['Time'], format='%H:%M:%S.%f', errors='coerce')
            df = df.dropna(subset=['Timestamp']).sort_values('Timestamp')

            time_diff = df['Timestamp'].diff().dt.total_seconds()
            gap_index = time_diff[time_diff > gap_threshold_sec].first_valid_index()

            if gap_index is not None:
                gap_row = df.index.get_loc(gap_index)
                before = df.iloc[gap_row - 1]['Timestamp']
                after = df.iloc[gap_row]['Timestamp']
                befores.append(before)
                afters.append(after)
                print(f"[{file.name}] Gap > {gap_threshold_sec}s: {before.time()} → {after.time()}")
            else:
                print(f"[{file.name}] No gap > {gap_threshold_sec}s found.")
        except Exception as e:
            print(f"[{file.name}] Error: {e}")

    if not befores or not afters:
        print("❌ No valid gaps found in any file.")
        return

    befores = pd.Series(befores)
    afters = pd.Series(afters)

    # Filter by alignment with group median
    margin = timedelta(minutes=alignment_margin_min)
    median_before = befores.median()
    median_after = afters.median()
    valid_mask = (abs(befores - median_before) < margin) & (abs(afters - median_after) < margin)

    filtered_befores = befores[valid_mask]
    filtered_afters = afters[valid_mask]

    if not filtered_befores.empty and not filtered_afters.empty:
        min_before = filtered_befores.min()
        max_after = filtered_afters.max()
        print("\n✅ Cleaned Halftime Split (filtered outliers):")
        print(f"🛑 End of First Half:   {min_before.time()}")
        print(f"▶️  Start of Second Half: {max_after.time()}")
    else:
        print("\n⚠️ All detected gaps were considered outliers.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect halftime gap across player GPS files.")
    parser.add_argument("folder", help="Path to folder containing player CSV files")
    parser.add_argument("--gap", type=int, default=60, help="Gap threshold in seconds (default: 60)")
    parser.add_argument("--margin", type=int, default=2, help="Allowed time margin in minutes for alignment (default: 2)")
    args = parser.parse_args()

    detect_half_gap(args.folder, gap_threshold_sec=args.gap, alignment_margin_min=args.margin)


