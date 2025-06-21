import pandas as pd
from pathlib import Path
import argparse
from datetime import timedelta

def get_gap_based_halftime_split(folder_path, gap_threshold_sec=60, alignment_margin_min=2):
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
        except Exception:
            continue

    if not befores or not afters:
        return None, None

    befores = pd.Series(befores)
    afters = pd.Series(afters)

    margin = timedelta(minutes=alignment_margin_min)
    median_before = befores.median()
    median_after = afters.median()
    valid_mask = (abs(befores - median_before) < margin) & (abs(afters - median_after) < margin)

    filtered_befores = befores[valid_mask]
    filtered_afters = afters[valid_mask]

    if not filtered_befores.empty and not filtered_afters.empty:
        return filtered_befores.min(), filtered_afters.max()
    else:
        return None, None

def find_half_boundaries(folder_path, up_threshold=2.0, down_threshold=1.5, gap_threshold=60, margin=2):
    player_speeds = []

    for file in Path(folder_path).glob("*.csv"):
        try:
            df = pd.read_csv(file)
            df['Timestamp'] = pd.to_datetime(df['Time'], format='%H:%M:%S.%f', errors='coerce')
            df = df.dropna(subset=['Timestamp'])

            df_speed = df[['Timestamp', 'Speed (m/s)']].dropna()
            df_speed = df_speed.drop_duplicates(subset='Timestamp')
            df_speed = df_speed.set_index('Timestamp')
            df_speed = df_speed.rename(columns={'Speed (m/s)': file.stem})

            player_speeds.append(df_speed)
        except Exception as e:
            print(f"⚠️ Skipping {file.name} due to error: {e}")

    if not player_speeds:
        print("❌ No valid player data found.")
        return

    combined_df = pd.concat(player_speeds, axis=1).sort_index()
    combined_df['mean_speed'] = combined_df.mean(axis=1, skipna=True)

    mean_speed_10s = combined_df['mean_speed'].resample('10s').mean()
    diff_series = mean_speed_10s.diff().round(4)

    # First Half Start
    fh_start = diff_series[diff_series >= up_threshold].first_valid_index()
    if fh_start is None:
        print("❌ First half start not found.")
        return

    print(f"✅ First half start at: {fh_start}")

    # Try to detect halftime using gaps
    print("\n🔍 Trying to detect halftime from data gaps...")
    fh_end_gap, sh_start_gap = get_gap_based_halftime_split(folder_path, gap_threshold_sec=gap_threshold, alignment_margin_min=margin)

    if fh_end_gap and sh_start_gap:
        fh_end = fh_end_gap
        sh_start = sh_start_gap
        print("✅ Using gap-based halftime detection:")
        print(f"🛑 First half end: {fh_end}")
        print(f"▶️  Second half start: {sh_start}")
    else:
        print("⚠️ No consistent halftime gap found. Falling back to speed-based detection.")
        # First Half End: look at least 45 minutes after start
        min_fh_end_time = fh_start + pd.Timedelta(minutes=45)
        fh_end = diff_series.loc[min_fh_end_time:][diff_series <= -down_threshold].first_valid_index()
        if fh_end is None:
            print("❌ First half end not found.")
            return
        print(f"🛑 First half end at: {fh_end}")

        # Second Half Start: at least 10 minutes after end
        min_sh_start_time = fh_end + pd.Timedelta(minutes=10)
        sh_start = diff_series.loc[min_sh_start_time:][diff_series >= up_threshold].first_valid_index()
        if sh_start is None:
            print("❌ Second half start not found.")
            return
        print(f"▶️  Second half start at: {sh_start}")

    # Second Half End: look at least 45 minutes after second half start
    min_sh_end_time = sh_start + pd.Timedelta(minutes=45)
    sh_end = diff_series.loc[min_sh_end_time:][diff_series <= -down_threshold].first_valid_index()
    if sh_end is None:
        print("❌ Second half end not found.")
        return

    print(f"⏹️  Second half end at: {sh_end}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect start and end times of game halves.")
    parser.add_argument("folder", help="Folder with player CSV files")
    parser.add_argument("--up_threshold", type=float, default=2.0, help="Speed increase threshold (default: 2.0 m/s)")
    parser.add_argument("--down_threshold", type=float, default=0.8, help="Speed drop threshold (default: 2.0 m/s)")
    parser.add_argument("--gap", type=int, default=60, help="Gap threshold in seconds (default: 60)")
    parser.add_argument("--margin", type=int, default=2, help="Time margin for halftime alignment (minutes)")
    args = parser.parse_args()

    find_half_boundaries(
        folder_path=args.folder,
        up_threshold=args.up_threshold,
        down_threshold=args.down_threshold,
        gap_threshold=args.gap,
        margin=args.margin
    )



# import pandas as pd
# from pathlib import Path
# import argparse
#
# def find_first_speed_jump_window(folder_path, speed_diff_threshold=2.0):
#     player_speeds = []
#
#     for file in Path(folder_path).glob("*.csv"):
#         try:
#             df = pd.read_csv(file)
#             df['Timestamp'] = pd.to_datetime(df['Time'], format='%H:%M:%S.%f', errors='coerce')
#             df = df.dropna(subset=['Timestamp'])
#
#             df_speed = df[['Timestamp', 'Speed (m/s)']].dropna()
#             df_speed = df_speed.drop_duplicates(subset='Timestamp')
#             df_speed = df_speed.set_index('Timestamp')
#             df_speed = df_speed.rename(columns={'Speed (m/s)': file.stem})
#
#             player_speeds.append(df_speed)
#         except Exception as e:
#             print(f"⚠️ Skipping {file.name} due to error: {e}")
#
#     if not player_speeds:
#         print("❌ No valid player data found.")
#         return
#
#     # Combine all player speed data on timestamps
#     combined_df = pd.concat(player_speeds, axis=1).sort_index()
#
#     # Compute mean speed across players, ignoring NaNs
#     combined_df['mean_speed'] = combined_df.mean(axis=1, skipna=True)
#
#     # Resample into 10-second windows, compute average mean speed in each
#     mean_speed_10s = combined_df['mean_speed'].resample('10s').mean()
#
#     # Compute difference between consecutive 10s windows
#     diff_series = mean_speed_10s.diff()
#
#     # Find first window where increase exceeds threshold
#     spike_idx = diff_series[diff_series > speed_diff_threshold].first_valid_index()
#
#     if spike_idx is None:
#         print(f"❌ No 10-second window found with mean speed increase > {speed_diff_threshold} m/s.")
#     else:
#         print(f"✅ First window with > {speed_diff_threshold} m/s jump:")
#         print(f"   Start time: {spike_idx.time()} on {spike_idx.date()}")
#         print(f"   Speed increase: {diff_series[spike_idx]:.2f} m/s")
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Find first 10-second window with speed increase above a threshold.")
#     parser.add_argument("folder", help="Folder containing player CSV files")
#     parser.add_argument("--threshold", type=float, default=2.0, help="Speed increase threshold in m/s (default: 1.0)")
#     args = parser.parse_args()
#
#     find_first_speed_jump_window(args.folder, speed_diff_threshold=args.threshold)


# import pandas as pd
# from pathlib import Path
# import argparse
#
# def find_first_high_speed_sample(folder_path, speed_threshold=2.0):
#     player_speeds = []
#
#     for file in Path(folder_path).glob("*.csv"):
#         try:
#             df = pd.read_csv(file)
#             df['Timestamp'] = pd.to_datetime(df['Time'], format='%H:%M:%S.%f', errors='coerce')
#             df = df.dropna(subset=['Timestamp'])
#
#             df_speed = df[['Timestamp', 'Speed (m/s)']].dropna()
#             df_speed = df_speed.drop_duplicates(subset='Timestamp')
#             df_speed = df_speed.set_index('Timestamp')
#             df_speed = df_speed.rename(columns={'Speed (m/s)': file.stem})
#
#             player_speeds.append(df_speed)
#         except Exception as e:
#             print(f"⚠️ Skipping {file.name} due to error: {e}")
#
#     if not player_speeds:
#         print("❌ No valid player data found.")
#         return
#
#     # Combine all player speed data on timestamps
#     combined_df = pd.concat(player_speeds, axis=1).sort_index()
#
#     # Compute mean speed across available players at each timestamp (skip NaNs)
#     combined_df['mean_speed'] = combined_df.mean(axis=1, skipna=True)
#
#     # Find the first timestamp with mean speed above threshold
#     high_speed_df = combined_df[combined_df['mean_speed'] > speed_threshold]
#     if high_speed_df.empty:
#         print(f"️ No sample found with mean speed above {speed_threshold} m/s.")
#     else:
#         first_time = high_speed_df.index[0]
#         print(f" First high-speed sample > {speed_threshold} m/s:")
#         print(f" Timestamp: {first_time.time()} on synthetic date {first_time.date()}")
#         print(f" Mean speed: {high_speed_df.iloc[0]['mean_speed']:.2f} m/s")
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Find the first sample where mean speed across players exceeds a threshold.")
#     parser.add_argument("folder", help="Folder containing player CSV files")
#     parser.add_argument("--speed", type=float, default=2.0, help="Speed threshold in m/s (default: 1.5)")
#     args = parser.parse_args()
#
#     find_first_high_speed_sample(args.folder, speed_threshold=args.speed)


