"""Compute team possession over time from event logs; utilities for cumulative possession computation."""
# import pandas as pd
# import argparse

# def compute_cumulative_possession(input_csv_path, output_csv_path, team_name="Maccabi Haifa",
#                                    interval=15, support_window=10, required_support=10):
#     df = pd.read_csv(input_csv_path)
#     df = df.sort_values(by='gameClock').reset_index(drop=True)

#     # Check for required columns
#     if 'Team' not in df.columns or 'gameClock' not in df.columns:
#         raise ValueError("CSV must include 'Team' and 'gameClock' columns.")

#     possession_timeline = []

#     # Step 1: Assign from time 0 to first event
#     first_time = int(df.loc[0, 'gameClock'])
#     current_possession_team = df.loc[0, 'Team']
#     possession_timeline.extend([current_possession_team] * first_time)

#     # Step 2: Loop through events and decide possession
#     for i in range(len(df) - 1):
#         start_time = int(df.loc[i, 'gameClock'])
#         end_time = int(df.loc[i + 1, 'gameClock'])
#         candidate_team = df.loc[i, 'Team']

#         if candidate_team != current_possession_team:
#             # Check next `support_window` events for support
#             future_teams = df['Team'].iloc[i + 1:i + 1 + support_window]
#             support_count = (future_teams == candidate_team).sum()

#             if support_count >= required_support:
#                 current_possession_team = candidate_team
#             # else: possession does not change

#         # Extend possession timeline
#         possession_timeline.extend([current_possession_team] * (end_time - start_time))

#     # Step 3: From last event to end of match
#     last_time = int(df['gameClock'].max())
#     last_event_time = int(df.iloc[-1]['gameClock'])
#     possession_timeline.extend([current_possession_team] * (last_time - last_event_time + 1))

#     # Step 4: Build timeline DataFrame
#     timeline_df = pd.DataFrame({
#         'second': list(range(len(possession_timeline))),
#         'team': possession_timeline
#     })

#     timeline_df['is_team'] = (timeline_df['team'] == team_name).astype(int)
#     timeline_df['cumulative_possession'] = timeline_df['is_team'].expanding().mean() * 100

#     # Step 5: Sample every `interval` seconds
#     sampled_df = timeline_df[timeline_df['second'] % interval == 0][['second', 'cumulative_possession']]
#     sampled_df.rename(columns={
#         'second': 'time_sec',
#         'cumulative_possession': f'{team_name.lower().replace(" ", "_")}_cumulative_possession_percent'
#     }, inplace=True)

#     # Save
#     sampled_df.to_csv(output_csv_path, index=False)
#     print(f"✅ Saved cumulative possession to {output_csv_path}")
#     final = timeline_df['is_team'].mean() * 100
#     print(f"📊 Final possession for '{team_name}': {final:.2f}%")

# def main():
#     parser = argparse.ArgumentParser(description="Compute cumulative possession percentage over time.")
#     parser.add_argument("input_csv", help="Path to input CSV file")
#     parser.add_argument("output_csv", help="Path to output CSV")
#     parser.add_argument("--team", default="Maccabi Haifa", help="Team to compute possession for (default: Maccabi Haifa)")
#     parser.add_argument("--interval", type=int, default=15, help="Sampling interval in seconds (default: 15)")
#     parser.add_argument("--support_window", type=int, default=5, help="Number of future events to check for possession switch (default: 2)")
#     parser.add_argument("--required_support", type=int, default=4, help="Minimum events from candidate team to confirm switch (default: 2)")
#     args = parser.parse_args()

#     compute_cumulative_possession(
#         args.input_csv,
#         args.output_csv,
#         team_name=args.team,
#         interval=args.interval,
#         support_window=args.support_window,
#         required_support=args.required_support
#     )

# if __name__ == "__main__":
#     main()


# import os
# import pandas as pd
# import argparse
# from pathlib import Path

# def compute_cumulative_possession(input_csv_path, output_csv_path, team_name="Maccabi Haifa",
#                                    interval=15, support_window=2, required_support=2):
#     df = pd.read_csv(input_csv_path)
#     df = df.sort_values8(by='gameClock').reset_index(drop=True)

#     if 'Team' not in df.columns or 'gameClock' not in df.columns:
#         raise ValueError(f"{input_csv_path} must include 'Team' and 'gameClock' columns.")

#     possession_timeline = []
#     first_time = int(df.loc[0, 'gameClock'])
#     current_possession_team = df.loc[0, 'Team']
#     possession_timeline.extend([current_possession_team] * first_time)

#     for i in range(len(df) - 1):
#         start_time = int(df.loc[i, 'gameClock'])
#         end_time = int(df.loc[i + 1, 'gameClock'])
#         candidate_team = df.loc[i, 'Team']

#         if candidate_team != current_possession_team:
#             future_teams = df['Team'].iloc[i + 1:i + 1 + support_window]
#             support_count = (future_teams == candidate_team).sum()
#             if support_count >= required_support:
#                 current_possession_team = candidate_team

#         possession_timeline.extend([current_possession_team] * (end_time - start_time))

#     last_time = int(df['gameClock'].max())
#     last_event_time = int(df.iloc[-1]['gameClock'])
#     possession_timeline.extend([current_possession_team] * (last_time - last_event_time + 1))

#     timeline_df = pd.DataFrame({
#         'second': list(range(len(possession_timeline))),
#         'team': possession_timeline
#     })

#     timeline_df['is_team'] = (timeline_df['team'] == team_name).astype(int)
#     timeline_df['cumulative_possession'] = timeline_df['is_team'].expanding().mean() * 100

#     sampled_df = timeline_df[timeline_df['second'] % interval == 0][['second', 'cumulative_possession']]
#     sampled_df.rename(columns={
#         'second': 'time_sec',
#         'cumulative_possession': f'{team_name.lower().replace(" ", "_")}_cumulative_possession_percent'
#     }, inplace=True)

#     sampled_df.to_csv(output_csv_path, index=False)

#     final_possession = timeline_df['is_team'].mean() * 100
#     print(f"✅ {os.path.basename(input_csv_path)} → {final_possession:.2f}% possession")
#     return final_possession

# def process_folder(input_folder, team_name, interval, support_window, required_support):
#     input_folder = Path(input_folder)
#     output_folder = input_folder.parent / "games_possession"
#     output_folder.mkdir(exist_ok=True)

#     for file in input_folder.glob("*.csv"):
#         output_csv_path = output_folder / file.name
#         try:
#             compute_cumulative_possession(
#                 file,
#                 output_csv_path,
#                 team_name=team_name,
#                 interval=interval,
#                 support_window=support_window,
#                 required_support=required_support
#             )
#         except Exception as e:
#             print(f"❌ Failed to process {file.name}: {e}")

# def main():
#     parser = argparse.ArgumentParser(description="Batch compute cumulative possession for all game CSVs in a folder.")
#     parser.add_argument("input_folder", help="Folder containing CSV files for each game")
#     parser.add_argument("--team", default="Maccabi Haifa", help="Team to compute possession for")
#     parser.add_argument("--interval", type=int, default=15, help="Sampling interval in seconds")
#     parser.add_argument("--support_window", type=int, default=4, help="Number of future events to check for possession switch")
#     parser.add_argument("--required_support", type=int, default=3, help="Minimum matches in support window to switch possession")
#     args = parser.parse_args()

#     process_folder(
#         args.input_folder,
#         team_name=args.team,
#         interval=args.interval,
#         support_window=args.support_window,
#         required_support=args.required_support
#     )

# if __name__ == "__main__":
#     main()


import os
import pandas as pd
import argparse
from pathlib import Path

def compute_window_possession(input_csv_path, output_csv_path, team_name="Maccabi Haifa",
                              window_size=120, support_window=2, required_support=2):
    df = pd.read_csv(input_csv_path)
    df = df.sort_values(by='gameClock').reset_index(drop=True)

    if 'Team' not in df.columns or 'gameClock' not in df.columns:
        raise ValueError(f"{input_csv_path} must include 'Team' and 'gameClock' columns.")

    possession_timeline = []
    first_time = int(df.loc[0, 'gameClock'])
    current_possession_team = df.loc[0, 'Team']
    possession_timeline.extend([current_possession_team] * first_time)

    # Walk through events and fill second-by-second timeline
    for i in range(len(df) - 1):
        start_time = int(df.loc[i, 'gameClock'])
        end_time = int(df.loc[i + 1, 'gameClock'])
        candidate_team = df.loc[i, 'Team']

        # Check if possession changes
        if candidate_team != current_possession_team:
            future_teams = df['Team'].iloc[i + 1:i + 1 + support_window]
            support_count = (future_teams == candidate_team).sum()
            if support_count >= required_support:
                current_possession_team = candidate_team

        possession_timeline.extend([current_possession_team] * (end_time - start_time))

    last_time = int(df['gameClock'].max())
    last_event_time = int(df.iloc[-1]['gameClock'])
    possession_timeline.extend([current_possession_team] * (last_time - last_event_time + 1))

    timeline_df = pd.DataFrame({
        'second': list(range(len(possession_timeline))),
        'team': possession_timeline
    })
    timeline_df['is_team'] = (timeline_df['team'] == team_name).astype(int)

    # ✅ Group by window of 2 minutes
    timeline_df['window'] = (timeline_df['second'] // window_size)  # 0,1,2...

    window_possession = timeline_df.groupby('window')['is_team'].mean() * 100
    window_possession = window_possession.reset_index()
    window_possession['time_start_sec'] = window_possession['window'] * window_size
    window_possession['time_end_sec'] = (window_possession['window'] + 1) * window_size

    window_possession.rename(columns={'is_team': f'{team_name.lower().replace(" ", "_")}_possession_percent'},
                             inplace=True)

    # Save to CSV
    window_possession[['time_start_sec', 'time_end_sec', f'{team_name.lower().replace(" ", "_")}_possession_percent']].to_csv(output_csv_path, index=False)

    print(f"✅ {os.path.basename(input_csv_path)} processed. Wrote possession per {window_size//60} min window.")
    return window_possession

def process_folder(input_folder, team_name, window_size, support_window, required_support):
    input_folder = Path(input_folder)
    output_folder = input_folder.parent / "games_possession_1min"
    output_folder.mkdir(exist_ok=True)

    for file in input_folder.glob("*.csv"):
        output_csv_path = output_folder / file.name
        try:
            compute_window_possession(
                file,
                output_csv_path,
                team_name=team_name,
                window_size=window_size,
                support_window=support_window,
                required_support=required_support
            )
        except Exception as e:
            print(f"❌ Failed to process {file.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Compute independent possession per time window.")
    parser.add_argument("input_folder", help="Folder containing CSV files for each game")
    parser.add_argument("--team", default="Maccabi Haifa", help="Team to compute possession for")
    parser.add_argument("--window", type=int, default=300, help="Window size in seconds (default: 2 min)")
    parser.add_argument("--support_window", type=int, default=4, help="Number of future events to confirm switch")
    parser.add_argument("--required_support", type=int, default=3, help="Min matches in support window to switch possession")
    args = parser.parse_args()

    process_folder(
        args.input_folder,
        team_name=args.team,
        window_size=args.window,
        support_window=args.support_window,
        required_support=args.required_support
    )

if __name__ == "__main__":
    main()
