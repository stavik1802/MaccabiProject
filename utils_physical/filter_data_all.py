"""
Script: filter_data_all.py

Description:
    This script processes and filters player tracking data from a game, handling both full-game
    players and substitutions. It splits the data into first and second half segments while
    accounting for player substitutions and game timing.

Key Features:
    - Processes GPS tracking data for all players in a game
    - Handles player substitutions by tracking sub-in and sub-out times
    - Splits data into first and second half segments
    - Identifies players who played full game vs. partial game
    - Integrates with find_ten_runners and players_lists modules for player analysis
    - Creates organized output structure with separate folders for each player

Input:
    - Input folder containing CSV files with player tracking data
    - Time ranges for first and second half
    - Player substitution information

Output:
    - Organized folder structure with player-specific subfolders
    - First half and second half CSV files for each player
    - Filtered data based on actual playing time (accounting for substitutions)

Usage:
    python filter_data_all.py <input_folder> <output_folder>
"""
import pandas as pd
from datetime import datetime
from pathlib import Path
import re

def parse_time(s):
    try:
        if '.' in s:
            secs, frac = s.strip().split('.')
            frac = (frac + '000000')[:6]
            s = f"{secs}.{frac}"
        else:
            s = s.strip() + '.000000'
        return datetime.strptime(s, "%H:%M:%S.%f").time()
    except ValueError:
        raise ValueError(f"Invalid time '{s}' – must be HH:MM:SS.fff (e.g., 18:58:00.0)")

def prompt_time_range(label):
    while True:
        try:
            start = input(f"{label} START (HH:MM:SS.fff): ").strip()
            end = input(f"{label} END   (HH:MM:SS.fff): ").strip()
            return parse_time(start), parse_time(end)
        except ValueError as e:
            print(f"❌ {e} Try again.\n")

def extract_player_id(filename):
    m = re.match(r"basic_metrics_\d{4}-\d{2}-\d{2}-([A-Z]+_\d+)-", filename)
    return m.group(1) if m else None

def main(input_folder, output_folder, first_half_start, first_half_end, second_half_start, second_half_end, top_player_ids, subs_analysis):
    """
    Process and filter player tracking data from a game.
    
    Args:
        input_folder (str): Path to input folder containing CSV files
        output_folder (str): Path to output folder for filtered data
        first_half_start (datetime.time): Start time of first half
        first_half_end (datetime.time): End time of first half
        second_half_start (datetime.time): Start time of second half
        second_half_end (datetime.time): End time of second half
        top_player_ids (list): List of player IDs who started the game
        subs_analysis (dict): Dictionary containing substitution analysis results
    """
    folder = Path(input_folder)
    output_root = Path(output_folder)
    output_root.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        print("❌ No CSV files found.")
        return

    first_half_start_dt = datetime.combine(datetime.today(), first_half_start)
    second_half_start_dt = datetime.combine(datetime.today(), second_half_start)
    print("this is the first_half_start_dt", first_half_start_dt)
    out_subs = {out_id: (minute, in_id) for out_id, in_id, minute in subs_analysis['probable_subs']}
    in_subs = {in_id: (minute, out_id) for out_id, in_id, minute in subs_analysis['probable_subs']}

    for file in csv_files:
        try:
            player_id = extract_player_id(file.name)
            if player_id is None or player_id in subs_analysis['did_not_play']:
                continue

            df = pd.read_csv(file)
            df['Time'] = df['Time'].apply(parse_time)
            df['Timestamp'] = df['Time'].apply(lambda t: datetime.combine(datetime.today(), t))

            sub_in_minute = in_subs[player_id][0] if player_id in in_subs else None
            sub_out_minute = out_subs[player_id][0] if player_id in out_subs else None

            player_folder = output_root / file.stem
            player_folder.mkdir(parents=True, exist_ok=True)

            df_first, df_second = None, None

            if player_id in subs_analysis['full_game']:
                df_first = df[(df['Timestamp'] >= first_half_start_dt) &
                              (df['Timestamp'] <= first_half_start_dt + pd.Timedelta(minutes=45))]
                df_second = df[(df['Timestamp'] >= second_half_start_dt) &
                               (df['Timestamp'] <= second_half_start_dt + pd.Timedelta(minutes=45))]

            elif sub_in_minute is not None and sub_out_minute is None:
                if sub_in_minute >= 45:
                    second_start = second_half_start_dt + pd.Timedelta(minutes=sub_in_minute - 45)
                    df_second = df[(df['Timestamp'] >= second_start) &
                                   (df['Timestamp'] <= second_half_start_dt + pd.Timedelta(minutes=45))]
                else:
                    first_start = first_half_start_dt + pd.Timedelta(minutes=sub_in_minute)
                    df_first = df[(df['Timestamp'] >= first_start) &
                                  (df['Timestamp'] <= first_half_start_dt + pd.Timedelta(minutes=45))]
                    df_second = df[(df['Timestamp'] >= second_half_start_dt) &
                                   (df['Timestamp'] <= second_half_start_dt + pd.Timedelta(minutes=45))]

            elif sub_out_minute is not None and sub_in_minute is None:
                if sub_out_minute >= 45:
                    df_first = df[(df['Timestamp'] >= first_half_start_dt) &
                                  (df['Timestamp'] <= first_half_start_dt + pd.Timedelta(minutes=45))]
                    df_second = df[(df['Timestamp'] >= second_half_start_dt) &
                                   (df['Timestamp'] <= second_half_start_dt + pd.Timedelta(minutes=sub_out_minute - 45))]
                else:
                    df_first = df[(df['Timestamp'] >= first_half_start_dt) &
                                  (df['Timestamp'] <= first_half_start_dt + pd.Timedelta(minutes=sub_out_minute))]

            if df_first is not None and not df_first.empty:
                df_first.to_csv(player_folder / "first_half.csv", index=False)
            if df_second is not None and not df_second.empty:
                df_second.to_csv(player_folder / "second_half.csv", index=False)

            print(f"✔ Processed {player_id}")

        except Exception as e:
            print(f"❌ Error with {file.name}: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python filter_data_all.py <input_folder> <output_folder>")
    else:
        # For standalone usage, prompt for time ranges
        print("📋 Enter time ranges (used for ALL files):\n")
        first_half = prompt_time_range("FIRST HALF")
        second_half = prompt_time_range("SECOND HALF")
        
        from find_ten_runners import top_runners
        from players_lists import analyze_subs
        
        first_half_start_dt = datetime.combine(datetime.today(), first_half[0])
        first_half_start_str = first_half_start_dt.strftime("%H:%M:%S.%f")
        
        top_players = top_runners(sys.argv[1], start_time_str=first_half_start_str)
        top_player_ids = [extract_player_id(name) for name in top_players if extract_player_id(name)]
        
        result = analyze_subs(sys.argv[1], "24.08.2024", top_player_ids)
        
        main(sys.argv[1], sys.argv[2], 
             first_half[0], first_half[1],
             second_half[0], second_half[1],
             top_player_ids, result)